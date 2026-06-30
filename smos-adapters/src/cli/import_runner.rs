//! `smos import` — import an opencode session transcript into SMOS memory.
//!
//! # Execution modes (forwarding)
//!
//! When `smos serve` is running and holds the RocksDB lock, the runner can
//! forward to `/v1/cli/import/opencode` on the service's HTTP API. The
//! server-side handler invokes the SAME [`ImportOpencodeSession`] use case
//! the CLI local branch invokes, and renders through the SAME
//! [`render_import_opencode_report`]. The CLI's remote branch streams the
//! response body verbatim to stdout.
//!
//! # Always-local carve-outs
//!
//! `--list` and `--dry-run` NEVER open `SurrealStore` — no lock
//! contention, no forwarding value. They skip `ExecMode` resolution
//! entirely (the caller passes `ExecMode::Local`).

use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cli::forwarding::{
    ExecMode, announce_forward, emit_lock_recovery_message, is_lock_error,
};
use crate::cli::import_helpers::{
    apply_offset_limit, derive_session_id, map_discovery_error, parse_memory_key, print_dry_run,
};
use crate::cli::tracing_setup::init_tracing_default;
use crate::config::SmosConfig;
use crate::opencode;
use crate::{OllamaEmbedding, OllamaExtractor, SurrealStore, SystemClock, TokioDelay};
use smos_application::use_cases::ImportOpencodeSession;
use smos_application::use_cases::extract_facts_from_response::MIN_INPUT_CHARS;
use smos_application::use_cases::import_opencode_session::AssistantTurn;
use smos_domain::{MemoryKey, chat::ToolCall};

/// Parsed `smos import` invocation. The `smos` binary's clap parser
/// constructs this struct so the runner does not depend on clap.
pub struct ImportArgs {
    pub session_id: Option<String>,
    pub from_file: Option<String>,
    pub memory_key: String,
    pub port: Option<u16>,
    pub agents: Vec<String>,
    pub limit: Option<usize>,
    pub offset: usize,
    pub dry_run: bool,
    pub list: bool,
}

/// Entry point: install tracing, load config, dispatch to list/dry-run/import.
///
/// `--list` and `--dry-run` are always-local (no forwarding): they never
/// open `SurrealStore`, so there is no lock contention to avoid. The
/// caller MUST pass `ExecMode::Local` when either flag is set; the
/// dispatch helper `import_opencode_should_skip_forwarding` makes that
/// decision explicit.
pub async fn run_import(config_path: &str, args: ImportArgs, mode: ExecMode) -> Result<()> {
    init_tracing_default();
    let config = SmosConfig::load(config_path)?;

    if args.list {
        return run_list(args.port).await;
    }

    let (session_id_str, transcript) = resolve_transcript(&args).await?;

    let turns = opencode::parse_transcript(&transcript);
    println!("Parsed {} assistant turns", turns.len());

    let windowed = apply_offset_limit(turns, args.offset, args.limit);
    println!("After offset/limit: {} turns to process", windowed.len());

    if args.dry_run {
        print_dry_run(&windowed);
        return Ok(());
    }

    match mode {
        ExecMode::Local => run_import_local(&config, &args, &session_id_str, windowed).await,
        ExecMode::Remote { client, base_url } => {
            announce_forward("import opencode", &base_url);
            // Clone for the remote call so the fallback path still has the
            // turns. The Vec contains Strings (cheap-ish to clone for the
            // typical 10-50 turn batch); the HTTP overhead dominates.
            match execute_remote(&client, &base_url, &args, &session_id_str, windowed.clone())
                .await?
            {
                RemoteOutcome::Body(body) => {
                    use std::io::Write as _;
                    std::io::stdout()
                        .write_all(&body)
                        .context("write import opencode result to stdout")?;
                    println!();
                    Ok(())
                }
                RemoteOutcome::EndpointNotFound => {
                    eprintln!(
                        "smos: server does not expose /v1/cli/import/opencode (older version?); \
                         falling back to local execution."
                    );
                    run_import_local(&config, &args, &session_id_str, windowed).await
                }
            }
        }
    }
}

/// Decide whether the opencode import dispatch should skip forwarding
/// entirely. `--list` and `--dry-run` never open the store → no lock
/// contention → no forwarding value. Exposed so the CLI binary's dispatch
/// can short-circuit before probing, and so the decision is unit-testable.
pub fn import_opencode_should_skip_forwarding(list: bool, dry_run: bool) -> bool {
    list || dry_run
}

/// Local opencode-import execution: open store, run pipeline, render.
/// Emits the TOCTOU lock-recovery message on lock errors. Shared by the
/// `ExecMode::Local` branch and the 404-fallback branch.
async fn run_import_local(
    config: &SmosConfig,
    args: &ImportArgs,
    session_id_str: &str,
    turns: Vec<AssistantTurn>,
) -> Result<()> {
    match execute_local(config, args, session_id_str, turns).await {
        Ok(stats) => {
            println!(
                "{}",
                render_import_opencode_report(&stats, &parse_memory_key(&args.memory_key)?)
            );
            Ok(())
        }
        Err(error) => {
            if is_lock_error(&error) {
                emit_lock_recovery_message();
            }
            Err(error)
        }
    }
}

/// Local execution: open store, build adapters, run the shared pipeline.
async fn execute_local(
    config: &SmosConfig,
    args: &ImportArgs,
    session_id_str: &str,
    turns: Vec<AssistantTurn>,
) -> Result<smos_application::use_cases::ImportStats> {
    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    let memory_key = parse_memory_key(&args.memory_key)?;
    run_import_opencode_pipeline(ImportOpencodePipelineRequest {
        store: &store,
        config,
        turns,
        memory_key: &memory_key,
        session_id_str,
        agents: &args.agents,
    })
    .await
}

/// Remote execution: POST parsed turns to `/v1/cli/import/opencode` and
/// return the raw response bytes. Returns `EndpointNotFound` on 404.
async fn execute_remote(
    client: &reqwest::Client,
    base_url: &str,
    args: &ImportArgs,
    session_id_str: &str,
    turns: Vec<AssistantTurn>,
) -> Result<RemoteOutcome> {
    let url = format!("{base_url}/v1/cli/import/opencode");
    let wire_turns: Vec<WireAssistantTurn> = turns.into_iter().map(Into::into).collect();
    let body = serde_json::to_vec(&ImportOpencodeRequest {
        turns: wire_turns,
        memory_key: args.memory_key.clone(),
        session_id: session_id_str.to_string(),
        agents: args.agents.clone(),
    })
    .context("serialise import opencode request")?;

    let response = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .with_context(|| format!("forward import opencode to {url}"))?;

    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(RemoteOutcome::EndpointNotFound);
    }
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("import opencode forwarding failed: HTTP {status}: {text}");
    }
    let bytes = response
        .bytes()
        .await
        .context("read forwarded import opencode response body")?;
    Ok(RemoteOutcome::Body(bytes))
}

enum RemoteOutcome {
    Body(bytes::Bytes),
    EndpointNotFound,
}

/// Resolve the transcript either from `--from-file` or via discovery.
/// Prints `Source: {kind_str}` ONLY in the discovery path (the
/// `--from-file` path early-returns before that line). Runs CLI-side in
/// BOTH execution-mode branches — the remote handler never re-discovers.
async fn resolve_transcript(args: &ImportArgs) -> Result<(String, serde_json::Value)> {
    if let Some(path) = &args.from_file {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("read --from-file {}", path))?;
        let value: serde_json::Value =
            serde_json::from_str(&content).with_context(|| format!("parse JSON {}", path))?;
        let id = value
            .get("info")
            .and_then(|i| i.get("id"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("imported")
            .to_string();
        return Ok((id, value));
    }

    let session_id = args
        .session_id
        .as_ref()
        .context("session_id required (or pass --from-file / --list)")?;
    let client = reqwest::Client::new();
    let source = opencode::resolve_source(&client, args.port).await;
    println!("Source: {}", source.kind_str());
    let transcript = opencode::fetch_session_export(&source, &client, session_id)
        .await
        .map_err(map_discovery_error)?;
    Ok((session_id.clone(), transcript))
}

/// Discover sessions via the chosen source and print their ids + titles.
async fn run_list(port: Option<u16>) -> Result<()> {
    let client = reqwest::Client::new();
    let source = opencode::resolve_source(&client, port).await;
    println!("Source: {}", source.kind_str());
    let sessions = opencode::list_sessions(&source, &client)
        .await
        .map_err(map_discovery_error)?;
    if sessions.is_empty() {
        println!("(no sessions found)");
        return Ok(());
    }
    for s in &sessions {
        let id = s
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let title = s
            .get("title")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        println!("{id}\t{title}");
    }
    Ok(())
}

/// Record-struct request for [`run_import_opencode_pipeline`] (ADR-0001
/// convention — no positional parameters).
pub(crate) struct ImportOpencodePipelineRequest<'a> {
    pub store: &'a SurrealStore,
    pub config: &'a SmosConfig,
    pub turns: Vec<AssistantTurn>,
    pub memory_key: &'a MemoryKey,
    pub session_id_str: &'a str,
    pub agents: &'a [String],
}

/// Shared import pipeline: build `ImportOpencodeSession`, run it, return
/// stats. Called by both the CLI local branch and the HTTP handler.
/// `pub(crate)` so the handler can reach it.
pub(crate) async fn run_import_opencode_pipeline(
    req: ImportOpencodePipelineRequest<'_>,
) -> Result<smos_application::use_cases::ImportStats> {
    let ImportOpencodePipelineRequest {
        store,
        config,
        turns,
        memory_key,
        session_id_str,
        agents,
    } = req;

    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))?;
    let extractor = OllamaExtractor::new(Arc::new(config.llm_extraction.clone()))?;
    let clock = SystemClock;
    let delay = TokioDelay;
    let session_id = derive_session_id(session_id_str);

    let import = ImportOpencodeSession {
        facts: store.clone(),
        sessions: store.clone(),
        embedder,
        extractor,
        clock,
        delay,
        confidence_cfg: Arc::new(config.confidence.clone()),
        extraction_cfg: Arc::new(config.extraction.clone()),
        enable_response_extraction: config.server.enable_response_extraction,
        min_chars: MIN_INPUT_CHARS,
    };

    let agent_filter = if agents.is_empty() {
        None
    } else {
        Some(agents)
    };

    Ok(import
        .execute(turns, memory_key, &session_id, agent_filter)
        .await?)
}

/// Render the import-complete postlude as the operator-facing stdout
/// document. `pub` because the integration test pins the render contract.
/// The returned string does NOT end with `\n`; the caller appends one via
/// `println!` (local branch) or `println!()` (remote branch after
/// `write_all`). Consistent with Pattern A used by search/finalize.
pub fn render_import_opencode_report(
    stats: &smos_application::use_cases::ImportStats,
    memory_key: &MemoryKey,
) -> String {
    format!(
        "\n=== Import complete ===\n\
         Session:      {}\n\
         Memory key:   {}\n\
         Processed:    {} turns\n\
         Skipped:      {} turns\n\
         New facts:    {}",
        stats.session_id,
        memory_key,
        stats.turns_processed,
        stats.turns_skipped,
        stats.facts_extracted
    )
}

// -----------------------------------------------------------------------
// Wire DTO for turns (avoids touching smos-application's AssistantTurn)
// -----------------------------------------------------------------------

/// Wire representation of [`AssistantTurn`] for `POST /v1/cli/import/opencode`.
/// Lives in the adapter crate (NOT in `smos-application`) to keep the
/// application layer free of serde concerns on this struct. `ToolCall` is
/// already `Serialize + Deserialize` in `smos-domain`, so it crosses the
/// wire natively.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireAssistantTurn {
    pub message_id: String,
    pub agent: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl From<AssistantTurn> for WireAssistantTurn {
    fn from(t: AssistantTurn) -> Self {
        Self {
            message_id: t.message_id,
            agent: t.agent,
            content: t.content,
            tool_calls: t.tool_calls,
        }
    }
}

impl From<WireAssistantTurn> for AssistantTurn {
    fn from(w: WireAssistantTurn) -> Self {
        Self {
            message_id: w.message_id,
            agent: w.agent,
            content: w.content,
            tool_calls: w.tool_calls,
        }
    }
}

/// Wire body for `POST /v1/cli/import/opencode`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ImportOpencodeRequest {
    pub turns: Vec<WireAssistantTurn>,
    pub memory_key: String,
    pub session_id: String,
    pub agents: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skip_forwarding_true_for_list() {
        assert!(import_opencode_should_skip_forwarding(true, false));
    }

    #[test]
    fn skip_forwarding_true_for_dry_run() {
        assert!(import_opencode_should_skip_forwarding(false, true));
    }

    #[test]
    fn skip_forwarding_false_for_real_import() {
        assert!(!import_opencode_should_skip_forwarding(false, false));
    }

    #[test]
    fn render_report_includes_all_stats_fields() {
        let stats = smos_application::use_cases::ImportStats {
            session_id: "sess_abc".into(),
            turns_processed: 10,
            turns_skipped: 2,
            facts_extracted: 7,
        };
        let mk = MemoryKey::from_raw("bob").unwrap();
        let out = render_import_opencode_report(&stats, &mk);
        assert!(out.starts_with("\n=== Import complete ===\n"));
        assert!(out.contains("Session:      sess_abc"));
        assert!(out.contains("Memory key:   bob"));
        assert!(out.contains("Processed:    10 turns"));
        assert!(out.contains("Skipped:      2 turns"));
        assert!(out.contains("New facts:    7"));
        // Pattern A: report does NOT end with \n (caller appends via println!).
        assert!(!out.ends_with('\n'));
    }

    #[test]
    fn wire_turn_roundtrips_through_assistant_turn() {
        let original = AssistantTurn {
            message_id: "msg_1".into(),
            agent: "build".into(),
            content: "hello world".into(),
            tool_calls: vec![],
        };
        let wire: WireAssistantTurn = original.clone().into();
        let back: AssistantTurn = wire.into();
        assert_eq!(back, original);
    }

    #[test]
    fn wire_turn_serialises_and_deserialises() {
        let turn = WireAssistantTurn {
            message_id: "msg_1".into(),
            agent: "build".into(),
            content: "hello".into(),
            tool_calls: vec![],
        };
        let json = serde_json::to_string(&turn).expect("serialize");
        let back: WireAssistantTurn = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.message_id, "msg_1");
        assert_eq!(back.content, "hello");
    }

    #[tokio::test]
    async fn execute_remote_maps_404_to_endpoint_not_found() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/cli/import/opencode"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let args = ImportArgs {
            session_id: Some("ses_abc".into()),
            from_file: None,
            memory_key: "bob".into(),
            port: None,
            agents: vec![],
            limit: None,
            offset: 0,
            dry_run: false,
            list: false,
        };
        let outcome = execute_remote(&client, &server.uri(), &args, "ses_abc", vec![])
            .await
            .expect("execute_remote should not hard-fail on 404");
        assert!(
            matches!(outcome, RemoteOutcome::EndpointNotFound),
            "404 must map to EndpointNotFound"
        );
    }
}
