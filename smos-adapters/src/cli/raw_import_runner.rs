//! `smos import raw <text>` — extract facts from arbitrary free-form text.
//!
//! Mirrors [`super::dir_import_runner`] but for a single text input instead
//! of a directory tree. The same [`ExtractFactsFromResponse`] pipeline runs,
//! so dedup, embedding, cross-session confirmation, and the
//! `MIN_INPUT_CHARS` floor all apply identically.
//!
//! After extraction, the runner drains this chunk's pending facts through
//! a single `FinalizeSession` run by default (NLI promotes them to
//! Accepted and detects conflicts against the accumulated Accepted pool
//! from prior chunks). Pass `--no-finalize` to skip the drain.
//!
//! # Execution modes (forwarding)
//!
//! When `smos serve` is running and holds the RocksDB lock, the runner can
//! forward to `/v1/cli/import/raw` on the service's HTTP API. The
//! server-side handler invokes the SAME extraction + optional finalize
//! pipeline the CLI local branch invokes, and renders through the SAME
//! [`render_raw_import_report`]. The CLI's remote branch streams the
//! response body verbatim to stdout.

use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::cli::forwarding::{
    ExecMode, announce_forward, emit_lock_recovery_message, is_lock_error,
};
use crate::cli::import_helpers::{derive_session_id, parse_memory_key};
use crate::cli::tracing_setup::init_tracing_default;
use crate::config::SmosConfig;
use crate::{
    NativeNliClassifier, OllamaEmbedding, OllamaExtractor, SurrealStore, SystemClock, TokioDelay,
};
use smos_application::ports::SessionRepository;
use smos_application::use_cases::{ExtractFactsFromResponse, FinalizeSession, FinalizeStats};
use smos_domain::{MemoryKey, SessionId};

/// Parsed `smos import raw` invocation. The `smos` binary's clap parser
/// constructs this struct so the runner stays clap-free.
pub struct RawImportArgs {
    pub text: String,
    pub memory_key: String,
    pub no_finalize: bool,
}

/// Wire body for `POST /v1/cli/import/raw`.
#[derive(Debug, Serialize)]
pub struct RawImportRequest {
    pub text: String,
    pub memory_key: String,
    pub no_finalize: bool,
}

/// Result of one raw-import pipeline run — shared between the CLI local
/// branch (renders to stdout) and the HTTP handler (renders to response
/// body). `pub` because the integration test pins the render contract.
pub struct RawImportResult {
    pub memory_key: MemoryKey,
    pub session_id: SessionId,
    pub new_facts: usize,
    pub finalize_stats: Option<FinalizeStats>,
}

/// Entry point: install tracing, load config, resolve execution mode, and
/// either run extraction + optional finalize locally or forward to
/// `/v1/cli/import/raw`. The render function is invoked ONCE per request —
/// on the side that owns the typed result. The remote branch streams the
/// response body verbatim.
pub async fn run_raw_import(config_path: &str, args: RawImportArgs, mode: ExecMode) -> Result<()> {
    init_tracing_default();
    let config = SmosConfig::load(config_path)?;

    match mode {
        ExecMode::Local => run_raw_import_local(&config, &args).await,
        ExecMode::Remote { client, base_url } => {
            announce_forward("import raw", &base_url);
            match execute_remote(&client, &base_url, &args).await? {
                RemoteOutcome::Body(body) => {
                    use std::io::Write as _;
                    std::io::stdout()
                        .write_all(&body)
                        .context("write raw import result to stdout")?;
                    println!();
                    Ok(())
                }
                RemoteOutcome::EndpointNotFound => {
                    eprintln!(
                        "smos: server does not expose /v1/cli/import/raw (older version?); \
                         falling back to local execution."
                    );
                    run_raw_import_local(&config, &args).await
                }
            }
        }
    }
}

/// Local raw-import execution: open store, run pipeline, render. Emits the
/// TOCTOU lock-recovery message on lock errors. Shared by the
/// `ExecMode::Local` branch and the 404-fallback branch.
async fn run_raw_import_local(config: &SmosConfig, args: &RawImportArgs) -> Result<()> {
    match execute_local(config, args).await {
        Ok(result) => {
            println!("{}", render_raw_import_report(&result, args.no_finalize));
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
async fn execute_local(config: &SmosConfig, args: &RawImportArgs) -> Result<RawImportResult> {
    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    let memory_key = parse_memory_key(&args.memory_key)?;
    let session_id = derive_session_id("raw-import");

    let classifier = if !args.no_finalize {
        Some(crate::nli::build_classifier(config).await?)
    } else {
        None
    };

    run_raw_import_pipeline(RawImportPipelineRequest {
        store: &store,
        config,
        text: &args.text,
        memory_key: &memory_key,
        session_id: &session_id,
        no_finalize: args.no_finalize,
        classifier: classifier.as_ref(),
    })
    .await
}

/// Remote execution: POST to `/v1/cli/import/raw` and return the raw
/// response bytes. Returns `EndpointNotFound` on 404 so the caller can
/// fall back to local execution.
async fn execute_remote(
    client: &reqwest::Client,
    base_url: &str,
    args: &RawImportArgs,
) -> Result<RemoteOutcome> {
    let url = format!("{base_url}/v1/cli/import/raw");
    let body = serde_json::to_vec(&RawImportRequest {
        text: args.text.clone(),
        memory_key: args.memory_key.clone(),
        no_finalize: args.no_finalize,
    })
    .context("serialise raw import request")?;

    let response = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await
        .with_context(|| format!("forward raw import to {url}"))?;

    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(RemoteOutcome::EndpointNotFound);
    }
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("raw import forwarding failed: HTTP {status}: {text}");
    }
    let bytes = response
        .bytes()
        .await
        .context("read forwarded raw import response body")?;
    Ok(RemoteOutcome::Body(bytes))
}

enum RemoteOutcome {
    Body(bytes::Bytes),
    EndpointNotFound,
}

/// Record-struct request for [`run_raw_import_pipeline`] (ADR-0001
/// convention — no positional parameters).
pub(crate) struct RawImportPipelineRequest<'a> {
    pub store: &'a SurrealStore,
    pub config: &'a SmosConfig,
    pub text: &'a str,
    pub memory_key: &'a MemoryKey,
    pub session_id: &'a SessionId,
    pub no_finalize: bool,
    pub classifier: Option<&'a NativeNliClassifier>,
}

/// Shared extraction + optional finalize pipeline. Called by both the CLI
/// local branch (with a freshly built `NativeNliClassifier`) and the HTTP
/// handler (with the shared `state.classifier`). `pub(crate)` so the
/// handler can reach it without expanding the crate's public API.
///
/// When `no_finalize` is false AND extraction produces facts AND
/// `classifier` is `None`, the pipeline returns `Err` — the caller is
/// responsible for pre-checking availability (the HTTP handler returns
/// 503 before calling this; the CLI local branch builds its own
/// classifier).
pub(crate) async fn run_raw_import_pipeline(
    req: RawImportPipelineRequest<'_>,
) -> Result<RawImportResult> {
    let RawImportPipelineRequest {
        store,
        config,
        text,
        memory_key,
        session_id,
        no_finalize,
        classifier,
    } = req;

    store.get_or_create(session_id, memory_key).await?;

    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))?;
    let extractor = OllamaExtractor::new(Arc::new(config.llm_extraction.clone()))?;
    let clock = SystemClock;
    let delay = TokioDelay;

    let use_case = ExtractFactsFromResponse {
        facts: store,
        sessions: store,
        embedder: &embedder,
        extractor: &extractor,
        clock: &clock,
        delay: &delay,
        confidence_cfg: &config.confidence,
        extraction_cfg: &config.extraction,
        enable_response_extraction: true,
    };

    let count = use_case
        .execute("", text, &[], memory_key, session_id)
        .await?;

    let finalize_stats = if !no_finalize && count > 0 {
        let classifier = classifier.ok_or_else(|| {
            anyhow::anyhow!(
                "classifier required for finalize but not available (this is a bug: the caller should pre-check)"
            )
        })?;
        let finalize = FinalizeSession {
            facts: store,
            sessions: store,
            classifier,
            confidence_cfg: &config.confidence,
            nli_cfg: &config.nli,
            merge_cfg: &config.merge,
        };
        Some(finalize.execute(session_id, memory_key).await?)
    } else {
        None
    };

    Ok(RawImportResult {
        memory_key: memory_key.clone(),
        session_id: session_id.clone(),
        new_facts: count,
        finalize_stats,
    })
}

/// Render the raw-import report as the operator-facing stdout document.
/// `pub` because the integration test pins the render contract.
///
/// Shared between the CLI local branch and the `/v1/cli/import/raw`
/// handler so the two paths produce byte-equal stdout. The returned
/// string does NOT end with `\n`; the caller appends one via `println!`
/// (local branch) or `println!()` (remote branch after `write_all`).
/// Consistent with Pattern A used by search/finalize.
pub fn render_raw_import_report(result: &RawImportResult, no_finalize: bool) -> String {
    let mut out = String::new();
    out.push_str("\n=== Raw import complete ===\n");
    out.push_str(&format!("Memory key: {}\n", result.memory_key));
    out.push_str(&format!("Session:    {}\n", result.session_id));
    out.push_str(&format!("New facts:  {}\n", result.new_facts));

    if let Some(stats) = &result.finalize_stats {
        out.push_str("\n=== Finalizing ===\n");
        out.push_str(&format_finalize_stats(stats));
        out.push('\n');
    } else if no_finalize {
        out.push_str("(finalize skipped via --no-finalize)\n");
    }

    // Pattern A: the report body keeps all internal newlines; strip only
    // the trailing one so the caller's `println!` adds exactly one.
    out.trim_end_matches('\n').to_string()
}

/// Pretty-printed JSON mirroring
/// [`crate::cli::finalize_runner::print_finalize_report`] so operators can
/// grep the same fields across `smos import raw` and `smos finalize`
/// output. Omits `memory_keys_scanned` (raw import is always scoped to one
/// key).
fn format_finalize_stats(stats: &FinalizeStats) -> String {
    let payload = serde_json::json!({
        "session_id": stats.session_id,
        "processed": stats.processed,
        "finalized": stats.finalized,
        "merged": stats.merged,
        "conflicts": stats.conflicts,
        "rejected": stats.rejected,
    });
    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| format!("{payload:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_report_includes_all_fields_when_finalized() {
        let mk = MemoryKey::from_raw("bob").unwrap();
        let sid = SessionId::from_raw("sess_0123456789ab").unwrap();
        let result = RawImportResult {
            memory_key: mk.clone(),
            session_id: sid.clone(),
            new_facts: 3,
            finalize_stats: Some(FinalizeStats {
                session_id: sid.as_str().to_string(),
                processed: 3,
                finalized: 2,
                merged: 1,
                conflicts: 0,
                rejected: 1,
            }),
        };
        let out = render_raw_import_report(&result, false);
        assert!(out.contains("=== Raw import complete ==="));
        assert!(out.contains("Memory key: bob"));
        assert!(out.contains(&format!("Session:    {sid}")));
        assert!(out.contains("New facts:  3"));
        assert!(out.contains("=== Finalizing ==="));
        assert!(out.contains("\"processed\": 3"));
    }

    #[test]
    fn render_report_shows_skip_message_when_no_finalize() {
        let mk = MemoryKey::from_raw("bob").unwrap();
        let result = RawImportResult {
            memory_key: mk,
            session_id: SessionId::from_raw("sess_0123456789ab").unwrap(),
            new_facts: 0,
            finalize_stats: None,
        };
        let out = render_raw_import_report(&result, true);
        assert!(out.contains("(finalize skipped via --no-finalize)"));
    }

    #[test]
    fn render_report_no_finalize_section_when_count_zero() {
        let mk = MemoryKey::from_raw("bob").unwrap();
        let result = RawImportResult {
            memory_key: mk,
            session_id: SessionId::from_raw("sess_0123456789ab").unwrap(),
            new_facts: 0,
            finalize_stats: None,
        };
        let out = render_raw_import_report(&result, false);
        assert!(!out.contains("=== Finalizing ==="));
        assert!(!out.contains("(finalize skipped"));
        // Pattern A: report does NOT end with \n (caller appends via println!).
        assert!(out.ends_with("New facts:  0"));
    }

    #[test]
    fn render_report_is_deterministic_across_calls() {
        let mk = MemoryKey::from_raw("bob").unwrap();
        let result = RawImportResult {
            memory_key: mk,
            session_id: SessionId::from_raw("sess_0123456789ab").unwrap(),
            new_facts: 5,
            finalize_stats: None,
        };
        let out1 = render_raw_import_report(&result, true);
        let out2 = render_raw_import_report(&result, true);
        assert_eq!(out1, out2, "render must be deterministic");
    }

    #[tokio::test]
    async fn execute_remote_maps_404_to_endpoint_not_found() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/cli/import/raw"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let args = RawImportArgs {
            text: "x".into(),
            memory_key: "bob".into(),
            no_finalize: true,
        };
        let outcome = execute_remote(&client, &server.uri(), &args)
            .await
            .expect("execute_remote should not hard-fail on 404");
        assert!(
            matches!(outcome, RemoteOutcome::EndpointNotFound),
            "404 must map to EndpointNotFound"
        );
    }
}
