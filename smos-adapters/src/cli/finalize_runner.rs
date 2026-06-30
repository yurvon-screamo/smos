//! `smos finalize` — manual single-session drain trigger.
//!
//! Smoke-test entry point that drains one session's pending facts through
//! the full NLI pipeline. After the drain completes (and only when
//! `[git].repo_url` is set), the runner exports every accepted fact in
//! each touched namespace to the configured git clone and commits + pushes
//! so two SMOS instances can share memory through git.
//!
//! # Execution modes (forwarding)
//!
//! When `smos serve` is running and holds the RocksDB lock, the runner can
//! forward to `/v1/cli/finalize` on the service's HTTP API. The server-side
//! handler invokes the SAME [`FinalizeSession`] use case the local branch
//! invokes, and renders through the SAME [`print_finalize_report`]. The
//! CLI's remote branch streams the response body verbatim to stdout.

use anyhow::{Context, Result};

use crate::SurrealStore;
use crate::cli::forwarding::{
    ExecMode, announce_forward, emit_lock_recovery_message, is_lock_error,
};
use crate::cli::tracing_setup::init_tracing_for_server;
use crate::config::SmosConfig;
use crate::git_sync::GitSyncManager;
use smos_application::log_nonfatal;
use smos_application::ports::FactRepository;
use smos_application::ports::NliClassifier;
use smos_application::use_cases::{FinalizeSession, FinalizeStats};
use smos_domain::{Fact, MemoryKey, SessionId};

/// Entry point: install tracing, load config, resolve execution mode, and
/// either run [`FinalizeSession`] locally or forward to
/// `/v1/cli/finalize`. The render function is invoked ONCE per request —
/// on the side that owns the typed result (local-branch here, server
/// handler for remote). The remote branch streams the response body
/// verbatim.
pub async fn run_finalize(
    config_path: &str,
    session_id_str: &str,
    memory_key: Option<&str>,
    mode: ExecMode,
) -> Result<()> {
    let config = SmosConfig::load(config_path)?;
    init_tracing_for_server(&config.server);

    let session_id = SessionId::from_raw(session_id_str)
        .map_err(|e| anyhow::anyhow!("invalid session id {session_id_str:?}: {e}"))?;

    match mode {
        ExecMode::Local => run_finalize_local(&config, &session_id, memory_key).await,
        ExecMode::Remote { client, base_url } => {
            announce_forward("finalize", &base_url);
            match execute_remote(&client, &base_url, session_id_str, memory_key).await? {
                RemoteOutcome::Body(body) => {
                    use std::io::Write as _;
                    std::io::stdout()
                        .write_all(&body)
                        .context("write finalize result to stdout")?;
                    println!();
                    Ok(())
                }
                RemoteOutcome::EndpointNotFound => {
                    eprintln!(
                        "smos: server does not expose /v1/cli/finalize (older version?); \
                         falling back to local execution."
                    );
                    run_finalize_local(&config, &session_id, memory_key).await
                }
            }
        }
    }
}

/// Local finalize execution: open store, build classifier, run pipeline,
/// print report, optional git sync. Emits the TOCTOU lock-recovery message
/// if `SurrealStore::connect` fails on a lock error. Shared by the
/// `ExecMode::Local` branch and the 404-fallback branch.
async fn run_finalize_local(
    config: &SmosConfig,
    session_id: &SessionId,
    memory_key: Option<&str>,
) -> Result<()> {
    let store = open_store(config).await;
    let store = match store {
        Ok(s) => s,
        Err(error) => {
            if is_lock_error(&error) {
                emit_lock_recovery_message();
            }
            return Err(error);
        }
    };

    let classifier = crate::nli::build_classifier(config).await?;
    let (aggregated, keys_scanned) =
        run_finalize_pipeline(&store, &classifier, config, session_id, memory_key).await?;
    println!("{}", print_finalize_report(&aggregated, keys_scanned));
    run_git_sync_if_configured(config, &store, &aggregated.memory_keys).await;
    Ok(())
}

/// Shared finalize pipeline: resolve memory_keys, run `FinalizeSession` per
/// key, aggregate stats. Generic over `NC: NliClassifier` so the CLI local
/// branch (with `NativeNliClassifier`) and tests (with
/// `ScriptedNliClassifier`) exercise the SAME orchestration. The HTTP
/// handler also calls this helper — it is the single source of finalize
/// logic shared between the two driving adapters.
pub(crate) async fn run_finalize_pipeline<NC>(
    store: &SurrealStore,
    classifier: &NC,
    config: &SmosConfig,
    session_id: &SessionId,
    memory_key: Option<&str>,
) -> Result<(AggregatedStats, usize)>
where
    NC: NliClassifier,
{
    tracing::info!(
        session = %session_id,
        memory_key = ?memory_key,
        model = %config.nli_backend.model,
        "starting finalize trigger"
    );

    let memory_keys = resolve_memory_keys(store, session_id, memory_key).await?;
    let keys_count = memory_keys.len();

    let mut aggregated = AggregatedStats::new(session_id.as_str().to_string());
    for mk in &memory_keys {
        let finalize = FinalizeSession {
            facts: store,
            sessions: store,
            classifier,
            confidence_cfg: &config.confidence,
            nli_cfg: &config.nli,
            merge_cfg: &config.merge,
        };
        let stats = finalize.execute(session_id, mk).await?;
        aggregated.accumulate(&stats);
    }

    aggregated.memory_keys = memory_keys;
    Ok((aggregated, keys_count))
}

/// Remote execution: POST to `/v1/cli/finalize` and return the raw response
/// bytes. Returns `EndpointNotFound` on 404 so the caller can fall back to
/// local execution with a stderr notice.
async fn execute_remote(
    client: &reqwest::Client,
    base_url: &str,
    session_id: &str,
    memory_key: Option<&str>,
) -> Result<RemoteOutcome> {
    let url = format!("{base_url}/v1/cli/finalize");
    let body = serde_json::json!({
        "session_id": session_id,
        "memory_key": memory_key,
    });
    let response = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(serde_json::to_vec(&body)?)
        .send()
        .await
        .with_context(|| format!("forward finalize to {url}"))?;
    let status = response.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(RemoteOutcome::EndpointNotFound);
    }
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("finalize forwarding failed: HTTP {status}: {text}");
    }
    let bytes = response
        .bytes()
        .await
        .context("read forwarded finalize response body")?;
    Ok(RemoteOutcome::Body(bytes))
}

enum RemoteOutcome {
    Body(bytes::Bytes),
    EndpointNotFound,
}

/// Open the SurrealStore at the configured path. Shared between the local
/// branch and the 404-fallback path.
async fn open_store(config: &SmosConfig) -> Result<SurrealStore> {
    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;
    Ok(store)
}

/// Run git sync when `[git].repo_url` is configured. Fail-open via
/// `log_nonfatal!` — the finalize result is still valid even if the git
/// export fails; the operator can re-run the export manually.
async fn run_git_sync_if_configured(
    config: &SmosConfig,
    store: &SurrealStore,
    memory_keys: &[MemoryKey],
) {
    if config.git.repo_url.trim().is_empty() {
        return;
    }
    let mgr = match GitSyncManager::open_or_clone(&config.git) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                error = %format!("{e:#}"),
                "git sync manager open failed; finalize result is still valid"
            );
            return;
        }
    };
    log_nonfatal!(
        export_to_git(&mgr, store, memory_keys).await,
        "git sync export failed; finalize result is still valid"
    );
}

/// Dump every accepted fact across `memory_keys` to the git clone, then
/// commit + push. The caller is responsible for obtaining the
/// [`GitSyncManager`] (CLI branch opens its own; the HTTP handler uses a
/// shared `Arc<Mutex<GitSyncManager>>` from `AppState`).
pub(crate) async fn export_to_git(
    mgr: &GitSyncManager,
    store: &SurrealStore,
    memory_keys: &[MemoryKey],
) -> Result<()> {
    let facts = collect_accepted_facts(store, memory_keys).await?;
    if facts.is_empty() {
        tracing::info!("git sync: no accepted facts to export");
        return Ok(());
    }
    mgr.export_facts(&facts)?;
    let message = format!("memory: sync {} facts", facts.len());
    mgr.commit_and_push(&message)?;
    tracing::info!(facts_exported = facts.len(), "git sync completed");
    Ok(())
}

/// Gather every accepted fact across `memory_keys` for the export step.
async fn collect_accepted_facts(
    store: &SurrealStore,
    memory_keys: &[MemoryKey],
) -> Result<Vec<Fact>> {
    let mut all = Vec::new();
    for mk in memory_keys {
        let facts = FactRepository::list_accepted(store, mk).await?;
        all.extend(facts);
    }
    Ok(all)
}

/// Resolve the memory_key set to scan. The explicit `--memory-key` path
/// is the fast path; the discovery fallback walks every namespace that
/// the session touched.
pub(crate) async fn resolve_memory_keys(
    store: &SurrealStore,
    session_id: &SessionId,
    memory_key: Option<&str>,
) -> Result<Vec<MemoryKey>> {
    match memory_key {
        Some(raw) => {
            let mk = MemoryKey::from_raw(raw)
                .map_err(|e| anyhow::anyhow!("invalid memory key {raw:?}: {e}"))?;
            Ok(vec![mk])
        }
        None => {
            let discovered =
                FactRepository::list_memory_keys_for_session(store, session_id).await?;
            if discovered.is_empty() {
                tracing::warn!(
                    session = %session_id,
                    "no memory_key references the session; nothing to finalize"
                );
            } else {
                tracing::info!(
                    session = %session_id,
                    keys = ?discovered.iter().map(|k| k.as_str().to_string()).collect::<Vec<_>>(),
                    "discovered memory_keys for session (no --memory-key supplied)"
                );
            }
            Ok(discovered)
        }
    }
}

/// Pretty-printed JSON so the operator can pipe it into jq. `pub` because
/// the `/v1/cli/finalize` handler and the integration test share this
/// exact render to guarantee byte-equal stdout between the local and
/// forwarded paths.
pub fn print_finalize_report(aggregated: &AggregatedStats, memory_keys_scanned: usize) -> String {
    let payload = serde_json::json!({
        "session_id": aggregated.session_id,
        "memory_keys_scanned": memory_keys_scanned,
        "processed": aggregated.processed,
        "finalized": aggregated.finalized,
        "merged": aggregated.merged,
        "conflicts": aggregated.conflicts,
        "rejected": aggregated.rejected,
    });
    serde_json::to_string_pretty(&payload).unwrap_or_else(|_| format!("{payload:?}"))
}

/// Per-session aggregate of [`FinalizeStats`] across multiple memory_keys.
/// `pub` because the integration test constructs it to pin the render
/// contract; the field set mirrors the JSON shape `print_finalize_report`
/// emits.
pub struct AggregatedStats {
    pub session_id: String,
    pub processed: usize,
    pub finalized: usize,
    pub merged: usize,
    pub conflicts: usize,
    pub rejected: usize,
    pub memory_keys: Vec<MemoryKey>,
}

impl AggregatedStats {
    fn new(session_id: String) -> Self {
        Self {
            session_id,
            processed: 0,
            finalized: 0,
            merged: 0,
            conflicts: 0,
            rejected: 0,
            memory_keys: Vec::new(),
        }
    }

    fn accumulate(&mut self, stats: &FinalizeStats) {
        self.processed += stats.processed;
        self.finalized += stats.finalized;
        self.merged += stats.merged;
        self.conflicts += stats.conflicts;
        self.rejected += stats.rejected;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_finalize_report_renders_deterministic_json() {
        let agg = AggregatedStats {
            session_id: "sess_abc".into(),
            processed: 5,
            finalized: 3,
            merged: 1,
            conflicts: 1,
            rejected: 1,
            memory_keys: Vec::new(),
        };
        let out = print_finalize_report(&agg, 2);
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert_eq!(v["session_id"], "sess_abc");
        assert_eq!(v["memory_keys_scanned"], 2);
        assert_eq!(v["processed"], 5);
        assert_eq!(v["finalized"], 3);
        assert_eq!(v["merged"], 1);
        assert_eq!(v["conflicts"], 1);
        assert_eq!(v["rejected"], 1);
    }

    #[test]
    fn print_finalize_report_empty_session() {
        let agg = AggregatedStats {
            session_id: "sess_empty".into(),
            processed: 0,
            finalized: 0,
            merged: 0,
            conflicts: 0,
            rejected: 0,
            memory_keys: Vec::new(),
        };
        let out = print_finalize_report(&agg, 0);
        let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
        assert_eq!(v["processed"], 0);
        assert_eq!(v["memory_keys_scanned"], 0);
    }

    #[tokio::test]
    async fn execute_remote_maps_404_to_endpoint_not_found() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/cli/finalize"))
            .respond_with(wiremock::ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = reqwest::Client::new();
        let outcome = execute_remote(&client, &server.uri(), "sess_abc", None)
            .await
            .expect("execute_remote should not hard-fail on 404");
        assert!(
            matches!(outcome, RemoteOutcome::EndpointNotFound),
            "404 must map to EndpointNotFound"
        );
    }
}
