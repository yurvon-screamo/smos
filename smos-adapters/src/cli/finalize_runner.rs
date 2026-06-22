//! `smos finalize` — manual single-session drain trigger.
//!
//! Smoke-test entry point that drains one session's pending facts through
//! the full NLI pipeline. After the drain completes (and only when
//! `[git].repo_url` is set), the runner exports every accepted fact in
//! each touched namespace to the configured git clone and commits + pushes
//! so two SMOS instances can share memory through git.

use anyhow::Result;

use crate::SurrealStore;
use crate::cli::tracing_setup::init_tracing_for_server;
use crate::config::SmosConfig;
use crate::git_sync::GitSyncManager;
use smos_application::ports::FactRepository;
use smos_application::use_cases::{FinalizeSession, FinalizeStats};
use smos_domain::{Fact, MemoryKey, SessionId};

/// Run a single `FinalizeSession` drain against `session_id_str` and exit.
///
/// Loads config + store + NLI classifier, wires the use case, executes it,
/// prints the stats. Used as a smoke-test entry point — the production
/// watcher (Slice-7) wraps the same use case with a polling loop instead.
///
/// `memory_key`:
/// - `Some(key)` → scoped finalize (one namespace, fast).
/// - `None` → discovery fallback: the store scans every memory_key whose
///   facts reference `session_id` and runs finalize once per key, summing
///   the stats. Slower but works when the operator does not know the
///   namespace off-hand.
pub async fn run_finalize(
    config_path: &str,
    session_id_str: &str,
    memory_key: Option<&str>,
) -> Result<()> {
    let config = SmosConfig::load(config_path)?;
    init_tracing_for_server(&config.server);

    let session_id = SessionId::from_raw(session_id_str)
        .map_err(|e| anyhow::anyhow!("invalid session id {session_id_str:?}: {e}"))?;

    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    tracing::info!(
        session = %session_id,
        memory_key = ?memory_key,
        model = %config.nli_backend.model,
        "starting finalize trigger"
    );

    let classifier = crate::nli::build_classifier(&config).await?;

    let finalize = FinalizeSession {
        facts: &store,
        sessions: &store,
        classifier: &classifier,
        confidence_cfg: &config.confidence,
        nli_cfg: &config.nli,
        merge_cfg: &config.merge,
    };

    let memory_keys = resolve_memory_keys(&store, &session_id, memory_key).await?;

    let mut aggregated = AggregatedStats::new(session_id.as_str().to_string());
    for memory_key in &memory_keys {
        let stats = finalize.execute(&session_id, memory_key).await?;
        aggregated.accumulate(&stats);
    }

    print_finalize_report(&aggregated, memory_keys.len());

    // Git-sync: export accepted facts for every scanned namespace so the
    // remote clone reflects the post-finalize state. The clone is opened
    // lazily here (not in `run_server`) so `smos finalize` stays usable
    // even when the server is not running. A git failure is logged but
    // never fatal — the finalize itself already succeeded and the operator
    // can re-run the export manually.
    if !config.git.repo_url.trim().is_empty()
        && let Err(e) = export_to_git(&config, &store, &memory_keys).await
    {
        tracing::warn!(
            error = %format!("{e:#}"),
            "git sync export failed; finalize result is still valid"
        );
    }

    tracing::info!(session = %session_id, "finalize trigger complete");
    Ok(())
}

/// Open the configured git clone and dump every accepted fact in
/// `memory_keys` to disk, then commit + push. Called only when
/// `[git].repo_url` is non-empty.
async fn export_to_git(
    config: &SmosConfig,
    store: &SurrealStore,
    memory_keys: &[MemoryKey],
) -> Result<()> {
    let mgr = GitSyncManager::open_or_clone(&config.git)?;
    let facts = collect_accepted_facts(store, memory_keys).await?;
    if facts.is_empty() {
        tracing::info!("git sync: no accepted facts to export");
        return Ok(());
    }
    mgr.export_facts(&facts)?;
    let pushed = config.git.auto_push;
    let message = format!("memory: sync {} facts", facts.len());
    mgr.commit_and_push(&message)?;
    tracing::info!(facts_exported = facts.len(), pushed, "git sync completed");
    Ok(())
}

/// Gather every accepted fact across `memory_keys` for the export step.
/// The watcher already ran FinalizeSession for the requested session, so
/// any fact that has just crossed the accept threshold is included.
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
/// the session touched (HTTP extraction does not persist SessionState,
/// so cross-namespace scans are the only recovery when the operator
/// does not name a key).
async fn resolve_memory_keys(
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

/// Pretty-printed JSON so the operator can pipe it into jq; the Debug
/// format is hard to grep in production logs.
///
/// Serialisation of a flat object cannot fail in practice (every field is
/// a primitive), but the fallback to a Debug dump keeps the smoke-test
/// output readable even if a future field shape breaks `serde_json`.
fn print_finalize_report(aggregated: &AggregatedStats, memory_keys_scanned: usize) {
    let payload = serde_json::json!({
        "session_id": aggregated.session_id,
        "memory_keys_scanned": memory_keys_scanned,
        "processed": aggregated.processed,
        "finalized": aggregated.finalized,
        "merged": aggregated.merged,
        "conflicts": aggregated.conflicts,
        "rejected": aggregated.rejected,
    });
    let json = serde_json::to_string_pretty(&payload).unwrap_or_else(|_| format!("{payload:?}"));
    println!("{json}");
}

/// Per-session aggregate of [`FinalizeStats`] across multiple memory_keys.
/// The CLI discovery fallback iterates several namespaces for one session;
/// this struct folds the per-namespace output into a single operator-facing
/// report without losing any individual counter.
struct AggregatedStats {
    session_id: String,
    processed: usize,
    finalized: usize,
    merged: usize,
    conflicts: usize,
    rejected: usize,
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
