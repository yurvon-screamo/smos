//! `smos serve` — proxy server runner.
//!
//! Owns the §12 drain ordering (HTTP → extraction → watcher) and the
//! optional-watcher degrade behaviour.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use smos_application::helpers::person_router::{PersonEntry, ProviderEntry};
use tokio::sync::mpsc;

use crate::cli::llama_runner::spawn_llama_cpp;
use crate::cli::shutdown::shutdown_signal;
use crate::cli::tracing_setup::init_tracing_for_server;
use crate::config::SmosConfig;
use crate::dreaming::start_scheduler;
use crate::http::axum_server::{AppState, build_router, is_loopback_host, serve_with_shutdown};
use crate::nli::build_classifier;
use crate::runtime::{ExtractionSupervisor, SessionWatcher, WatcherConfig, WatcherDeps};
use crate::upstream::ReqwestUpstreamRouter;
use crate::{
    LlamaCppReranker, OllamaEmbedding, OllamaExtractor, SurrealStore, SystemClock,
    SystemIdGenerator,
};
use smos_application::ports::{Clock, IdGenerator};

/// Handle returned by [`spawn_watcher`] so [`run_server`] can drive the
/// §12 drain ordering. The watcher task + its shutdown sender live or die
/// together; `None` means no NLI backend was available so the watcher
/// never started.
///
/// The third tuple element is the shared `Arc<OnceLock<Instant>>`
/// coordination point for the §12 unified shutdown deadline (B3):
/// `run_server` populates it with the single wall-clock deadline before
/// signalling shutdown, so the watcher's `drain_all` competes with the
/// extraction drain for the SAME remaining budget instead of starting a
/// fresh full grace window (closing the pre-B3 2× grace regression).
type WatcherHandle = Option<(
    tokio::task::JoinHandle<()>,
    mpsc::Sender<()>,
    Arc<std::sync::OnceLock<tokio::time::Instant>>,
)>;

/// Handle returned by [`spawn_audit_scheduler`]. The [`JobScheduler`] is
/// held until [`run_server`] returns so the audit cron keeps firing for the
/// lifetime of the server. `None` means the audit is disabled or its
/// dependencies could not be built — in either case the HTTP server keeps
/// running.
type AuditHandle = Option<tokio_cron_scheduler::JobScheduler>;

/// Start the SMOS proxy (default `smos serve` mode).
pub async fn run_server(config_path: &str) -> Result<()> {
    // Auto-materialise ~/.smos (or $SMOS_HOME) on boot so the operator can
    // drop a binary on a fresh box and `smos serve` immediately — no
    // mandatory `smos init` step. A failure here is logged but never fatal:
    // a read-only home directory is recoverable (the operator can point
    // SMOS_HOME at a writable location) and chat completions would still
    // work as long as SurrealDB finds a writable path through its own
    // default config.
    if let Err(e) = crate::paths::ensure_smos_home() {
        tracing::warn!(
            error = %e,
            "failed to create ~/.smos (or $SMOS_HOME); startup continues but \
             config / logs / persona files may not be readable"
        );
    }

    let config = SmosConfig::load(config_path)?;
    init_tracing_for_server(&config.server);

    warn_on_insecure_config(&config);

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        providers = config.providers.len(),
        persons = config.persons.len(),
        extraction_url = %config.llm_extraction.url,
        embedding_url = %config.embedding.url,
        reranker = %config.reranker.url,
        host = %config.server.host,
        port = config.server.port,
        "starting SMOS proxy"
    );

    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    // llama.cpp auto-launch. Spawns `llama-server` for every configured
    // service (embedding / reranker / extraction) when `auto_launch = true`.
    // A startup failure is logged at WARN but never fatal — the HTTP server
    // keeps running and the operator can launch the servers by hand. The
    // manager handle is held until shutdown so spawned children survive for
    // the lifetime of the server and are killed in `shutdown_all`.
    //
    // The git-sync manager is NOT opened here. It is opened in the
    // `smos finalize` runner where FinalizeSession actually executes, so
    // the clone is touched only when there is something to export. The
    // server's own watcher does not (yet) wire git sync through — that
    // would require a SessionWatcher signature change.
    let llama_manager = spawn_llama_cpp(&config.llama_cpp).await;

    // ExtractionSupervisor is `#[derive(Clone)]` with shared `Arc` interior,
    // so both clones observe the same in-flight counter — required for the
    // §12 drain to wait on tasks spawned through the AppState clone.
    let extraction_supervisor = ExtractionSupervisor::new();

    let state = build_app_state(&config, store.clone(), extraction_supervisor.clone())?;
    let watcher_handle = spawn_watcher(&config, store.clone()).await;
    // The dreaming audit scheduler is built unconditionally so a startup
    // failure (bad cron, missing NLI backend) is logged at server boot
    // rather than the first tick. When `audit.enabled = false` (the
    // default), `spawn_audit_scheduler` returns `None` immediately.
    let audit_handle = spawn_audit_scheduler(&config, store.clone()).await;

    let router = build_router(Arc::new(state));
    let listener =
        tokio::net::TcpListener::bind((config.server.host.as_str(), config.server.port)).await?;
    tracing::info!(
        host = %config.server.host,
        port = config.server.port,
        "SMOS HTTP server listening"
    );

    let extraction_grace =
        std::time::Duration::from_secs(config.server.shutdown_extraction_grace_seconds);
    // B3 unified deadline: captured ONCE before the HTTP + extraction drain
    // starts, then shared with BOTH the extraction drain (inside
    // `serve_with_shutdown`, which computes `remaining = deadline - now` after
    // the axum graceful-shutdown phase) and the watcher drain (via the
    // `Arc<OnceLock>` in the `WatcherHandle`). The SMOS-controlled drains
    // (extraction + watcher) therefore together consume at most
    // `extraction_grace` wall-clock. The axum graceful-shutdown phase that
    // precedes them is bounded by HTTP connection keep-alive (not
    // SMOS-configurable), so operators setting K8s
    // `terminationGracePeriodSeconds` / systemd `TimeoutStopSec` should budget
    // for `keepalive_window + extraction_grace`, not `extraction_grace` alone.
    // Pre-B3, each drain started a fresh full budget and the worst-case
    // SMOS-controlled total was 2× grace.
    let shutdown_deadline = tokio::time::Instant::now() + extraction_grace;
    if let Some((_, _, ref deadline_slot)) = watcher_handle {
        // Best-effort: if the OnceLock was already set (shouldn't happen on
        // a clean shutdown — the watcher is single-shot), the `set` call
        // returns Err and we keep the original value. Logged at debug so a
        // double-shutdown attempt is visible without spamming.
        if deadline_slot.set(shutdown_deadline).is_err() {
            tracing::debug!(
                "shutdown deadline already set; keeping existing value (double shutdown signal?)"
            );
        }
    }
    serve_with_shutdown(
        listener,
        router,
        extraction_supervisor,
        shutdown_deadline,
        shutdown_signal(),
    )
    .await?;

    drain_watcher(watcher_handle).await;
    // Drop the audit scheduler explicitly so its shutdown is logged in a
    // predictable order (after watcher drain, before the final "stopped"
    // line). Dropping triggers the scheduler's internal shutdown path.
    drop(audit_handle);

    // Kill every spawned `llama-server`. Done after the audit + watcher
    // drains so a pending FinalizeSession that still talks to the
    // reranker / extractor does not see its upstream vanish mid-request.
    if let Some(mgr) = llama_manager.as_ref() {
        mgr.shutdown_all().await;
    }

    tracing::info!("SMOS proxy stopped");
    Ok(())
}

/// Emit a startup warning when the operator is about to ship a request
/// whose bearer token is the built-in placeholder, or when permissive
/// CORS meets a non-localhost bind.
fn warn_on_insecure_config(config: &SmosConfig) {
    let is_loopback = is_loopback_host(&config.server.host);

    // Inspect every configured provider's resolved api-key. A placeholder
    // key is acceptable on loopback (a local `llama-server` ignores the
    // header); on a non-localhost bind it is an outright insecure
    // configuration and gets an ERROR-level log so the operator notices
    // before going to production.
    for provider in &config.providers {
        let api_key = provider.resolve_api_key();
        if is_placeholder_key(&api_key) {
            if is_loopback {
                tracing::warn!(
                    provider = %provider.name,
                    api_key = %api_key,
                    "upstream api_key is a known placeholder; set the env var \
                     named in api_key_env before exposing the proxy on a \
                     non-localhost interface"
                );
            } else {
                tracing::error!(
                    provider = %provider.name,
                    host = %config.server.host,
                    "api_key is a known placeholder AND host is non-localhost — \
                     this is insecure. Set a real api_key before deploying."
                );
            }
        }
    }

    let is_wildcard_host = matches!(config.server.host.as_str(), "0.0.0.0" | "::" | "[::]" | "*");
    if is_wildcard_host {
        tracing::warn!(
            host = %config.server.host,
            "server.host binds to a non-localhost interface; the router ships an \
             EMPTY CORS layer (no Access-Control-Allow-* headers are emitted, so \
             browsers block cross-origin requests by default). Same-origin requests \
             and non-browser clients (curl) keep working. Add an explicit origin \
             allow-list (`[server].allowed_origins`) if browser-driven cross-origin \
             access is needed."
        );
    }
}

/// Known placeholder api_keys that MUST NOT be used outside loopback.
///
/// `placeholder` and `changeme` are the canonical stand-ins operators reach
/// for when pointing at a local `llama-server` (which ignores the key).
/// `test`, `password`, `secret`, and the `sk-test*` family are the textbook
/// examples operators copy-paste from a tutorial "just to get it running" —
/// flagging them prevents a placeholder from ending up in production.
const PLACEHOLDER_API_KEYS: &[&str] = &[
    "placeholder",
    "changeme",
    "sk-test",
    "test",
    "password",
    "secret",
    "",
];

fn is_placeholder_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    PLACEHOLDER_API_KEYS.iter().any(|p| lower == *p) || lower.starts_with("sk-test")
}

/// Wire every concrete adapter into [`AppState`] so the axum router can
/// reach storage, providers, upstream, and the extraction supervisor.
fn build_app_state(
    config: &SmosConfig,
    store: SurrealStore,
    extraction_supervisor: ExtractionSupervisor,
) -> Result<AppState> {
    let upstream = ReqwestUpstreamRouter::from_config(&config.providers)?;
    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))?;
    let reranker = LlamaCppReranker::new(Arc::new(config.reranker.clone()))?;
    let extractor = OllamaExtractor::new(Arc::new(config.llm_extraction.clone()))?;
    let clock: Arc<dyn Clock + Send + Sync> = Arc::new(SystemClock);
    let id_generator: Arc<dyn IdGenerator + Send + Sync> = Arc::new(SystemIdGenerator);
    let retrieval_cfg = Arc::new(config.retrieval.clone());
    let heat_cfg = Arc::new(config.heat.clone());
    let confidence_cfg = Arc::new(config.confidence.clone());
    let extraction_cfg = Arc::new(config.extraction.clone());

    // Pre-build the IO-free routing views once at startup so the
    // chat-completion handler does not pay per-request allocation cost.
    // Live config reload (if ever added) MUST swap these Arcs atomically
    // (e.g. via `ArcSwap`) so a request in flight observes a consistent
    // persons + providers snapshot.
    let persons_view = Arc::new(build_person_view(&config.persons));
    let providers_view = Arc::new(build_provider_view(&config.providers));

    Ok(AppState {
        config: Arc::new(config.clone()),
        store,
        embedder,
        reranker,
        extractor,
        upstream,
        clock,
        id_generator,
        retrieval_cfg,
        heat_cfg,
        confidence_cfg,
        extraction_cfg,
        extraction_supervisor,
        persons_view,
        providers_view,
    })
}

/// Project `smos::config::PersonConfig` into the IO-free
/// [`PersonEntry`] view consumed by the routing layer.
fn build_person_view(
    persons: &std::collections::HashMap<String, crate::config::PersonConfig>,
) -> HashMap<String, PersonEntry> {
    persons
        .iter()
        .map(|(name, p)| {
            (
                name.clone(),
                PersonEntry {
                    provider: p.provider.clone(),
                    model: p.model.clone(),
                    persona: p.persona.clone(),
                },
            )
        })
        .collect()
}

/// Project `smos::config::ProviderConfig` into the IO-free
/// [`ProviderEntry`] view.
fn build_provider_view(providers: &[crate::config::ProviderConfig]) -> Vec<ProviderEntry> {
    providers
        .iter()
        .map(|p| ProviderEntry {
            name: p.name.clone(),
        })
        .collect()
}

/// Spawn the NLI backend (optional) and the [`SessionWatcher`] that uses
/// it. Returns `None` when the backend failed to start so the caller can
/// keep serving HTTP without NLI — chat completions never need NLI, so a
/// failed startup degrades to "watcher disabled" rather than crashing.
async fn spawn_watcher(config: &SmosConfig, store: SurrealStore) -> WatcherHandle {
    let classifier = match build_classifier(config).await {
        Ok(c) => {
            tracing::info!(
                model = %config.nli_backend.model,
                "NLI backend started for session watcher"
            );
            c
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "NLI backend failed to start; session watcher disabled \
                 (HTTP server still serves chat completions). Restart the \
                 proxy once the model / interpreter is available."
            );
            return None;
        }
    };

    let shutdown_deadline: Arc<std::sync::OnceLock<tokio::time::Instant>> =
        Arc::new(std::sync::OnceLock::new());

    let watcher = SessionWatcher::new(
        WatcherDeps {
            facts: store.clone(),
            sessions: store.clone(),
            classifier,
        },
        Arc::new(WatcherConfig {
            confidence: Arc::new(config.confidence.clone()),
            nli: Arc::new(config.nli.clone()),
            merge: Arc::new(config.merge.clone()),
            session: Arc::new(config.session.clone()),
            server: Arc::new(config.server.clone()),
            shutdown_deadline: shutdown_deadline.clone(),
        }),
    );
    let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
    // Spawn at a concrete-type call site so the `Send` bound on
    // `tokio::spawn` discharges against `SurrealStore` +
    // `NativeNliClassifier` (both return `Send` futures).
    let handle = tokio::spawn(watcher.into_loop(shutdown_rx));
    Some((handle, shutdown_tx, shutdown_deadline))
}

/// §12 ordering step 4: stop the watcher scan loop and drain every
/// still-tracked session through FinalizeSession so pending facts reach
/// `Accepted` / `Rejected` before the process exits.
async fn drain_watcher(watcher_handle: WatcherHandle) {
    if let Some((handle, shutdown_tx, _deadline_slot)) = watcher_handle {
        // The shared deadline slot was already populated by `run_server`
        // before `serve_with_shutdown` was invoked, so by the time the
        // watcher's `drain_all` runs (after we send the shutdown signal
        // and the loop picks it up) it reads the SAME deadline the
        // extraction drain competed against — no fresh budget window.
        let _ = shutdown_tx.send(()).await;
        let _ = handle.await;
    }
}

/// Build the dreaming audit scheduler.
///
/// Returns `None` (and logs the reason) when:
/// - the audit is disabled (`config.audit.enabled = false`); or
/// - the NLI backend, embedder, or scheduler could not be built.
///
/// The HTTP server keeps running in every `None` case so chat completions
/// stay available even if the audit stack failed to start. This mirrors the
/// watcher's own degrade behaviour: a missing ML backend must never take
/// down the proxy.
async fn spawn_audit_scheduler(config: &SmosConfig, store: SurrealStore) -> AuditHandle {
    if !config.audit.enabled {
        tracing::info!("dreaming audit disabled (audit.enabled = false); scheduler not started");
        return None;
    }

    // Build a fresh NLI classifier for the audit. This intentionally does
    // NOT share the watcher's classifier: `NativeNliClassifier` is not
    // `Clone` (its `Tokenizer` is `!Clone`), and sharing would require an
    // invasive refactor of `SessionWatcher`'s generic parameter. The cost
    // is one extra ~643 MB resident model when BOTH the watcher and the
    // audit are enabled; operators with constrained memory can disable the
    // watcher OR the audit to halve the resident footprint.
    let classifier = match build_classifier(config).await {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                error = %format!("{e:#}"),
                "audit NLI backend failed to start; dreaming scheduler disabled \
                 (HTTP server keeps running). Restart the proxy once the model \
                 / interpreter is available."
            );
            return None;
        }
    };

    let embedder = match OllamaEmbedding::new(Arc::new(config.embedding.clone())) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            tracing::warn!(
                error = %format!("{e:#}"),
                "audit embedder failed to start; dreaming scheduler disabled \
                 (HTTP server keeps running)."
            );
            return None;
        }
    };

    let clock: Arc<dyn Clock + Send + Sync> = Arc::new(SystemClock);

    match start_scheduler(&config.audit, store, Arc::new(classifier), embedder, clock).await {
        Ok(sched) => Some(sched),
        Err(e) => {
            tracing::warn!(
                error = %format!("{e:#}"),
                "audit scheduler failed to start; dreaming disabled \
                 (HTTP server keeps running)."
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_loopback_host_recognises_canonical_loopback() {
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("::1"));
    }

    #[test]
    fn is_loopback_host_rejects_wildcard_and_public() {
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("192.168.0.1"));
        assert!(!is_loopback_host("smos.example.com"));
    }

    #[test]
    fn is_placeholder_key_flags_known_placeholders() {
        for k in ["placeholder", "changeme", "test", "password", "secret", ""] {
            assert!(
                is_placeholder_key(k),
                "expected {k:?} to be flagged as a placeholder"
            );
        }
    }

    #[test]
    fn is_placeholder_key_flags_sk_test_prefix() {
        assert!(is_placeholder_key("sk-test-abc"));
        assert!(is_placeholder_key("SK-TEST-UPPER"));
    }

    #[test]
    fn is_placeholder_key_passes_through_real_keys() {
        assert!(!is_placeholder_key("sk-or-1234567890abcdef"));
        assert!(!is_placeholder_key("live-key-XYZ"));
    }
}
