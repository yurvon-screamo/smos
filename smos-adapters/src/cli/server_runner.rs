//! `smos serve` — proxy server runner.
//!
//! Owns the §12 drain ordering (HTTP → extraction → watcher) and the
//! optional-watcher degrade behaviour.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use smos_application::helpers::person_router::{PersonEntry, ProviderEntry};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::cli::llama_runner::spawn_llama_cpp;
use crate::cli::shutdown::shutdown_signal;
use crate::cli::tracing_setup::init_tracing_for_server;
use crate::config::SmosConfig;
use crate::dreaming::start_scheduler;
use crate::http::axum_server::{AppState, build_router, is_loopback_host, serve_with_shutdown};
use crate::nli::{NativeNliClassifier, build_classifier};
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
///
/// This is the CLI entry point: it wires Ctrl+C / SIGTERM into a
/// [`CancellationToken`] and forwards the pair to
/// [`run_server_with_shutdown`] with no readiness callback. The Windows
/// service entry point calls [`run_server_with_shutdown`] directly with
/// a token that is cancelled by the SCM `Stop` control and a callback
/// that flips the SCM status to `RUNNING` once the HTTP listener is
/// bound, so the §12 drain ordering is identical regardless of how the
/// process was launched.
pub async fn run_server(config_path: &str) -> Result<()> {
    let shutdown = CancellationToken::new();
    let cli_token = shutdown.clone();
    tokio::spawn(async move {
        shutdown_signal().await;
        cli_token.cancel();
    });
    run_server_with_shutdown(config_path, shutdown, None).await
}

/// Start the SMOS proxy driven by an external shutdown trigger and an
/// optional readiness hook.
///
/// `shutdown` is cancelled by the caller — either from Ctrl+C / SIGTERM
/// (CLI mode) or from the SCM `Stop` control (Windows service mode). The
/// rest of the §12 drain sequence (HTTP → extraction → watcher) is
/// launched off `shutdown.cancelled()`, so a single cancellation point
/// drives both the axum graceful phase and the SMOS-controlled drains.
///
/// `on_ready` fires once after the HTTP listener is bound and before
/// `serve_with_shutdown` enters its accept loop — the exact moment the
/// process begins serving traffic. The Windows service mode uses it to
/// report `SERVICE_RUNNING` to SCM only after the port is live, so SCM
/// (and any `DependOnService` consumers) do not see "started" while the
/// server is still mid-init (model load, migrations, llama auto-launch).
/// CLI mode passes `None`.
pub async fn run_server_with_shutdown(
    config_path: &str,
    shutdown: CancellationToken,
    on_ready: Option<Box<dyn FnOnce() + Send>>,
) -> Result<()> {
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
    warn_on_shared_extraction_endpoint(&config);

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
    let llama_manager = spawn_llama_cpp(&config.llama_cpp).await;

    // ExtractionSupervisor is `#[derive(Clone)]` with shared `Arc` interior,
    // so both clones observe the same in-flight counter — required for the
    // §12 drain to wait on tasks spawned through the AppState clone.
    let extraction_supervisor = ExtractionSupervisor::new();

    // Build the NLI classifier ONCE and share it across the watcher, the
    // dreaming audit, and the `/v1/cli/finalize` handler. A single
    // `NativeNliClassifier` owns a ~643 MB ort Session; pre-refactor the
    // watcher and the audit each built their own, doubling the resident
    // footprint. The shared handle is `Arc<NativeNliClassifier>`; every
    // clone observes the SAME inner `Mutex<Session>`, so NLI inference is
    // still serialised exactly as `native_nli.rs` documents. A build
    // failure (model unavailable, OOM) degrades gracefully: `None`
    // propagates to watcher (disabled) + audit (disabled) + finalize
    // handler (HTTP 503); chat completions keep working — they never need
    // NLI. This mirrors the pre-refactor per-consumer degrade behavior
    // but collapses two failure points into one.
    let shared_classifier: Option<Arc<NativeNliClassifier>> = match build_classifier(&config).await
    {
        Ok(c) => {
            tracing::info!(
                model = %config.nli_backend.model,
                "NLI backend started (shared across watcher / audit / finalize)"
            );
            Some(Arc::new(c))
        }
        Err(e) => {
            tracing::warn!(
                error = %format!("{e:#}"),
                "NLI backend failed to start; watcher / audit / finalize are disabled. \
                 HTTP server keeps serving chat completions. Restart the proxy once the \
                 model / interpreter is available."
            );
            None
        }
    };

    let state = build_app_state(
        &config,
        store.clone(),
        extraction_supervisor.clone(),
        shared_classifier.clone(),
    )?;
    let watcher_handle = spawn_watcher(&config, store.clone(), shared_classifier.clone()).await;
    // The dreaming audit scheduler is built unconditionally so a startup
    // failure (bad cron, missing NLI backend) is logged at server boot
    // rather than the first tick. When `audit.enabled = false` (the
    // default), `spawn_audit_scheduler` returns `None` immediately.
    let audit_handle =
        spawn_audit_scheduler(&config, store.clone(), shared_classifier.clone()).await;

    let router = build_router(Arc::new(state));
    let listener =
        tokio::net::TcpListener::bind((config.server.host.as_str(), config.server.port)).await?;
    tracing::info!(
        host = %config.server.host,
        port = config.server.port,
        "SMOS HTTP server listening"
    );

    if let Some(notify_ready) = on_ready {
        notify_ready();
    }

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
        async move { shutdown.cancelled().await },
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

/// Warn when the extraction LLM endpoint is shared with the chat-completion
/// upstream (and, when enabled, the dreaming audit). The shared endpoint is
/// typically a single-slot `llama-server` (`-np 1`, required for draft-mtp),
/// so background extraction can occupy the only slot a chat-completion forward
/// needs. The extraction gate (`llm_extraction.max_concurrent_extractions`,
/// default 1) bounds the worst-case forward wait to one extraction duration,
/// but eliminating the contention entirely requires a dedicated extraction
/// endpoint (D-60). This warn makes the shared-slot risk visible to the
/// operator at startup rather than as an intermittent 0-byte hang under load.
fn warn_on_shared_extraction_endpoint(config: &SmosConfig) {
    let extraction = endpoint_host_port(&config.llm_extraction.url);
    if extraction.is_empty() {
        return;
    }
    let shared_providers: Vec<&str> = config
        .providers
        .iter()
        .filter(|p| endpoint_host_port(&p.url) == extraction)
        .map(|p| p.name.as_str())
        .collect();
    let audit_shares =
        config.audit.enabled && endpoint_host_port(&config.audit.local_url) == extraction;

    if shared_providers.is_empty() && !audit_shares {
        return;
    }

    // Build the "shared with …" description from the ACTUAL flags so the
    // message never claims the chat upstream is shared when only the audit is
    // (a rare config: audit enabled, audit.local_url == llm_extraction.url,
    // but no provider points there).
    let shared_with = match (!shared_providers.is_empty(), audit_shares) {
        (true, true) => "the chat upstream and the dreaming audit".to_string(),
        (true, false) => "the chat upstream".to_string(),
        (false, true) => "the dreaming audit".to_string(),
        (false, false) => String::new(),
    };

    tracing::warn!(
        extraction_endpoint = %config.llm_extraction.url,
        shared_providers = ?shared_providers,
        audit_shares,
        max_concurrent_extractions = config.llm_extraction.max_concurrent_extractions,
        "extraction LLM endpoint is shared with {shared_with}; background \
         extraction can occupy the shared slot and starve chat forwards. The \
         extraction gate caps the forward wait at one extraction duration; \
         eliminate the contention by pointing llm_extraction.url at a dedicated \
         endpoint (D-60)."
    );
}

/// Extract the `host:port` portion of a URL for endpoint-sharing comparison.
///
/// `"http://localhost:28082/v1/chat/completions"` → `"localhost:28082"`. The
/// parse strips the scheme, path, query, and any `userinfo@` prefix so two
/// URLs that point at the same socket still compare equal regardless of how
/// they are written. A URL without an explicit port (e.g. `http://localhost`)
/// keeps its bare host and will NOT match `http://localhost:28082` — that is
/// intentional (they may resolve to different ports) and the warn is
/// best-effort, not a security boundary.
fn endpoint_host_port(url: &str) -> &str {
    let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
    let authority = authority.split('?').next().unwrap_or(authority);
    // Strip `userinfo@` — keep the segment after the last `@` (the host:port).
    authority
        .rsplit_once('@')
        .map(|(_, rest)| rest)
        .unwrap_or(authority)
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
    classifier: Option<Arc<NativeNliClassifier>>,
) -> Result<AppState> {
    let upstream = ReqwestUpstreamRouter::from_config(&config.providers)?;
    let embedder = OllamaEmbedding::new(Arc::new(config.embedding.clone()))?;
    let reranker = LlamaCppReranker::new(Arc::new(config.reranker.clone()))?;
    // The extraction gate bounds concurrent background-extraction HTTP calls
    // to `llm_extraction.max_concurrent_extractions`. With the default
    // single-slot upstream (`-np 1`) shared between chat forwards and
    // extraction, this prevents queued extractions from piling up and
    // starving chat-completion forwards. CLI import paths are sequential and
    // intentionally leave the extractor ungated.
    let max_concurrent = config.llm_extraction.max_concurrent_extractions;
    let extraction_gate = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
    let extractor = OllamaExtractor::new(Arc::new(config.llm_extraction.clone()))?
        .with_slot(extraction_gate, max_concurrent);
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
        classifier,
        git_sync: tokio::sync::OnceCell::new(),
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
/// it. Returns `None` when the shared classifier was not built at startup
/// so the caller can keep serving HTTP without NLI — chat completions
/// never need NLI, so a missing backend degrades to "watcher disabled"
/// rather than crashing.
async fn spawn_watcher(
    config: &SmosConfig,
    store: SurrealStore,
    classifier: Option<Arc<NativeNliClassifier>>,
) -> WatcherHandle {
    let classifier = match classifier {
        Some(c) => c,
        None => {
            tracing::info!("session watcher disabled (NLI backend not available at startup)");
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
/// - the shared NLI classifier is unavailable; or
/// - the embedder or scheduler could not be built.
///
/// The HTTP server keeps running in every `None` case so chat completions
/// stay available even if the audit stack failed to start. This mirrors the
/// watcher's own degrade behaviour: a missing ML backend must never take
/// down the proxy.
async fn spawn_audit_scheduler(
    config: &SmosConfig,
    store: SurrealStore,
    classifier: Option<Arc<NativeNliClassifier>>,
) -> AuditHandle {
    if !config.audit.enabled {
        tracing::info!("dreaming audit disabled (audit.enabled = false); scheduler not started");
        return None;
    }

    // The shared `Arc<NativeNliClassifier>` was built once at startup; the
    // dreaming module consumes the same Session the watcher and the
    // finalize handler use — no separate ~643 MB load.
    let classifier = match classifier {
        Some(c) => c,
        None => {
            tracing::warn!(
                "audit disabled: shared NLI backend not available at startup \
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

    match start_scheduler(&config.audit, store, classifier, embedder, clock).await {
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

    #[test]
    fn endpoint_host_port_strips_scheme_and_path() {
        assert_eq!(
            endpoint_host_port("http://localhost:28082/v1/chat/completions"),
            "localhost:28082"
        );
    }

    #[test]
    fn endpoint_host_port_handles_base_url_without_path() {
        assert_eq!(
            endpoint_host_port("http://localhost:28082"),
            "localhost:28082"
        );
    }

    #[test]
    fn endpoint_host_port_strips_query_string() {
        // A query without a path must not leak into the comparison key.
        assert_eq!(
            endpoint_host_port("http://localhost:28082?x=1"),
            "localhost:28082"
        );
        assert_eq!(
            endpoint_host_port("http://localhost:28082/v1/chat?stream=true"),
            "localhost:28082"
        );
    }

    #[test]
    fn endpoint_host_port_strips_userinfo() {
        assert_eq!(
            endpoint_host_port("http://user:pass@localhost:28082/v1/chat/completions"),
            "localhost:28082"
        );
    }

    #[test]
    fn endpoint_host_port_equivalence_drives_shared_endpoint_detection() {
        // The two URLs SMOS configures (provider vs llm_extraction) point at
        // the same socket via different paths — they MUST reduce to the same
        // host:port key so the startup warn fires.
        let provider = endpoint_host_port("http://localhost:28082/v1/chat/completions");
        let extraction = endpoint_host_port("http://localhost:28082");
        assert_eq!(provider, extraction);
    }

    #[test]
    fn endpoint_host_port_keeps_bare_host_when_no_port() {
        // Best-effort: a bare host (default port) intentionally does NOT match
        // the explicit-port form — they may resolve to different sockets.
        assert_eq!(endpoint_host_port("http://localhost"), "localhost");
        assert_ne!(
            endpoint_host_port("http://localhost"),
            endpoint_host_port("http://localhost:28082")
        );
    }
}
