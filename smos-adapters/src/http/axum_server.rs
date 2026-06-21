//! axum server wiring: `AppState`, router, serve + graceful shutdown.
//!
//! `AppState` owns the concrete adapters (SurrealStore, OllamaEmbedding,
//! LlamaCppReranker, ReqwestUpstream, …) plus the active config snapshots the
//! handlers need. Slice-4 wires the full chat-completion pipeline through
//! `HandleChatCompletion`; the per-request use-case struct is built inline in
//! the handler so this state remains flat and easy to assemble in tests.

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::{get, post};
use smos_application::ports::{Clock, IdGenerator};
use smos_domain::config::{ConfidenceConfig, ExtractionConfig, HeatConfig, RetrievalConfig};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::SurrealStore;
use crate::config::SmosConfig;
use crate::http::routes;
use crate::providers::{LlamaCppReranker, OllamaEmbedding, OllamaExtractor};
use crate::runtime::ExtractionSupervisor;
use crate::upstream::ReqwestUpstreamPool;

/// Shared state handed to every handler via axum's `State` extractor.
///
/// The concrete adapter types are baked in (no `dyn` traits) because the
/// `async fn` port traits are not object-safe and a single-binary proxy does
/// not need the indirection. Tests and the production binary assemble this
/// from the same concrete pieces; changing an adapter means changing this
/// struct (and its call sites) — that is the intended trade-off.
pub struct AppState {
    pub config: Arc<SmosConfig>,
    pub store: SurrealStore,
    pub embedder: OllamaEmbedding,
    pub reranker: LlamaCppReranker,
    /// Slice-5 response extractor (Ollama Qwen3.5-2B via `/api/chat`).
    pub extractor: OllamaExtractor,
    pub upstream: ReqwestUpstreamPool,
    pub clock: Arc<dyn Clock + Send + Sync>,
    /// Fresh session-id source. The domain's `SessionId::new()` constructor
    /// is `pub(crate)`; production wiring goes through this port so id
    /// generation is an explicit, mockable capability.
    pub id_generator: Arc<dyn IdGenerator + Send + Sync>,
    pub retrieval_cfg: Arc<RetrievalConfig>,
    pub heat_cfg: Arc<HeatConfig>,
    pub confidence_cfg: Arc<ConfidenceConfig>,
    /// Semantic-dedup safety net for the background extractor
    /// (`persist_facts` step 2). Owned by the shared state so every
    /// [`ResponseExtractionSpawner`] clone hands the same snapshot to the
    /// background task.
    pub extraction_cfg: Arc<ExtractionConfig>,
    /// Tracks background extraction tasks so `serve` can drain them on
    /// shutdown (see `shutdown_extraction_grace_seconds`).
    pub extraction_supervisor: ExtractionSupervisor,
}

/// Build the router with both routes, conditional CORS, and HTTP tracing.
///
/// # CORS — conditional on the bind host
///
/// The router uses `CorsLayer::permissive()` (which emits
/// `Access-Control-Allow-Origin: *`) only when the configured `host` is a
/// loopback address (`127.0.0.1`, `localhost`, `::1`). A non-localhost
/// bind gets an EMPTY `CorsLayer` — no CORS headers are emitted at all,
/// which means browsers refuse cross-origin requests against the proxy
/// by default. The operator MUST add a configurable allow-list
/// (`[server].allowed_origins`) before the proxy can be driven from a
/// browser on a different origin; until then, only same-origin requests
/// (or non-browser clients like `curl`) work.
///
/// This is the fail-secure default: the previous "mirror_request" layer
/// was permissive in disguise (`AllowOrigin::mirror_request` echoes the
/// request's `Origin` header back, which for browsers is functionally
/// equivalent to `*` and worse for credentialed requests). The empty
/// layer is the only correct default short of a real allow-list.
pub fn build_router(state: Arc<AppState>) -> Router {
    let host = state.config.server.host.as_str();
    let cors = build_cors_layer(host);
    Router::new()
        .route(
            "/v1/chat/completions",
            post(routes::chat_completions::handle),
        )
        .route("/health", get(routes::health::handle))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Construct the CORS layer appropriate for the configured bind host.
///
/// Loopback hosts get the permissive layer (browser-driven OpenAI clients
/// work without configuration); any other host gets an empty layer that
/// emits no CORS headers — the operator must add a configurable
/// allow-list before the proxy can be driven cross-origin from a browser.
pub(crate) fn build_cors_layer(host: &str) -> CorsLayer {
    if is_loopback_host(host) {
        CorsLayer::permissive()
    } else {
        tracing::warn!(
            host = %host,
            "non-localhost bind detected: CORS is disabled (no \
             Access-Control-Allow-* headers will be emitted). Add a \
             configurable origin allow-list (`[server].allowed_origins`) \
             before deploying if browser-driven cross-origin access is needed."
        );
        // An empty `CorsLayer` emits no CORS headers — browsers will block
        // cross-origin requests by default. Same-origin requests (and
        // non-browser clients like `curl`) keep working unchanged.
        CorsLayer::new()
    }
}

/// `true` when `host` is a loopback bind that is safe to pair with a
/// permissive CORS layer. Anything else (wildcard `0.0.0.0`, a public IP,
/// a hostname) needs the strict (empty) layer. Shared with
/// `cli::server_runner` so the placeholder-api_key warning uses the same
/// definition of "loopback" as the CORS decision.
pub(crate) fn is_loopback_host(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost" | "::1")
}

/// Bind `host:port` and serve the router until graceful shutdown completes.
///
/// # Shutdown drain
/// `axum::serve`'s graceful shutdown waits for in-flight HTTP connections to
/// finish (bounded by the connection keep-alive, no SMOS config field
/// controls it). After the HTTP layer drains, this function ALSO drains
/// background extraction tasks for `shutdown_extraction_grace_seconds` — so
/// a Ctrl+C does not silently cancel half-finished fact extraction (the
/// response already reached the client, so a cancelled extraction is an
/// unrecoverable loss).
pub async fn serve(
    router: Router,
    host: &str,
    port: u16,
    extraction_supervisor: ExtractionSupervisor,
    extraction_grace: Duration,
) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    tracing::info!(host, port, "SMOS HTTP server listening");
    serve_with_shutdown(
        listener,
        router,
        extraction_supervisor,
        extraction_grace,
        shutdown_signal(),
    )
    .await
}

/// Serve + drain core, parameterised over the shutdown signal so the wiring
/// (axum shutdown → supervisor drain) is integration-testable without firing
/// a real OS signal.
///
/// Public since Slice-7: the watcher-aware main binary composes this with the
/// `SessionWatcher` drain — the HTTP module still does not know the watcher
/// type, the main binary just calls `serve_with_shutdown` first and then
/// drives the watcher shutdown channel + join handle.
pub async fn serve_with_shutdown<F>(
    listener: tokio::net::TcpListener,
    router: Router,
    extraction_supervisor: ExtractionSupervisor,
    extraction_grace: Duration,
    shutdown: F,
) -> Result<(), std::io::Error>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await?;
    tracing::info!(
        ?extraction_grace,
        "draining background extraction tasks before shutdown"
    );
    extraction_supervisor.drain(extraction_grace).await;
    Ok(())
}

/// Wait for Ctrl+C (all platforms) or SIGTERM (unix). Returns on the first
/// signal so `axum::serve` stops accepting new connections and drains.
///
/// Signal-handler setup errors are logged and the future returns without
/// panicking — a missing Ctrl+C handler is a degraded mode (the proxy can
/// still be killed via SIGKILL or `docker stop`), not a reason to crash
/// the server we are trying to drain.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::error!("ctrl_c signal handler failed: {e}");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(e) => {
                tracing::error!("SIGTERM handler installation failed: {e}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Ctrl+C received, initiating graceful shutdown"),
        _ = terminate => tracing::info!("SIGTERM received, initiating graceful shutdown"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The serve()→drain wiring: after axum's graceful shutdown completes,
    /// `serve_with_shutdown` MUST drain the extraction supervisor before
    /// returning. We inject an immediate-resolving shutdown future so no OS
    /// signal is needed, spawn one tracked task, and assert it ran to
    /// completion before `serve_with_shutdown` returned.
    #[tokio::test]
    async fn serve_with_shutdown_drains_extraction_supervisor() {
        let supervisor = ExtractionSupervisor::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let tracked = counter.clone();
        supervisor.spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            tracked.fetch_add(1, Ordering::SeqCst);
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let router = Router::new();
        // Shutdown resolves immediately → axum stops accepting → drain runs.
        serve_with_shutdown(
            listener,
            router,
            supervisor,
            Duration::from_secs(2),
            async {},
        )
        .await
        .unwrap();
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "tracked extraction task must complete during drain before serve returns"
        );
    }

    // ---- build_cors_layer / is_loopback_host --------------------------

    #[test]
    fn is_loopback_host_recognises_loopback_addresses() {
        assert!(is_loopback_host("127.0.0.1"));
        assert!(is_loopback_host("localhost"));
        assert!(is_loopback_host("::1"));
    }

    #[test]
    fn is_loopback_host_rejects_wildcard_and_public_hosts() {
        assert!(!is_loopback_host("0.0.0.0"));
        assert!(!is_loopback_host("::"));
        assert!(!is_loopback_host("192.168.1.10"));
        assert!(!is_loopback_host("example.com"));
    }

    // The two `build_cors_layer` branches return CorsLayer values whose
    // internals tower-http does not expose for inspection; the
    // behavioural difference is verified end-to-end in the HTTP suite
    // (preflight `OPTIONS /v1/chat/completions` returns the matching
    // `Access-Control-Allow-Origin` header). The unit tests above pin
    // the host-classification predicate that drives the branch choice.
}
