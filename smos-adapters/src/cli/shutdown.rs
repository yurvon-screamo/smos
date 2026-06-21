//! Process signal handling for the proxy server's graceful shutdown.
//!
//! [`shutdown_signal`] resolves on the first signal received (Ctrl+C on
//! every platform; SIGTERM on unix) so the §12 drain sequence in
//! `serve_with_shutdown` stops accepting new connections in order.

/// Wait for Ctrl+C (all platforms) or SIGTERM (unix). Returns on the first
/// signal so the server stops accepting new connections and the §12 drain
/// sequence begins.
///
/// Signal-handler setup errors are logged and the future returns without
/// panicking — the proxy can still be killed via SIGKILL or `docker stop`,
/// and crashing the server we are trying to drain is the worst possible
/// response to a missing signal handler.
pub async fn shutdown_signal() {
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
