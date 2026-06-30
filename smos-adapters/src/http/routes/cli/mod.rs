//! Internal CLI-forwarding endpoints — `/v1/cli/*`.
//!
//! These routes exist SOLELY so that a `smos` CLI process can route use-case
//! invocations through the running `smos serve` instance when the latter
//! holds the RocksDB lock. They are NOT part of the OpenAI-compatible public
//! API (`/v1/chat/completions`, `/v1/embeddings`, …) and carry the `cli`
//! prefix to make that explicit.
//!
//! # Loopback-only access (defense-in-depth)
//!
//! The CLI client refuses to forward when `server.host` is non-loopback
//! (`forwarding::should_consider_forward`). This sub-router adds a
//! SERVER-SIDE gate so that even if an operator binds `0.0.0.0` (or a tunnel
//! exposes the port), `/v1/cli/*` mutation endpoints reject non-loopback
//! peers with HTTP 403. The OpenAI-compatible routes (`/v1/chat/completions`,
//! `/health`) are intentionally NOT gated — they are the public API surface.
//!
//! # Hexagonal invariant
//!
//! Every handler here invokes the SAME use-case struct the matching CLI
//! runner invokes on its local branch, and renders the result through the
//! SAME `print_*` / `render_*` function the local branch uses. The CLI's
//! remote branch streams the response body verbatim to stdout — no
//! deserialisation, no re-render — so local and forwarded paths produce
//! byte-equal stdout.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::extract::{ConnectInfo, Request};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::post;

pub mod finalize;
pub mod import_opencode;
pub mod import_raw;
pub mod search;

use crate::http::axum_server::AppState;

/// Build the `/v1/cli` sub-router. Mounted under `/v1/cli` by
/// [`crate::http::axum_server::build_router`].
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/search", post(search::handle))
        .route("/finalize", post(finalize::handle))
        .route("/import/raw", post(import_raw::handle))
        .route("/import/opencode", post(import_opencode::handle))
        .layer(middleware::from_fn(require_loopback_peer))
}

/// Reject non-loopback peers with HTTP 403. This is a defense-in-depth gate
/// that complements the CLIENT-SIDE loopback check in
/// [`crate::cli::forwarding::should_consider_forward`]. Even if the operator
/// binds `0.0.0.0`, `/v1/cli/*` mutation endpoints (which can inject facts
/// or trigger expensive NLI inference) are not reachable from remote hosts.
async fn require_loopback_peer(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    if !addr.ip().is_loopback() {
        tracing::warn!(
            peer = %addr,
            path = %request.uri().path(),
            "rejected non-loopback peer on /v1/cli/* (defense-in-depth gate)"
        );
        return (
            axum::http::StatusCode::FORBIDDEN,
            "smos /v1/cli/* endpoints accept loopback peers only",
        )
            .into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: these tests verify the `std::net::IpAddr::is_loopback()`
    // predicate that `require_loopback_peer` delegates to. The middleware
    // function itself is exercised end-to-end by every `e2e_cli_forwarding`
    // test (all connect from loopback → pass through the gate). The 403
    // reject path is a defense-in-depth gate documented above; a full
    // non-loopback-reject test would require a second network interface.

    #[test]
    fn loopback_ip_predicate_rejects_non_loopback() {
        let non_loopback: SocketAddr = "192.168.1.10:12345".parse().unwrap();
        assert!(!non_loopback.ip().is_loopback());
    }

    #[test]
    fn loopback_ip_predicate_accepts_loopback_v4_and_v6() {
        let loopback_v4: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        let loopback_v6: SocketAddr = "[::1]:12345".parse().unwrap();
        assert!(loopback_v4.ip().is_loopback());
        assert!(loopback_v6.ip().is_loopback());
    }
}
