//! Upstream (LLM proxy) errors.
//!
//! Returned by the OpenAI-compatible HTTP upstream adapter. The shape mirrors
//! a typical HTTP client error surface: connection, timeout, status code +
//! body, stream error during SSE, and (de)serialisation issues.

use std::time::Duration;
use thiserror::Error;

/// Errors returned by the LLM upstream adapter.
#[derive(Debug, Error)]
pub enum UpstreamError {
    #[error("upstream connect failed: {0}")]
    ConnectFailed(String),

    #[error("upstream timeout after {0:?}")]
    Timeout(Duration),

    #[error("upstream returned {status}: {body}")]
    StatusError { status: u16, body: String },

    #[error("upstream stream error: {0}")]
    StreamError(String),

    /// Upstream returned a 2xx body that the adapter could not parse (non-JSON
    /// or malformed). Distinct from `SerializationError` (which is an SMOS-side
    /// request-encoding bug) so the HTTP layer can map it to 502 (upstream's
    /// fault) rather than 500 (our fault).
    #[error("upstream returned an unparseable body: {0}")]
    BadResponse(String),

    #[error("upstream serialization error: {0}")]
    SerializationError(String),

    /// Every provider in an [`UpstreamConfig`] multi-provider list failed.
    /// Carries the last provider's error message so the caller still sees a
    /// meaningful cause; earlier errors are logged at `WARN` by the pool's
    /// failover path before this variant is constructed.
    #[error("all upstream providers failed; last error: {0}")]
    AllProvidersFailed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_error_display_shows_code_and_body() {
        let e = UpstreamError::StatusError {
            status: 503,
            body: "service unavailable".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("503"));
        assert!(msg.contains("service unavailable"));
    }

    #[test]
    fn timeout_display_uses_debug_format() {
        let e = UpstreamError::Timeout(Duration::from_secs(5));
        assert!(e.to_string().contains("5s"));
    }
}
