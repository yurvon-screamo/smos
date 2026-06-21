//! Provider-layer errors.
//!
//! Returned by ML model adapters (embedding, rerank, NLI). These errors model
//! flaky external services: `Unavailable` is a connection issue, `Timeout` is
//! a slow response, `InvalidResponse` is a malformed payload.

use std::time::Duration;
use thiserror::Error;

/// Errors returned by ML-provider adapters.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("provider unavailable: {0}")]
    Unavailable(String),

    #[error("provider timeout after {0:?}")]
    Timeout(Duration),

    #[error("provider request failed: {0}")]
    RequestFailed(String),

    #[error("provider response invalid: {0}")]
    InvalidResponse(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_display_includes_duration() {
        let e = ProviderError::Timeout(Duration::from_millis(750));
        assert!(e.to_string().contains("750ms"));
    }

    #[test]
    fn unavailable_display_includes_message() {
        let e = ProviderError::Unavailable("connection refused".into());
        assert!(e.to_string().contains("connection refused"));
    }
}
