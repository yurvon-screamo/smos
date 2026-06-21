//! Umbrella use-case error.
//!
//! Use cases depend on multiple ports simultaneously; rather than forcing each
//! call site to enumerate every error leaf, `UseCaseError` aggregates the
//! three port-specific errors (plus domain errors) via `#[from]` conversions.
//! The variants preserve the original error for inspection via `downcast_ref`
//! or pattern matching.

use thiserror::Error;

use crate::errors::{ProviderError, RepoError, UpstreamError};

/// Top-level error returned by use cases in later slices.
#[derive(Debug, Error)]
pub enum UseCaseError {
    #[error(transparent)]
    Repo(#[from] RepoError),

    #[error(transparent)]
    Provider(#[from] ProviderError),

    #[error(transparent)]
    Upstream(#[from] UpstreamError),

    #[error(transparent)]
    Domain(#[from] smos_domain::DomainError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_error_converts_via_from() {
        let repo_err = RepoError::QueryFailed("boom".into());
        let use_case: UseCaseError = repo_err.into();
        assert!(matches!(
            use_case,
            UseCaseError::Repo(RepoError::QueryFailed(_))
        ));
    }

    #[test]
    fn provider_error_converts_via_from() {
        let provider_err = ProviderError::Unavailable("down".into());
        let use_case: UseCaseError = provider_err.into();
        assert!(matches!(
            use_case,
            UseCaseError::Provider(ProviderError::Unavailable(_))
        ));
    }

    #[test]
    fn upstream_error_converts_via_from() {
        let upstream_err = UpstreamError::ConnectFailed("refused".into());
        let use_case: UseCaseError = upstream_err.into();
        assert!(matches!(
            use_case,
            UseCaseError::Upstream(UpstreamError::ConnectFailed(_))
        ));
    }
}
