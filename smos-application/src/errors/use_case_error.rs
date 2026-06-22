//! Umbrella use-case error.
//!
//! Use cases depend on multiple ports simultaneously; rather than forcing each
//! call site to enumerate every error leaf, `UseCaseError` aggregates the
//! three port-specific errors (plus domain errors) via `#[from]` conversions.
//! The variants preserve the original error for inspection via `downcast_ref`
//! or pattern matching.

use thiserror::Error;

use crate::errors::{ProviderError, RepoError, UpstreamError};
use crate::helpers::person_router::RouteError;

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

    /// Routing-layer error returned by `route_request`. Mapped to 400 by
    /// the HTTP layer when the requested person is unknown or carries an
    /// unsafe name, and to 502 when the person references a provider that
    /// is missing from the `[[providers]]` array (a config-level mistake).
    #[error(transparent)]
    Route(#[from] RouteError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::person_router::PersonEntry;
    use std::collections::HashMap;

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

    #[test]
    fn route_error_converts_via_from() {
        let route_err = RouteError::UnknownPerson("ghost".into());
        let use_case: UseCaseError = route_err.into();
        assert!(matches!(
            use_case,
            UseCaseError::Route(RouteError::UnknownPerson(_))
        ));
    }

    /// Compile-time guard: the routing plumbing (`HandleChatCompletion`)
    /// expects the existence of an empty `HashMap<String, PersonEntry>`
    /// constructor path so it can build mock persons without depending on
    /// the adapter config crate.
    #[test]
    fn person_entry_map_can_be_constructed_empty() {
        let _map: HashMap<String, PersonEntry> = HashMap::new();
    }
}
