//! Error hierarchy at the application/adapter boundary.
//!
//! Three independent leaves (`RepoError`, `ProviderError`, `UpstreamError`)
//! model the three distinct failure modes (persistence, ML provider, LLM
//! upstream) so adapters can return the precise shape and the umbrella
//! `UseCaseError` can wrap any of them with `#[from]`.

pub mod provider_error;
pub mod repo_error;
pub mod upstream_error;
pub mod use_case_error;

pub use provider_error::ProviderError;
pub use repo_error::RepoError;
pub use upstream_error::UpstreamError;
pub use use_case_error::UseCaseError;
