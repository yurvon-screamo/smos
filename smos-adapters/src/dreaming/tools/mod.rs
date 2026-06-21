//! rig `Tool` implementations exposed to the dreaming LLM.
//!
//! The tools are split across submodules by concern, one tool per file, so
//! every file stays under the workspace's 200-line hard limit:
//!
//! - [`shared`] — parsing helpers, fact-to-view projection, rate-limit
//!   slot acquisition, audit-privileged rehydrate.
//! - [`list_facts`], [`get_fact`], [`count_facts`] — read-only queries.
//! - [`search_facts`], [`nli_classify`] — semantic search + NLI verdict.
//! - [`update_fact`], [`merge_facts`], [`flag_conflict`], [`delete_fact`] —
//!   bounded write operations.
//! - [`write_report`] — final markdown report writer.
//!
//! Every tool is `'static + Send + Sync` so it can be moved into a rig
//! [`AgentBuilder`](rig::agent::AgentBuilder); the bounded write tools
//! additionally hold an [`std::sync::Arc<std::sync::atomic::AtomicUsize>`]
//! counter that is shared with [`super::agent::run_audit`] to tally
//! mutations per run.

pub mod count_facts;
pub mod delete_fact;
pub mod flag_conflict;
pub mod get_fact;
pub mod list_facts;
pub mod merge_facts;
pub mod nli_classify;
pub mod search_facts;
pub mod shared;
pub mod update_fact;
pub mod write_report;

// Re-exports so call sites can write `tools::ListFactsTool` instead of
// `tools::list_facts::ListFactsTool`.
pub use count_facts::{CountFactsArgs, CountFactsTool};
pub use delete_fact::{DeleteFactArgs, DeleteFactTool};
pub use flag_conflict::{FlagConflictArgs, FlagConflictTool};
pub use get_fact::{GetFactArgs, GetFactTool};
pub use list_facts::{ListFactsArgs, ListFactsTool};
pub use merge_facts::{MergeFactsArgs, MergeFactsTool};
pub use nli_classify::{NliClassifyArgs, NliClassifyTool};
pub use search_facts::{SearchFactsArgs, SearchFactsTool};
pub use update_fact::{UpdateFactArgs, UpdateFactTool};
pub use write_report::{WriteReportArgs, WriteReportTool};

use thiserror::Error;

/// Per-run mutation caps.
///
/// Each value is consumed by the bounded write tool that enforces it; see
/// [`shared::acquire_slot`] for the per-call check.
#[derive(Debug, Clone, Copy)]
pub struct AuditLimits {
    /// Maximum `delete_fact` invocations per audit run.
    pub max_deletions: usize,
    /// Maximum `merge_facts` invocations per audit run.
    pub max_merges: usize,
}

/// Unified error type returned by every dreaming tool.
///
/// Implements [`std::error::Error`] (required by `rig::tool::Tool::Error`).
/// The [`std::convert::From`] impls preserve the source error via
/// [`std::error::Error::source`] so an operator chasing a SurrealDB query
/// failure through the LLM tool boundary can still recover the original
/// `RepoError` and its underlying message.
#[derive(Debug, Error)]
pub enum ToolError {
    #[error("rate limit exceeded: {0}")]
    RateLimitExceeded(&'static str),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("repository error: {0}")]
    Repo(String),

    #[error("provider error: {0}")]
    Provider(String),

    #[error("io error: {0}")]
    Io(String),
}

impl From<smos_application::errors::RepoError> for ToolError {
    fn from(e: smos_application::errors::RepoError) -> Self {
        Self::Repo(e.to_string())
    }
}

impl From<smos_application::errors::ProviderError> for ToolError {
    fn from(e: smos_application::errors::ProviderError) -> Self {
        Self::Provider(e.to_string())
    }
}

impl From<smos_domain::DomainError> for ToolError {
    fn from(e: smos_domain::DomainError) -> Self {
        Self::InvalidInput(e.to_string())
    }
}

impl From<std::io::Error> for ToolError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_error_wraps_repo_error() {
        let repo = smos_application::errors::RepoError::NotFound("fact abc".into());
        let tool: ToolError = repo.into();
        let msg = tool.to_string();
        assert!(msg.contains("repository error"), "msg = {msg}");
        assert!(msg.contains("fact abc"), "msg = {msg}");
    }

    #[test]
    fn tool_error_wraps_provider_error() {
        let provider = smos_application::errors::ProviderError::Unavailable("conn refused".into());
        let tool: ToolError = provider.into();
        let msg = tool.to_string();
        assert!(msg.contains("provider error"), "msg = {msg}");
        assert!(msg.contains("conn refused"), "msg = {msg}");
    }

    #[test]
    fn tool_error_wraps_io_error() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing dir");
        let tool: ToolError = io.into();
        let msg = tool.to_string();
        assert!(msg.contains("io error"), "msg = {msg}");
        assert!(msg.contains("missing dir"), "msg = {msg}");
    }

    #[test]
    fn rate_limit_error_carries_canonical_label() {
        let err = ToolError::RateLimitExceeded("max_deletions_per_run");
        assert!(err.to_string().contains("max_deletions_per_run"));
    }
}
