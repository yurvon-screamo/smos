//! SMOS Dreaming Agent — autonomous memory audit.
//!
//! The dreaming agent is an LLM (cloud or local) that periodically reviews the
//! stored facts and applies bounded mutations:
//!
//! - deletes trivial facts (SQL echoes, file paths, single-word replies);
//! - merges semantic duplicates the NLI layer missed during finalize;
//! - flags contradictions between accepted facts;
//! - writes a markdown report documenting every change.
//!
//! Every mutation runs through a [`rig::tool::Tool`] implementation that
//! enforces a per-run rate limit, so a misbehaving LLM cannot destroy the
//! memory store.
//!
//! # Layout
//!
//! - [`agent`] — `run_audit` entry point + provider dispatch.
//! - [`prompts`] — system prompt.
//! - [`report`] — `AuditReport` summary struct.
//! - [`scheduler`] — cron trigger (tokio-cron-scheduler).
//! - [`tools`] — the ten rig `Tool` impls exposed to the LLM.

pub mod agent;
pub mod prompts;
pub mod report;
pub mod scheduler;
pub mod tools;

pub use agent::{resolve_env_var, run_audit};
pub use report::AuditReport;
pub use scheduler::start_scheduler;
pub use tools::{AuditLimits, ToolError};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_env_var_passes_through_literal() {
        assert_eq!(resolve_env_var("sk-or-abcdef"), "sk-or-abcdef");
        assert_eq!(resolve_env_var(""), "");
    }

    #[test]
    fn resolve_env_var_passes_through_fragment_without_closing_brace() {
        // "${FOO" without the closing brace is not a placeholder; return
        // verbatim so a typo in the config does not silently produce an
        // empty key.
        assert_eq!(resolve_env_var("${FOO"), "${FOO");
    }

    #[test]
    fn resolve_env_var_passes_through_fragment_without_opening_pattern() {
        // "FOO}" alone is also not a placeholder.
        assert_eq!(resolve_env_var("FOO}"), "FOO}");
    }

    #[test]
    fn resolve_env_var_expands_present_env_var() {
        // SAFETY: env var mutation is process-global; this test sets a unique
        // var name to avoid colliding with any other test or code path.
        unsafe {
            std::env::set_var("SMOS_DREAMING_TEST_KEY_PRESENT", "expanded-value");
        }
        assert_eq!(
            resolve_env_var("${SMOS_DREAMING_TEST_KEY_PRESENT}"),
            "expanded-value"
        );
        // SAFETY: same serialisation guarantee — the var name is unique.
        unsafe {
            std::env::remove_var("SMOS_DREAMING_TEST_KEY_PRESENT");
        }
    }

    #[test]
    fn resolve_env_var_returns_empty_when_env_var_missing() {
        // An unset env var yields an empty string rather than a panic; the
        // rig client surfaces the auth error with a clearer downstream message.
        assert_eq!(resolve_env_var("${SMOS_DREAMING_TEST_KEY_MISSING}"), "");
    }
}
