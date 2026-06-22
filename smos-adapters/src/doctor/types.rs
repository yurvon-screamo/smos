//! Pure data types used by the doctor binary and its unit tests.
//!
//! The shapes here are intentionally `pub` so integration tests in
//! `tests/doctor_unit.rs` can call the formatters directly. Pure helpers
//! (matching, formatting, aggregation) live next to the types they operate
//! on so the surface stays discoverable from one module root.

use serde::{Deserialize, Serialize};

/// Outcome of a single doctor probe.
///
/// `Warn` covers infrastructure that is real but not on the chat hot path —
/// for example, partial SurrealDB stats failures, or the Ollama
/// connectivity / model-availability rows when the HTTP client itself
/// cannot be constructed. These never block `smos serve` startup directly,
/// so a warning keeps the doctor non-fatal for them while still surfacing
/// the issue. `Fail` always marks a hard dependency — an Ollama model, the
/// binary, the database, AND the reranker (every chat-completion request
/// returns HTTP 503 while the reranker is down).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl CheckStatus {
    /// Lowercase label used in both terminal output and Markdown cells so
    /// the two surfaces stay visually consistent without a separate mapping
    /// table per output mode.
    pub fn as_label(self) -> &'static str {
        match self {
            CheckStatus::Pass => "PASS",
            CheckStatus::Warn => "WARN",
            CheckStatus::Fail => "FAIL",
        }
    }

    pub fn is_pass(self) -> bool {
        matches!(self, CheckStatus::Pass)
    }

    pub fn is_warn(self) -> bool {
        matches!(self, CheckStatus::Warn)
    }

    pub fn is_fail(self) -> bool {
        matches!(self, CheckStatus::Fail)
    }
}

/// One row of the doctor report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    /// Single-line human-readable detail (URL, version, count). Multi-line
    /// details are joined with `; ` by the formatter so the Markdown table
    /// stays one cell per row.
    pub details: String,
    /// Optional remediation hint shown when the check did not pass.
    pub recommendation: Option<String>,
}

impl CheckResult {
    pub fn pass(name: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Pass,
            details: details.into(),
            recommendation: None,
        }
    }

    pub fn warn(name: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Warn,
            details: details.into(),
            recommendation: None,
        }
    }

    pub fn fail(name: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Fail,
            details: details.into(),
            recommendation: None,
        }
    }

    pub fn with_recommendation(mut self, hint: impl Into<String>) -> Self {
        self.recommendation = Some(hint.into());
        self
    }
}

/// Stats snapshot pulled from SurrealDB. Populated only when the database
/// check succeeds; otherwise the doctor report omits the stats section.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatsSnapshot {
    pub total_facts: usize,
    pub accepted: usize,
    pub pending: usize,
    pub rejected: usize,
    pub total_sessions: usize,
    /// Sessions whose `last_active` is within the configured inactivity
    /// timeout. Counts sessions that are conceptually "live" even when the
    /// server is not currently running.
    pub active_sessions: usize,
    pub ended_sessions: usize,
}

/// Aggregated totals used by the summary line and the Markdown report.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReportSummary {
    pub pass: usize,
    pub warn: usize,
    pub fail: usize,
}

impl ReportSummary {
    pub fn total(&self) -> usize {
        self.pass + self.warn + self.fail
    }

    /// `true` when the run is considered successful for smoke-test
    /// continuation. Warnings are tolerated (optional infra); any failure
    /// blocks the operator.
    pub fn is_success(&self) -> bool {
        self.fail == 0
    }
}

/// Top-level doctor report — wires metadata, individual checks, and (when
/// available) a database stats snapshot into one serialisable object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub generated_at: String,
    pub config_path: String,
    pub checks: Vec<CheckResult>,
    pub stats: Option<StatsSnapshot>,
}

impl DoctorReport {
    pub fn new(generated_at: impl Into<String>, config_path: impl Into<String>) -> Self {
        Self {
            generated_at: generated_at.into(),
            config_path: config_path.into(),
            checks: Vec::new(),
            stats: None,
        }
    }

    pub fn push(&mut self, result: CheckResult) {
        self.checks.push(result);
    }

    pub fn extend(&mut self, results: impl IntoIterator<Item = CheckResult>) {
        self.checks.extend(results);
    }

    pub fn summary(&self) -> ReportSummary {
        aggregate(&self.checks)
    }
}

/// Aggregate a slice of results into the (pass, warn, fail) counters.
pub fn aggregate(results: &[CheckResult]) -> ReportSummary {
    let mut s = ReportSummary::default();
    for r in results {
        match r.status {
            CheckStatus::Pass => s.pass += 1,
            CheckStatus::Warn => s.warn += 1,
            CheckStatus::Fail => s.fail += 1,
        }
    }
    s
}

#[cfg(test)]
mod tests {
    //! Behaviour tests for the aggregation + status helpers. Pure logic —
    //! no IO. Format-level assertions (Markdown, terminal) live in their
    //! own modules.

    use super::*;

    #[test]
    fn aggregate_counts_each_status_independently() {
        let results = vec![
            CheckResult::pass("a", ""),
            CheckResult::pass("b", ""),
            CheckResult::warn("c", ""),
            CheckResult::fail("d", ""),
        ];
        let s = aggregate(&results);
        assert_eq!(s.pass, 2);
        assert_eq!(s.warn, 1);
        assert_eq!(s.fail, 1);
        assert_eq!(s.total(), 4);
    }

    #[test]
    fn is_success_tolerates_warn_but_not_fail() {
        let with_warn = vec![CheckResult::pass("a", ""), CheckResult::warn("b", "")];
        assert!(aggregate(&with_warn).is_success());
        let with_fail = vec![CheckResult::pass("a", ""), CheckResult::fail("b", "")];
        assert!(!aggregate(&with_fail).is_success());
    }

    #[test]
    fn status_label_round_trips_uppercase_constant() {
        assert_eq!(CheckStatus::Pass.as_label(), "PASS");
        assert_eq!(CheckStatus::Warn.as_label(), "WARN");
        assert_eq!(CheckStatus::Fail.as_label(), "FAIL");
    }

    #[test]
    fn check_result_constructors_set_status_correctly() {
        assert_eq!(CheckResult::pass("n", "d").status, CheckStatus::Pass);
        assert_eq!(CheckResult::warn("n", "d").status, CheckStatus::Warn);
        assert_eq!(CheckResult::fail("n", "d").status, CheckStatus::Fail);
    }

    #[test]
    fn with_recommendation_attaches_hint() {
        let r = CheckResult::fail("n", "d").with_recommendation("pull model");
        assert_eq!(r.recommendation.as_deref(), Some("pull model"));
    }
}
