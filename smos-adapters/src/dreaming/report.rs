//! Audit report types and helpers.
//!
//! [`AuditReport`] is the value returned by [`super::agent::run_audit`] — a
//! compact summary of what the agent did plus the raw model response. The
//! report is built from per-tool atomic counters (deletions / merges), which
//! are accurate because each write tool increments its counter exactly once
//! per successful call.

use smos_domain::Timestamp;

/// Result of one audit run.
///
/// Carries the per-run mutation counters (incremented atomically by the write
/// tools) plus the raw LLM response. The model response is preserved verbatim
/// because it is the canonical record of *why* the agent took each action —
/// the markdown report is also persisted to disk by `write_report`, but the
/// in-memory copy is useful for tracing logs and tests.
#[derive(Debug, Clone)]
pub struct AuditReport {
    /// Number of facts successfully deleted by `delete_fact`.
    pub deletions: usize,
    /// Number of fact pairs successfully merged by `merge_facts`.
    pub merges: usize,
    /// Raw final assistant message returned by the LLM at the end of the
    /// audit conversation.
    pub response: String,
    /// Wall-clock instant captured by the injected `Clock` when the audit
    /// completed. Stored as a domain `Timestamp` so audit logs share the
    /// same time representation as every other SMOS subsystem.
    pub timestamp: Timestamp,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts() -> Timestamp {
        Timestamp::from_unix_secs(1_750_000_000).expect("valid unix secs")
    }

    #[test]
    fn audit_report_carries_counters_response_and_timestamp() {
        let timestamp = ts();
        let report = AuditReport {
            deletions: 3,
            merges: 7,
            response: "done".into(),
            timestamp,
        };
        assert_eq!(report.deletions, 3);
        assert_eq!(report.merges, 7);
        assert_eq!(report.response, "done");
        assert_eq!(report.timestamp, timestamp);
    }
}
