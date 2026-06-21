//! Merge-result DTO (POC `smos/models.py:140-159`).
//!
//! Captures the outcome of an NLI-driven merge attempt between a pending fact
//! and a candidate. `reason` carries the policy decision (merged / drift /
//! skipped); `merged_fact` is present only when the candidate absorbed the
//! pending fact; `nli_result` is echoed so the caller does not need to re-run
//! the (expensive) NLI model on the same fact pair.

use smos_domain::{NliResult, entities::Fact, enums::MergeReason};

/// Outcome of an NLI-driven merge attempt.
///
/// Deliberately *not* `PartialEq`: comparing full `Fact` clones is brittle
/// (it drags in `f32` via `Confidence`/`Heat`). Tests compare fields directly,
/// mirroring the existing convention used by `MergeCandidate` in the domain.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// `true` iff `merged_fact` was updated and should be persisted back.
    pub merged: bool,
    /// Policy reason (entailment → `Merged`, contradiction → `Drift`, …).
    pub reason: MergeReason,
    /// Updated candidate fact, present only when `merged == true`.
    pub merged_fact: Option<Fact>,
    /// Verbatim NLI verdict that drove this decision.
    pub nli_result: Option<NliResult>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use smos_domain::enums::MergeReason;

    #[test]
    fn no_candidate_result_carries_reason_and_no_fact() {
        let r = MergeResult {
            merged: false,
            reason: MergeReason::NoCandidate,
            merged_fact: None,
            nli_result: None,
        };
        assert!(!r.merged);
        assert_eq!(r.reason, MergeReason::NoCandidate);
        assert!(r.merged_fact.is_none());
    }
}
