//! `ScanState` - encapsulates the four mutable accumulators of the
//! drift-priority candidate walk performed by
//! `super::FinalizeSession::resolve_one`.
//!
//! Split out of `finalize_session.rs` (R8); behaviour-preserving. The state
//! transitions exposed below are 1:1 with the pre-R8 inline locals
//! (`merge_pick`, `last_observed_nli`, `nli_observed`, and the `pool` borrow):
//! `resolve_one` still decides WHEN each transition fires, `ScanState` only
//! owns the storage so the four accumulators live behind one named struct
//! instead of four loose locals.

use smos_domain::{Fact, NliResult};

/// Drift-priority scan accumulator.
///
/// Holds the first entailment candidate picked during the walk
/// (`merge_pick`), the strongest non-merge verdict observed
/// (`last_observed_nli`), whether any real NLI verdict was observed at all
/// (`nli_observed`), and the growing comparison `pool`.
///
/// The `pool` lives inside the scan so every mutation path (merge / conflict /
/// standalone promotion) goes through one owner; the drift-priority algorithm
/// in `resolve_one` drives the transitions via the methods below.
pub(crate) struct ScanState<'a> {
    merge_pick: Option<(Fact, NliResult)>,
    last_observed_nli: Option<NliResult>,
    nli_observed: bool,
    pool: &'a mut Vec<Fact>,
}

impl<'a> ScanState<'a> {
    pub(crate) fn new(pool: &'a mut Vec<Fact>) -> Self {
        Self {
            merge_pick: None,
            last_observed_nli: None,
            nli_observed: false,
            pool,
        }
    }

    /// Record that a real NLI verdict - or an exact-match / C3-guard
    /// equivalent - was observed for at least one candidate. Drives the
    /// graceful-degradation gate: when no verdict is ever observed the pending
    /// fact stays pending.
    pub(crate) fn mark_nli_observed(&mut self) {
        self.nli_observed = true;
    }

    /// `true` once a merge candidate has been picked (first entailment).
    pub(crate) fn has_merge_pick(&self) -> bool {
        self.merge_pick.is_some()
    }

    /// Pick the first entailment candidate as the merge target. Mirrors the
    /// pre-R8 `merge_pick = Some((existing.clone(), nli))`, guarded at the
    /// call site by `!has_merge_pick()` so the first entailment wins.
    pub(crate) fn commit_merge_pick(&mut self, existing: Fact, nli: NliResult) {
        self.merge_pick = Some((existing, nli));
    }

    /// Record a non-contradiction, non-entailment-merge verdict - feeds the
    /// `no_contradiction_bonus` on the standalone promotion path.
    pub(crate) fn observe_other_verdict(&mut self, nli: NliResult) {
        self.last_observed_nli = Some(nli);
    }

    /// Move the picked merge target out of the scan (consumed once, terminally,
    /// by `apply_merge`).
    pub(crate) fn take_merge_pick(&mut self) -> Option<(Fact, NliResult)> {
        self.merge_pick.take()
    }

    /// Move the last observed non-merge verdict out of the scan (feeds
    /// `finalize_standalone`).
    pub(crate) fn take_last_observed_nli(&mut self) -> Option<NliResult> {
        self.last_observed_nli.take()
    }

    /// `true` when at least one real verdict (or C3-guard / exact-match flag)
    /// was observed during the walk.
    pub(crate) fn nli_observed(&self) -> bool {
        self.nli_observed
    }

    /// Borrow the comparison pool for a merge / conflict / standalone mutation.
    pub(crate) fn pool_mut(&mut self) -> &mut Vec<Fact> {
        self.pool
    }
}
