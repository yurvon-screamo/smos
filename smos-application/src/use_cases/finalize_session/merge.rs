//! Merge / conflict / standalone promotion helpers for the finalize walk.
//!
//! Split out of `finalize_session.rs` (R8); behaviour-preserving. The bodies
//! below are the verbatim originals (`apply_merge`, `apply_conflict_flag`,
//! `finalize_standalone`), moved into a sibling `impl` block.

use smos_domain::enums::FactStatus;
use smos_domain::{Fact, NliResult};

use crate::log_nonfatal;
use crate::ports::{FactRepository, NliClassifier, SessionRepository};

use super::FinalizeSession;
use super::outcome::FactOutcome;

impl<'a, FR, SR, NC> FinalizeSession<'a, FR, SR, NC>
where
    FR: FactRepository,
    SR: SessionRepository,
    NC: NliClassifier,
{
    /// Apply a bidirectional drift flag between `pending` and `existing`.
    /// Status is unchanged on both sides (POC `_apply_conflict_flag`).
    pub(crate) async fn apply_conflict_flag(
        &self,
        pending: &Fact,
        existing: &Fact,
        pool: &mut Vec<Fact>,
    ) -> FactOutcome {
        let mut existing_mut = existing.clone();
        let mut pending_mut = pending.clone();
        // Encapsulate the §5.2 invariant "both facts must carry the conflict
        // link" in one call. The bidirectional helper short-circuits on the
        // first failure; in this path `flag_conflict` cannot fail because
        // `find_merge_candidates` already excluded self-matches.
        log_nonfatal!(
            existing_mut.flag_conflict_bidirectional(&mut pending_mut),
            existing = %existing_mut.id(),
            pending = %pending_mut.id(),
            "flag_conflict_bidirectional failed"
        );
        log_nonfatal!(
            self.facts.save(&existing_mut).await,
            fact = %existing_mut.id(),
            "save existing after flag failed"
        );
        if let Err(e) = self.facts.save(&pending_mut).await {
            tracing::warn!(fact = %pending_mut.id(), error = %e, "save pending after flag failed");
            // Pending twin failed to persist its flag — leave it pending so
            // the next finalize re-attempts the same scan (idempotent).
            return FactOutcome::Skipped;
        }
        // The pending twin stays pending (status unchanged). The pool does
        // NOT grow — a flagged pair should not silently become a merge
        // candidate for the next pending fact.
        pool.push(pending.clone());
        FactOutcome::Conflict
    }

    /// Merge `pending` into `existing`, then mark the pending twin `Rejected`
    /// (POC `_apply_merge`). Source sessions and conflict flags are unioned
    /// into the existing fact, then confidence is recomputed with the
    /// entailment verdict (which carries the `no_contradiction_bonus`).
    pub(crate) async fn apply_merge(
        &self,
        pending: &Fact,
        existing: &Fact,
        nli: &NliResult,
        pool: &mut Vec<Fact>,
    ) -> FactOutcome {
        let mut existing_mut = existing.clone();
        log_nonfatal!(
            existing_mut.merge_into(pending),
            fact = %existing_mut.id(),
            "merge_into failed"
        );
        log_nonfatal!(
            existing_mut.reclassify(Some(nli), self.confidence_cfg),
            fact = %existing_mut.id(),
            "reclassify(existing) failed"
        );
        if let Err(e) = self.facts.save(&existing_mut).await {
            tracing::warn!(fact = %existing_mut.id(), error = %e, "save merged existing failed");
            return FactOutcome::Skipped;
        }

        // Mark the pending twin Rejected so it stops appearing in pending
        // listings. The `ConfidenceConfig` is forwarded so the validation
        // gate's transition guards (`Pending → Rejected` is always allowed)
        // can run; the confidence value itself is carried over unchanged.
        let mut pending_mut = pending.clone();
        if let Err(e) = pending_mut.set_status_and_confidence(
            FactStatus::Rejected,
            pending_mut.confidence(),
            self.confidence_cfg,
        ) {
            tracing::warn!(fact = %pending_mut.id(), error = %e, "reject pending twin failed");
        } else if let Err(e) = self.facts.save(&pending_mut).await {
            tracing::warn!(fact = %pending_mut.id(), error = %e, "save rejected pending failed");
        }

        // The (updated) existing fact rejoins the pool so a later pending
        // fact can merge with the unioned provenance.
        pool.push(existing_mut);
        FactOutcome::Merged
    }

    /// Promote a standalone pending fact through the validation gate.
    /// `nli` is the strongest non-contradiction verdict observed during the
    /// scan (or `None` when the scan had no candidate at all) and feeds the
    /// `no_contradiction_bonus` in the confidence scorer.
    pub(crate) async fn finalize_standalone(
        &self,
        pending: &Fact,
        nli: Option<&NliResult>,
        pool: &mut Vec<Fact>,
    ) -> FactOutcome {
        let mut fact = pending.clone();
        log_nonfatal!(
            fact.reclassify(nli, self.confidence_cfg),
            fact = %fact.id(),
            "reclassify(standalone) failed"
        );
        if let Err(e) = self.facts.save(&fact).await {
            tracing::warn!(fact = %fact.id(), error = %e, "save standalone failed");
            return FactOutcome::Skipped;
        }
        // The promoted fact joins the comparison pool so a later pending
        // fact can merge with it — even if the validation gate kept it
        // `Pending` (it is still a candidate for the same-session twin).
        pool.push(fact);
        FactOutcome::Finalized
    }
}
