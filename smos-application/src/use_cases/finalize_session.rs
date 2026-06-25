//! `FinalizeSession` — session-end batch resolution pipeline (§5, §9).
//!
//! Drains a session's pending facts and resolves each one against the currently
//! accepted pool via NLI: entailment merges into the existing fact, contradiction
//! flags a bidirectional drift pair, neutral (or no candidate) promotes the
//! pending fact through the validation gate. Resolution is **drift-priority**:
//! a contradiction against a less-similar candidate must NOT be masked by a
//! neutral/entailment hit on the top candidate, so the scan walks every
//! candidate and only commits a merge after the full pass is contradiction-free.
//!
//! # Fail-open contract
//!
//! The use case NEVER raises on a per-fact failure (§9 known limitation
//! "NLI backend unavailable graceful"): any NLI / save / mutation error is
//! logged and the loop continues. Pending facts that could not be resolved
//! stay pending for the next session-end cycle. Only the outer pool-load
//! error surface propagates as `Err` (and even then the use case degrades
//! to `Ok(stats)` with `processed == 0`).
//!
//! # Session ownership — `source_sessions`, not `SessionState.pending_facts`
//!
//! Pending ownership is derived from `Fact.source_sessions`: every fact whose
//! provenance list references `session_id` is in scope. The HTTP extraction
//! path NEVER persists a `SessionState` row — it only mutates
//! `fact.source_sessions` at extraction time — so reading
//! `SessionState.pending_facts()` left real pending facts invisible to
//! finalize (the operator-facing "24 pending facts but finalize says
//! nothing to do" bug). `source_sessions` is the only durable provenance
//! signal that survives the request path; this use case is the sole reader
//! that drives resolution off it.
//!
//! The `memory_key` is supplied by the caller (CLI `--memory-key`, watcher
//! reading `SessionState.memory_key()`) because `source_sessions` does NOT
//! pin a namespace — the same `session_id` could in principle appear under
//! multiple memory_keys (e.g. after a key migration), so the caller picks the
//! scope. The CLI additionally exposes a discovery fallback
//! (`FactRepository::list_memory_keys_for_session`) that iterates every key
//! when the operator does not name one.
//!
//! # Session bookkeeping
//!
//! `owned_ids` is snapshotted BEFORE the first await so concurrent extraction
//! appends (which race the drain) survive: only the snapshotted ids are
//! removed from `pending_facts` after finalize. Fresh pending ids appended by
//! another flow during finalize are preserved for the next cycle. The
//! `remove_pending_owned` cleanup is best-effort — a missing `SessionState`
//! row (the common case on the HTTP path) makes it a no-op; a present row
//! gets its bookkeeping cleared so the watcher does not re-schedule an idle
//! session.
//!
//! See `smos-poc/smos/session_end.py::process_session_end` for the canonical
//! Python reference; this implementation mirrors `_resolve_one`,
//! `_apply_merge`, `_apply_conflict_flag`, and `_finalize_standalone`.

use smos_domain::config::NliConfig;
use smos_domain::config::{ConfidenceConfig, MergeConfig};
use smos_domain::{Fact, FactContent, FactId, MemoryKey, NliResult, SessionId};

use crate::errors::{ProviderError, UseCaseError};
use crate::log_nonfatal;
use crate::ports::{FactRepository, NliClassifier, SessionRepository};

use outcome::FactOutcome;
use scan::ScanState;

pub mod merge;
pub mod outcome;
pub mod scan;

#[cfg(test)]
mod tests;

/// Aggregate outcome counters for one finalize run.
///
/// `FinalizeStats` is the wire shape the watcher (Slice-7) and the CLI
/// `--finalize` trigger surface to operators, so every field is `pub`. The
/// `rejected` counter overlaps with `merged` (every merge rejects the pending
/// twin) — both are reported because operators want to see "how many facts
/// left the pending pool by each exit".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FinalizeStats {
    /// Id of the session that was finalized.
    pub session_id: String,
    /// Pending facts the use case attempted to resolve.
    pub processed: usize,
    /// Standalone facts promoted through the validation gate (may still be
    /// `Pending` if the validation gate rejected the promotion).
    pub finalized: usize,
    /// Pending facts merged into an existing accepted fact (entailment path).
    pub merged: usize,
    /// Pending facts whose strongest NLI verdict was a contradiction (drift).
    /// Both sides of the pair are flagged; status is unchanged.
    pub conflicts: usize,
    /// Pending facts marked `Rejected` after being absorbed into another fact.
    /// Equals `merged` after a clean run, but kept separate so a partial run
    /// (e.g. save failure between the merge save and the reject save) is
    /// visible to operators.
    pub rejected: usize,
}

/// Borrow-style bundle of every dependency the finalize pipeline needs.
///
/// Built inline at the call site (the watcher in Slice-7, or the CLI
/// `--finalize` trigger), dropped right after [`FinalizeSession::execute`]
/// returns. References keep allocation to one borrow per call.
pub struct FinalizeSession<'a, FR, SR, NC> {
    pub facts: &'a FR,
    pub sessions: &'a SR,
    pub classifier: &'a NC,
    pub confidence_cfg: &'a ConfidenceConfig,
    pub nli_cfg: &'a NliConfig,
    pub merge_cfg: &'a MergeConfig,
}

impl<'a, FR, SR, NC> FinalizeSession<'a, FR, SR, NC>
where
    FR: FactRepository,
    SR: SessionRepository,
    NC: NliClassifier,
{
    /// Resolve every pending fact owned by `session_id` within `memory_key`.
    ///
    /// Ownership is derived from `Fact.source_sessions` (see module docs):
    /// every pending fact whose provenance list contains `session_id` is in
    /// scope. Returns `Ok(stats)` even on per-fact failures; the only `Err`
    /// paths are store catastrophes that prevent reading the pending or
    /// accepted pools.
    ///
    /// `memory_key` scopes the namespace scan. Callers that already know the
    /// namespace (the watcher reading `SessionState.memory_key()`, the CLI
    /// with `--memory-key`) pass it directly; the CLI additionally exposes a
    /// discovery fallback that iterates every key when the operator does not
    /// name one.
    pub async fn execute(
        &self,
        session_id: &SessionId,
        memory_key: &MemoryKey,
    ) -> Result<FinalizeStats, UseCaseError> {
        let mut stats = FinalizeStats {
            session_id: session_id.as_str().to_string(),
            ..FinalizeStats::default()
        };

        // Step 1 — load the pending pool for this memory_key, then filter to
        // the facts whose `source_sessions` references `session_id`. The
        // HTTP extraction path never persists `SessionState`, so this is the
        // only ownership signal that survives the request path. We do NOT
        // consult `SessionState.pending_facts()` for ownership: a missing or
        // empty session row must NOT mask real pending facts (the
        // operator-facing "nothing to do" bug).
        let all_pending = self.facts.list_pending(memory_key).await?;
        let pending: Vec<Fact> = all_pending
            .into_iter()
            .filter(|f| f.source_sessions().iter().any(|s| s == session_id))
            .collect();

        if pending.is_empty() {
            tracing::info!(
                session = %session_id,
                memory_key = %memory_key,
                "finalize: no pending facts for session"
            );
            return Ok(stats);
        }

        // Step 2 — snapshot owned_ids BEFORE any await on the resolution
        // walk. Concurrent extraction may save more pending facts carrying
        // this session in `source_sessions` while we drain; those survive
        // and are picked up by the next cycle (no leak, no double-resolve).
        let owned_ids: Vec<FactId> = pending.iter().map(|f| f.id().clone()).collect();

        let accepted = self.facts.list_accepted(memory_key).await?;
        stats.processed = pending.len();
        tracing::info!(
            session = %session_id,
            memory_key = %memory_key,
            pending = pending.len(),
            accepted = accepted.len(),
            "finalizing session"
        );

        // Step 3 — drift-priority walk. The comparison pool grows as standalone
        // facts are promoted (so a later pending fact can merge with one that
        // was itself pending a moment ago); merges and conflicts consume the
        // pending twin without growing the pool.
        let mut comparison_pool: Vec<Fact> = accepted;
        for fact in &pending {
            let outcome = self.resolve_one(fact, &mut comparison_pool).await;
            self.tally(&mut stats, outcome);
        }

        // Step 4 — bookkeeping cleanup. Only the originally-owned ids are
        // removed; concurrent additions survive (see step 2 comment). This
        // is best-effort: a missing `SessionState` (the common case on the
        // HTTP extraction path) makes the call a no-op; a present row gets
        // its bookkeeping cleared so the watcher does not re-schedule an
        // idle session. A failure here is non-fatal — the session just
        // re-drains on the next finalize, which is idempotent.
        log_nonfatal!(
            self.sessions
                .remove_pending_owned(session_id, &owned_ids)
                .await,
            "session cleanup failed (non-fatal)"
        );

        tracing::info!(
            session = %session_id,
            processed = stats.processed,
            finalized = stats.finalized,
            merged = stats.merged,
            conflicts = stats.conflicts,
            skipped = stats.processed - stats.finalized - stats.merged - stats.conflicts,
            "finalize complete"
        );

        Ok(stats)
    }

    /// Resolve one pending fact against the (growing) comparison pool.
    ///
    /// Drift-priority semantics (§9):
    /// - Exact-match short-circuit returns entailment WITHOUT an NLI call.
    /// - C3 guard skips pairs already flagged as conflicting (no double-flag).
    /// - First contradiction wins immediately (flag + return). We do NOT
    ///   commit an earlier entailment candidate before the contradiction is
    ///   observed, because drift is a stronger signal than merge.
    /// - First entailment candidate becomes the merge pick, but the scan
    ///   continues so a later less-similar candidate can still surface a
    ///   contradiction.
    /// - Otherwise the pending fact is finalized standalone, carrying the
    ///   last observed (non-contradiction, non-entailment-merge) NLI verdict
    ///   for the `no_contradiction_bonus`.
    async fn resolve_one(&self, pending: &Fact, pool: &mut Vec<Fact>) -> FactOutcome {
        let candidates = pending.find_merge_candidates(pool, self.merge_cfg);
        if candidates.is_empty() {
            return self.finalize_standalone(pending, None, pool).await;
        }

        // Drift-priority scan state: the four mutable accumulators of the
        // candidate walk (AAD-5) encapsulated in `ScanState`. The transition
        // logic is 1:1 with the pre-R8 inline locals (`merge_pick`,
        // `last_observed_nli`, `nli_observed`, and the `pool` borrow).
        let mut scan = ScanState::new(pool);

        for candidate in &candidates {
            let existing = &candidate.fact;

            // C3 guard — already-flagged conflict pair. Skip the (expensive)
            // NLI call entirely; the conflict is already recorded. The pair
            // still counts as "NLI observed" because the conflict was
            // resolved by an earlier finalize cycle — without this, a
            // pending twin of an already-flagged pair would be stuck in
            // pending forever (every cycle would skip the same pair and
            // report "NLI never observed").
            if pending.conflicts_with().contains(existing.id())
                || existing.conflicts_with().contains(pending.id())
            {
                scan.mark_nli_observed();
                tracing::debug!(
                    pending = %pending.id(),
                    existing = %existing.id(),
                    "C3 guard: skip NLI for already-flagged conflict pair"
                );
                continue;
            }

            // Exact-match short-circuit — identical text is entailment by
            // definition. Avoids DeBERTa's known quirk of returning `neutral`
            // on identical pairs.
            let nli = if FactContent::text_equals_normalized(existing.content(), pending.content())
            {
                scan.mark_nli_observed();
                NliResult::exact_match_result()
            } else {
                match self
                    .classifier
                    .classify(existing.content(), pending.content())
                    .await
                {
                    Ok(nli) if nli.available => {
                        // Real verdict from the NLI backend. An
                        // `available = false` reply (the backend's own
                        // graceful-degradation placeholder) is treated as
                        // Unavailable: skip pair, do NOT bump `nli_observed`
                        // — otherwise a permanently broken backend would
                        // silently promote facts without drift detection.
                        scan.mark_nli_observed();
                        nli
                    }
                    Ok(_unavailable) => {
                        tracing::warn!(
                            pending = %pending.id(),
                            existing = %existing.id(),
                            "NLI replied with available=false; leaving pending (skip pair)"
                        );
                        continue;
                    }
                    Err(ProviderError::Unavailable(msg)) => {
                        tracing::warn!(
                            pending = %pending.id(),
                            existing = %existing.id(),
                            error = %msg,
                            "NLI unavailable; leaving pending (skip pair)"
                        );
                        // Graceful: skip this pair, keep scanning. If every
                        // pair is unavailable the pending fact stays pending.
                        continue;
                    }
                    Err(other) => {
                        tracing::warn!(
                            pending = %pending.id(),
                            existing = %existing.id(),
                            error = %other,
                            "NLI error (non-fatal, skip pair)"
                        );
                        continue;
                    }
                }
            };

            // Drift wins immediately — flag both sides bidirectionally and
            // exit. We do NOT commit any earlier merge candidate.
            if nli.is_contradiction(self.nli_cfg) {
                return self
                    .apply_conflict_flag(pending, existing, scan.pool_mut())
                    .await;
            }

            if nli.is_entailment(self.nli_cfg) && !scan.has_merge_pick() {
                scan.commit_merge_pick(existing.clone(), nli);
                // Continue scanning: a later less-similar candidate may still
                // contradict this pending fact (drift-priority walk).
            } else {
                scan.observe_other_verdict(nli);
            }
        }

        if let Some((existing, nli)) = scan.take_merge_pick() {
            return self
                .apply_merge(pending, &existing, &nli, scan.pool_mut())
                .await;
        }

        // The NLI backend never answered for any candidate → keep the fact
        // pending. We have candidates but no NLI signal; promoting would
        // silently mask a potential drift.
        if !scan.nli_observed() {
            tracing::info!(
                pending = %pending.id(),
                candidates = candidates.len(),
                "NLI never observed for any candidate; leaving pending"
            );
            return FactOutcome::Skipped;
        }

        // No merge, no conflict — promote standalone. `last_observed_nli`
        // (the strongest non-contradiction verdict we observed) feeds the
        // `no_contradiction_bonus` in the confidence scorer.
        let last_observed_nli = scan.take_last_observed_nli();
        self.finalize_standalone(pending, last_observed_nli.as_ref(), scan.pool_mut())
            .await
    }

    /// Fold a per-fact outcome into the running stats.
    fn tally(&self, stats: &mut FinalizeStats, outcome: FactOutcome) {
        match outcome {
            FactOutcome::Finalized => stats.finalized += 1,
            FactOutcome::Merged => {
                stats.merged += 1;
                stats.rejected += 1;
            }
            FactOutcome::Conflict => stats.conflicts += 1,
            FactOutcome::Skipped => {
                // Skipped facts stay pending — not tallied into any counter.
                // Detectable via `processed - finalized - merged - conflicts`.
            }
        }
    }
}
