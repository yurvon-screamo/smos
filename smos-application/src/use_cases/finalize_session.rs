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
use smos_domain::enums::FactStatus;
use smos_domain::{Fact, FactContent, FactId, MemoryKey, NliResult, SessionId};

use crate::errors::{ProviderError, UseCaseError};
use crate::ports::{FactRepository, NliClassifier, SessionRepository};

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

/// Per-fact resolution outcome. Internal to the use case; surfaced in
/// [`FinalizeStats`] via the `tally` step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FactOutcome {
    /// Pending fact was reclassified standalone (no candidate, or only neutral
    /// NLI verdicts). Status may be `Accepted` / `Pending` / `Rejected`
    /// depending on the validation gate.
    Finalized,
    /// Pending fact was merged into an existing accepted fact and the twin
    /// was marked `Rejected`.
    Merged,
    /// Pending fact drifted (contradiction) and was bidirectionally flagged
    /// against an existing fact. Status unchanged on both sides.
    Conflict,
    /// Pending fact could not be resolved (NLI unavailable, save failed, …).
    /// Stays `Pending` for the next cycle. NOT tallied into any counter so
    /// operators can detect "facts stuck in pending" via `processed - (finalized
    /// + merged + conflicts)`.
    Skipped,
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
        if let Err(e) = self
            .sessions
            .remove_pending_owned(session_id, &owned_ids)
            .await
        {
            tracing::warn!(error = %e, "session cleanup failed (non-fatal)");
        }

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

        // The first entailment candidate is the merge pick; we keep scanning
        // so a later contradiction can override it.
        let mut merge_pick: Option<(Fact, NliResult)> = None;
        // Last non-merge NLI verdict — feeds the `no_contradiction_bonus` on
        // the standalone path (POC `last_observed_nli`).
        let mut last_observed_nli: Option<NliResult> = None;
        // Did the NLI backend actually return ANY verdict for any candidate?
        // When the backend is fully unreachable, we cannot tell whether a
        // contradiction exists — keep the fact pending (graceful degradation,
        // §9). The flag also flips on an exact-match short-circuit (which is
        // a real verdict, just resolved locally).
        let mut nli_observed = false;

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
                nli_observed = true;
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
                nli_observed = true;
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
                        nli_observed = true;
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
                return self.apply_conflict_flag(pending, existing, pool).await;
            }

            if nli.is_entailment(self.nli_cfg) && merge_pick.is_none() {
                merge_pick = Some((existing.clone(), nli));
                // Continue scanning: a later less-similar candidate may still
                // contradict this pending fact (drift-priority walk).
            } else {
                last_observed_nli = Some(nli);
            }
        }

        if let Some((existing, nli)) = merge_pick {
            return self.apply_merge(pending, &existing, &nli, pool).await;
        }

        // The NLI backend never answered for any candidate → keep the fact
        // pending. We have candidates but no NLI signal; promoting would
        // silently mask a potential drift.
        if !nli_observed {
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
        self.finalize_standalone(pending, last_observed_nli.as_ref(), pool)
            .await
    }

    /// Apply a bidirectional drift flag between `pending` and `existing`.
    /// Status is unchanged on both sides (POC `_apply_conflict_flag`).
    async fn apply_conflict_flag(
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
        if let Err(e) = existing_mut.flag_conflict_bidirectional(&mut pending_mut) {
            tracing::warn!(
                existing = %existing_mut.id(),
                pending = %pending_mut.id(),
                error = %e,
                "flag_conflict_bidirectional failed"
            );
        }
        if let Err(e) = self.facts.save(&existing_mut).await {
            tracing::warn!(fact = %existing_mut.id(), error = %e, "save existing after flag failed");
        }
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
    async fn apply_merge(
        &self,
        pending: &Fact,
        existing: &Fact,
        nli: &NliResult,
        pool: &mut Vec<Fact>,
    ) -> FactOutcome {
        let mut existing_mut = existing.clone();
        if let Err(e) = existing_mut.merge_into(pending) {
            tracing::warn!(fact = %existing_mut.id(), error = %e, "merge_into failed");
        }
        if let Err(e) = existing_mut.reclassify(Some(nli), self.confidence_cfg) {
            tracing::warn!(fact = %existing_mut.id(), error = %e, "reclassify(existing) failed");
        }
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
    async fn finalize_standalone(
        &self,
        pending: &Fact,
        nli: Option<&NliResult>,
        pool: &mut Vec<Fact>,
    ) -> FactOutcome {
        let mut fact = pending.clone();
        if let Err(e) = fact.reclassify(nli, self.confidence_cfg) {
            tracing::warn!(fact = %fact.id(), error = %e, "reclassify(standalone) failed");
        }
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

#[cfg(test)]
mod tests {
    //! Classicist unit tests for `FinalizeSession`.
    //!
    //! The fakes (`InMemoryFacts`, `InMemorySessions`, `ScriptedNliClassifier`)
    //! are local to this module so the use case can be exercised without
    //! spinning up SurrealDB or a native NLI backend. E2E coverage against a
    //! real `SurrealStore` lives in `smos-adapters/tests/e2e_finalize.rs`.

    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    use smos_domain::config::{ConfidenceConfig, MergeConfig, NliConfig};
    use smos_domain::enums::NliLabel;
    use smos_domain::{
        Embedding, FactStatus, MemoryKey, NliScores, SessionId, SessionState, Timestamp,
    };

    // ---- Fakes (classicist style: in-memory state, scripted verdicts) ----

    #[derive(Default, Clone)]
    struct InMemoryFacts {
        store: std::sync::Arc<Mutex<HashMap<String, Fact>>>,
    }
    impl InMemoryFacts {
        fn seed(&self, fact: Fact) {
            self.store
                .lock()
                .unwrap()
                .insert(fact.id().as_str().to_string(), fact);
        }
        fn get_clone(&self, id: &FactId) -> Option<Fact> {
            self.store.lock().unwrap().get(id.as_str()).cloned()
        }
    }
    impl FactRepository for InMemoryFacts {
        async fn save(&self, fact: &Fact) -> Result<(), crate::errors::RepoError> {
            self.store
                .lock()
                .unwrap()
                .insert(fact.id().as_str().to_string(), fact.clone());
            Ok(())
        }
        async fn get(
            &self,
            id: &FactId,
            _mk: &MemoryKey,
        ) -> Result<Option<Fact>, crate::errors::RepoError> {
            Ok(self.get_clone(id))
        }
        async fn list_accepted(
            &self,
            _mk: &MemoryKey,
        ) -> Result<Vec<Fact>, crate::errors::RepoError> {
            Ok(self
                .store
                .lock()
                .unwrap()
                .values()
                .filter(|f| f.status() == FactStatus::Accepted)
                .cloned()
                .collect())
        }
        async fn list_pending(
            &self,
            _mk: &MemoryKey,
        ) -> Result<Vec<Fact>, crate::errors::RepoError> {
            Ok(self
                .store
                .lock()
                .unwrap()
                .values()
                .filter(|f| f.status() == FactStatus::Pending)
                .cloned()
                .collect())
        }
        async fn list_memory_keys_for_session(
            &self,
            session_id: &SessionId,
        ) -> Result<Vec<MemoryKey>, crate::errors::RepoError> {
            // Mirrors the SurrealStore implementation: scan facts for
            // `source_sessions` membership, dedupe the memory_keys in Rust
            // (insertion order preserved so the test fixture is stable).
            let mut out: Vec<MemoryKey> = Vec::new();
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            for fact in self.store.lock().unwrap().values() {
                if !fact.source_sessions().iter().any(|s| s == session_id) {
                    continue;
                }
                let mk_str = fact.memory_key().as_str().to_string();
                if seen.insert(mk_str) {
                    out.push(fact.memory_key().clone());
                }
            }
            Ok(out)
        }
        async fn search_similar(
            &self,
            _e: Vec<f32>,
            _mk: &MemoryKey,
            _l: usize,
        ) -> Result<Vec<crate::types::SearchHit>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn update_heat_batch(
            &self,
            _ids: &[FactId],
            _mk: &MemoryKey,
            _h: smos_domain::Heat,
            _t: Timestamp,
        ) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
    }

    #[derive(Default, Clone)]
    struct InMemorySessions {
        sessions: std::sync::Arc<Mutex<HashMap<String, SessionState>>>,
    }
    impl InMemorySessions {
        fn seed(&self, state: SessionState) {
            self.sessions
                .lock()
                .unwrap()
                .insert(state.id().as_str().to_string(), state);
        }
        fn pending_of(&self, id: &SessionId) -> Vec<FactId> {
            self.sessions
                .lock()
                .unwrap()
                .get(id.as_str())
                .map(|s| s.pending_facts().to_vec())
                .unwrap_or_default()
        }
    }
    impl SessionRepository for InMemorySessions {
        async fn get_or_create(
            &self,
            id: &SessionId,
            memory_key: &MemoryKey,
        ) -> Result<SessionState, crate::errors::RepoError> {
            Ok(self
                .sessions
                .lock()
                .unwrap()
                .entry(id.as_str().to_string())
                .or_insert_with(|| {
                    SessionState::new(
                        id.clone(),
                        memory_key.clone(),
                        Timestamp::from_unix_secs(0).unwrap(),
                    )
                })
                .clone())
        }
        async fn collect_expired(
            &self,
            _t: std::time::Duration,
        ) -> Result<Vec<(SessionId, SessionState)>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn snapshot_all(
            &self,
        ) -> Result<Vec<(SessionId, SessionState)>, crate::errors::RepoError> {
            Ok(self
                .sessions
                .lock()
                .unwrap()
                .iter()
                .map(|(k, v)| (SessionId::from_raw(k).unwrap(), v.clone()))
                .collect())
        }
        async fn add_pending(
            &self,
            id: &SessionId,
            fact_ids: &[FactId],
        ) -> Result<(), crate::errors::RepoError> {
            if let Some(state) = self.sessions.lock().unwrap().get_mut(id.as_str()) {
                state.add_pending(fact_ids);
            }
            Ok(())
        }
        async fn remove_pending_owned(
            &self,
            id: &SessionId,
            owned: &[FactId],
        ) -> Result<(), crate::errors::RepoError> {
            if let Some(state) = self.sessions.lock().unwrap().get_mut(id.as_str()) {
                state.remove_owned(owned);
            }
            Ok(())
        }
        async fn clear_session(&self, id: &SessionId) -> Result<(), crate::errors::RepoError> {
            self.sessions.lock().unwrap().remove(id.as_str());
            Ok(())
        }
        async fn dedup_and_mark(
            &self,
            _id: &SessionId,
            _mk: &MemoryKey,
            candidates: &[FactId],
        ) -> Result<Vec<FactId>, crate::errors::RepoError> {
            Ok(candidates.to_vec())
        }
        async fn save(
            &self,
            id: &SessionId,
            state: &SessionState,
        ) -> Result<(), crate::errors::RepoError> {
            self.sessions
                .lock()
                .unwrap()
                .insert(id.as_str().to_string(), state.clone());
            Ok(())
        }
    }

    /// Closure type used by the matcher variant of [`ScriptedNliClassifier`].
    /// Lifted into a `type` alias so clippy's `type_complexity` lint does not
    /// fire on the enum variant — the alias is also the right level of
    /// abstraction to give the contract a name.
    type NliResolver = Box<dyn Fn(&str, &str) -> Result<NliResult, ProviderError> + Send + Sync>;

    /// Scripted NLI classifier with two modes:
    /// - **FIFO**: each call pops the next verdict from the queue. Use when
    ///   the test controls the call order (e.g. exactly one candidate).
    /// - **Matcher**: each call dispatches to the closure supplied at build
    ///   time. Use when pending iteration order is not deterministic
    ///   (`HashMap` order) — the test keys verdicts on the candidate text.
    ///
    /// Both modes record every (premise, hypothesis) pair so tests can assert
    /// on the exact set of pairs the use case asked about.
    enum ScriptedNliClassifier {
        Fifo {
            verdicts: Mutex<Vec<Result<NliResult, ProviderError>>>,
            calls: Mutex<Vec<(String, String)>>,
        },
        Match {
            resolver: NliResolver,
            calls: Mutex<Vec<(String, String)>>,
        },
    }
    impl ScriptedNliClassifier {
        fn new(verdicts: Vec<Result<NliResult, ProviderError>>) -> Self {
            Self::Fifo {
                verdicts: Mutex::new(verdicts),
                calls: Mutex::new(Vec::new()),
            }
        }
        fn matching<F>(resolver: F) -> Self
        where
            F: Fn(&str, &str) -> Result<NliResult, ProviderError> + Send + Sync + 'static,
        {
            Self::Match {
                resolver: Box::new(resolver),
                calls: Mutex::new(Vec::new()),
            }
        }
        fn calls(&self) -> Vec<(String, String)> {
            match self {
                Self::Fifo { calls, .. } | Self::Match { calls, .. } => {
                    calls.lock().unwrap().clone()
                }
            }
        }
    }
    impl NliClassifier for ScriptedNliClassifier {
        async fn classify(
            &self,
            premise: &str,
            hypothesis: &str,
        ) -> Result<NliResult, ProviderError> {
            match self {
                Self::Fifo { verdicts, calls } => {
                    calls
                        .lock()
                        .unwrap()
                        .push((premise.to_string(), hypothesis.to_string()));
                    let mut queue = verdicts.lock().unwrap();
                    if queue.is_empty() {
                        Err(ProviderError::Unavailable("scripted queue empty".into()))
                    } else {
                        queue.remove(0)
                    }
                }
                Self::Match { resolver, calls } => {
                    calls
                        .lock()
                        .unwrap()
                        .push((premise.to_string(), hypothesis.to_string()));
                    resolver(premise, hypothesis)
                }
            }
        }
    }

    /// NLI verdict that always returns `Neutral` (above the no-contradiction
    /// threshold but below entailment). Used when tests do not care about the
    /// specific label, only that the NLI backend was reachable.
    fn neutral_available() -> NliResult {
        NliResult {
            label: NliLabel::Neutral,
            scores: NliScores {
                entailment: 0.2,
                neutral: 0.7,
                contradiction: 0.1,
            },
            available: true,
        }
    }

    fn entailment_available() -> NliResult {
        NliResult {
            label: NliLabel::Entailment,
            scores: NliScores {
                entailment: 0.9,
                neutral: 0.08,
                contradiction: 0.02,
            },
            available: true,
        }
    }

    fn contradiction_available() -> NliResult {
        NliResult {
            label: NliLabel::Contradiction,
            scores: NliScores {
                entailment: 0.05,
                neutral: 0.1,
                contradiction: 0.85,
            },
            available: true,
        }
    }

    // ---- Fixtures ----

    fn memory_key() -> MemoryKey {
        MemoryKey::from_raw("origa").unwrap()
    }
    fn sid(n: u8) -> SessionId {
        SessionId::from_raw(&format!("sess_{:012x}", n as u64)).unwrap()
    }
    fn ts() -> Timestamp {
        Timestamp::from_unix_secs(1_700_000_000).unwrap()
    }

    /// Build a pending fact whose content-derived id is deterministic.
    fn pending(content: &str, embedding: Vec<f32>) -> Fact {
        Fact::new_pending(
            content,
            memory_key(),
            sid(1),
            Embedding::new(embedding).unwrap(),
            ts(),
            ConfidenceConfig::default().base,
        )
        .unwrap()
    }

    /// Build an accepted fact (single source, base confidence lifted above the
    /// accept threshold via `set_status_and_confidence`).
    fn accepted(content: &str, embedding: Vec<f32>) -> Fact {
        let mut f = Fact::new_pending(
            content,
            memory_key(),
            sid(2),
            Embedding::new(embedding).unwrap(),
            ts(),
            ConfidenceConfig::default().base,
        )
        .unwrap();
        f.set_status_and_confidence(
            FactStatus::Accepted,
            smos_domain::Confidence::new(0.9).unwrap(),
            &ConfidenceConfig::default(),
        )
        .unwrap();
        f
    }

    /// Build a session state carrying `owned` pending fact ids.
    fn session_with_pending(owned: Vec<FactId>) -> SessionState {
        let mut state = SessionState::new(sid(1), memory_key(), ts());
        state.add_pending(&owned);
        state
    }

    /// Shared fixture: confidence / NLI / merge configs owned by the test so
    /// the returned use case can borrow them for its whole lifetime.
    /// Mirrors the `Fix` pattern in `extract_facts_from_response`.
    struct Fix {
        confidence_cfg: ConfidenceConfig,
        nli_cfg: NliConfig,
        merge_cfg: MergeConfig,
    }
    impl Fix {
        fn new() -> Self {
            Self {
                confidence_cfg: ConfidenceConfig::default(),
                nli_cfg: NliConfig::default(),
                merge_cfg: MergeConfig::default(),
            }
        }
    }

    fn build<'a>(
        facts: &'a InMemoryFacts,
        sessions: &'a InMemorySessions,
        classifier: &'a ScriptedNliClassifier,
        fix: &'a Fix,
    ) -> FinalizeSession<'a, InMemoryFacts, InMemorySessions, ScriptedNliClassifier> {
        FinalizeSession {
            facts,
            sessions,
            classifier,
            confidence_cfg: &fix.confidence_cfg,
            nli_cfg: &fix.nli_cfg,
            merge_cfg: &fix.merge_cfg,
        }
    }

    // -----------------------------------------------------------------------
    // Happy-path tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn execute_no_session_returns_empty_stats() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let classifier = ScriptedNliClassifier::new(vec![]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.processed, 0);
        assert_eq!(stats.finalized, 0);
        assert!(classifier.calls().is_empty(), "no NLI call without pending");
    }

    /// Regression guard for the operator-facing bug: HTTP extraction persists
    /// `fact.source_sessions` but NEVER writes a `SessionState` row, so the
    /// previous implementation (which read `SessionState.pending_facts()`
    /// for ownership) reported "nothing to do" while 24 pending facts sat in
    /// the store. The fix derives ownership from `source_sessions` instead,
    /// so a missing SessionState must NOT mask real pending facts.
    #[tokio::test]
    async fn execute_processes_pending_facts_even_when_session_state_is_absent() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        // NO `sessions.seed(...)` — the HTTP path leaves SessionState empty.
        // The pending fact still carries `source_sessions = [sid(1)]`
        // (the `pending()` fixture sets it via `Fact::new_pending`), which
        // is the only ownership signal the use case consults after the fix.
        let fact = pending("user prefers rust over go", vec![1.0, 0.0, 0.0]);
        let fact_id = fact.id().clone();
        facts.seed(fact);

        let classifier = ScriptedNliClassifier::new(vec![]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(
            stats.processed, 1,
            "missing SessionState must not mask the fact"
        );
        assert_eq!(stats.finalized, 1);
        let finalized = facts.get_clone(&fact_id).expect("fact still present");
        assert_eq!(finalized.status(), FactStatus::Pending);
    }

    /// A pending fact whose `source_sessions` does NOT contain the target
    /// session is skipped — finalize is scoped to one session's ownership,
    /// not to every pending fact in the namespace.
    #[tokio::test]
    async fn execute_skips_pending_fact_owned_by_a_different_session() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        // `pending()` fixture sets source_sessions = [sid(1)] — finalizing
        // sid(2) must NOT pick it up.
        let fact = pending("user prefers rust over go", vec![1.0, 0.0, 0.0]);
        let fact_id = fact.id().clone();
        facts.seed(fact);

        let classifier = ScriptedNliClassifier::new(vec![]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(2), &memory_key()).await.unwrap();
        assert_eq!(stats.processed, 0);
        // The fact survives untouched.
        let untouched = facts.get_clone(&fact_id).expect("fact still present");
        assert_eq!(untouched.status(), FactStatus::Pending);
    }

    #[tokio::test]
    async fn execute_empty_session_returns_empty_stats() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        sessions.seed(SessionState::new(sid(1), memory_key(), ts()));
        let classifier = ScriptedNliClassifier::new(vec![]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.processed, 0);
    }

    #[tokio::test]
    async fn execute_standalone_promotes_pending_fact_with_no_candidate() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        // Pending fact with a unique embedding → no candidate above the merge
        // threshold (no accepted fact exists at all).
        let fact = pending("user prefers rust over go", vec![1.0, 0.0, 0.0]);
        let fact_id = fact.id().clone();
        facts.seed(fact);
        sessions.seed(session_with_pending(vec![fact_id.clone()]));

        let classifier = ScriptedNliClassifier::new(vec![]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.processed, 1);
        assert_eq!(stats.finalized, 1);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.conflicts, 0);
        // Single-source, base confidence (0.5) → Pending (validation gate).
        let finalized = facts.get_clone(&fact_id).expect("fact still present");
        assert_eq!(finalized.status(), FactStatus::Pending);
        assert!(
            classifier.calls().is_empty(),
            "no NLI call without candidate"
        );
        assert!(
            sessions.pending_of(&sid(1)).is_empty(),
            "owned pending cleared"
        );
    }

    #[tokio::test]
    async fn execute_entailment_merges_pending_into_existing() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let existing = accepted("ttl=10 prevents refresh loop", vec![1.0, 0.0, 0.0]);
        let existing_id = existing.id().clone();
        facts.seed(existing);
        // Pending twin: identical embedding (cosine 1.0 ≥ 0.85 merge threshold).
        let pending_fact = pending("ttl=10 stops the refresh loop", vec![1.0, 0.0, 0.0]);
        let pending_id = pending_fact.id().clone();
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        let classifier = ScriptedNliClassifier::new(vec![Ok(entailment_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.processed, 1);
        assert_eq!(stats.merged, 1);
        assert_eq!(stats.rejected, 1);
        assert_eq!(stats.finalized, 0);

        // Existing fact grew provenance (union of source sessions) and was
        // reclassified with the entailment verdict (no_contradiction_bonus).
        let merged = facts.get_clone(&existing_id).expect("existing present");
        assert!(merged.source_sessions().distinct_count() >= 2);
        // Pending twin was rejected.
        let twin = facts.get_clone(&pending_id).expect("pending present");
        assert_eq!(twin.status(), FactStatus::Rejected);
        assert!(
            sessions.pending_of(&sid(1)).is_empty(),
            "owned pending cleared"
        );
    }

    #[tokio::test]
    async fn execute_contradiction_flags_bidirectional_conflict() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let existing = accepted("ttl=60 seconds", vec![1.0, 0.0, 0.0]);
        let existing_id = existing.id().clone();
        facts.seed(existing);
        let pending_fact = pending("ttl=10 seconds", vec![1.0, 0.0, 0.0]);
        let pending_id = pending_fact.id().clone();
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.processed, 1);
        assert_eq!(stats.conflicts, 1);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.finalized, 0);

        // Both sides carry the bidirectional conflict flag.
        let existing_after = facts.get_clone(&existing_id).expect("existing present");
        let pending_after = facts.get_clone(&pending_id).expect("pending present");
        assert!(existing_after.conflicts_with().contains(&pending_id));
        assert!(pending_after.conflicts_with().contains(&existing_id));
        // Status UNCHANGED on both sides (Accepted stays Accepted, Pending stays Pending).
        assert_eq!(existing_after.status(), FactStatus::Accepted);
        assert_eq!(pending_after.status(), FactStatus::Pending);
        // No valid_until tombstone on either side (drift is not a death).
        assert!(existing_after.valid_until().is_none());
        assert!(pending_after.valid_until().is_none());
    }

    // -----------------------------------------------------------------------
    // Drift-priority walk
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn drift_priority_walk_contradiction_beats_earlier_neutral() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        // Two accepted facts, both above the cosine threshold. The closer
        // candidate ("similar") would yield `Neutral` — the use case must
        // keep scanning so the contradiction against the less-similar
        // candidate ("drift") still wins.
        let closer = accepted("rust is memory safe", vec![1.0, 0.0, 0.0]);
        let closer_id = closer.id().clone();
        let farther = accepted("rust leaks memory everywhere", vec![0.9, 0.1, 0.0]);
        let farther_id = farther.id().clone();
        facts.seed(closer);
        facts.seed(farther);
        let pending_fact = pending("rust is memory safe language", vec![1.0, 0.0, 0.0]);
        let pending_id = pending_fact.id().clone();
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        // find_merge_candidates sorts by cosine descending, so the first NLI
        // call hits "closer" (Neutral), the second hits "farther" (Contradiction).
        let classifier = ScriptedNliClassifier::new(vec![
            Ok(neutral_available()),
            Ok(contradiction_available()),
        ]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(
            stats.conflicts, 1,
            "drift must win over the earlier neutral"
        );
        assert_eq!(stats.merged, 0, "no merge despite the neutral candidate");

        // The contradiction was flagged against "farther" (the contradicting
        // candidate), NOT against "closer".
        let pending_after = facts.get_clone(&pending_id).expect("pending present");
        assert!(
            pending_after.conflicts_with().contains(&farther_id),
            "drift flag points to the contradicting candidate"
        );
        assert!(
            !pending_after.conflicts_with().contains(&closer_id),
            "no spurious drift flag on the neutral candidate"
        );
    }

    #[tokio::test]
    async fn drift_priority_walk_keeps_merge_pick_but_still_scans_for_contradiction() {
        // Entailment candidate first, contradiction second → contradiction wins,
        // the earlier merge pick is NOT committed.
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let entailed = accepted("the api runs on port 8080", vec![1.0, 0.0, 0.0]);
        let entailed_id = entailed.id().clone();
        let drift = accepted("the api runs on port 9090", vec![0.95, 0.05, 0.0]);
        let drift_id = drift.id().clone();
        facts.seed(entailed);
        facts.seed(drift);
        let pending_fact = pending("the api runs on port 8080 today", vec![1.0, 0.0, 0.0]);
        let pending_id = pending_fact.id().clone();
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        let classifier = ScriptedNliClassifier::new(vec![
            Ok(entailment_available()),
            Ok(contradiction_available()),
        ]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.conflicts, 1);
        assert_eq!(stats.merged, 0);
        // The entailed candidate was NOT modified (no merge committed).
        let entailed_after = facts.get_clone(&entailed_id).expect("entailed present");
        assert_eq!(
            entailed_after.source_sessions().distinct_count(),
            1,
            "merge not committed for the entailed candidate"
        );
        // The drift candidate was flagged.
        let drift_after = facts.get_clone(&drift_id).expect("drift present");
        assert!(drift_after.conflicts_with().contains(&pending_id));
    }

    // -----------------------------------------------------------------------
    // C3 guard — already-flagged pairs skip the sidecar
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn c3_guard_skips_nli_for_already_flagged_conflict_pair() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let mut existing = accepted("ttl=60 seconds", vec![1.0, 0.0, 0.0]);
        let mut pending_fact = pending("ttl=10 seconds", vec![1.0, 0.0, 0.0]);
        // Pre-flag the pair so the C3 guard fires before any sidecar call.
        existing.flag_conflict(pending_fact.id().clone()).unwrap();
        pending_fact.flag_conflict(existing.id().clone()).unwrap();
        let existing_id = existing.id().clone();
        let pending_id = pending_fact.id().clone();
        facts.seed(existing);
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        // The classifier would have returned contradiction, but the C3 guard
        // must short-circuit before any call.
        let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.processed, 1);
        // The C3 guard skipped every candidate → standalone promotion.
        assert_eq!(stats.finalized, 1);
        assert_eq!(stats.conflicts, 0);
        assert!(
            classifier.calls().is_empty(),
            "C3 guard must skip every sidecar call"
        );
        // Existing flags UNCHANGED (no double-flag).
        let existing_after = facts.get_clone(&existing_id).expect("existing present");
        assert_eq!(existing_after.conflicts_with().len(), 1);
        assert!(existing_after.conflicts_with().contains(&pending_id));
        // Pending twin also keeps its pre-flagged conflict link — the C3
        // guard leaves both sides untouched, which is the contract that
        // keeps a re-finalized session idempotent (no spurious
        // double-flag, no leak of the conflict to fresh candidates).
        let pending_after = facts.get_clone(&pending_id).expect("pending present");
        assert_eq!(pending_after.conflicts_with().len(), 1);
        assert!(
            pending_after.conflicts_with().contains(&existing_id),
            "pending twin must retain its pre-existing conflict flag"
        );
    }

    // -----------------------------------------------------------------------
    // Multi-contradiction — pending fact drifts against 2+ existing facts
    // -----------------------------------------------------------------------

    /// A pending fact that contradicts MULTIPLE accepted facts flags every
    /// contradiction it finds. Drift-priority means the FIRST
    /// contradiction wins for the *outcome* (the loop returns
    /// `Conflict` on the first one), but `resolve_one` continues scanning
    /// only until the first contradiction — it does NOT keep flagging
    /// after the drift is observed. This test pins that semantics: the
    /// second contradicting candidate is NOT visited once the first
    /// contradiction has fired.
    #[tokio::test]
    async fn multi_contradiction_returns_after_first_drift() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let existing_a = accepted("ttl=60 seconds", vec![1.0, 0.0, 0.0]);
        let existing_b = accepted("ttl=30 seconds", vec![0.95, 0.05, 0.0]);
        let a_id = existing_a.id().clone();
        let b_id = existing_b.id().clone();
        facts.seed(existing_a);
        facts.seed(existing_b);
        let pending_fact = pending("ttl=10 seconds", vec![1.0, 0.0, 0.0]);
        let pending_id = pending_fact.id().clone();
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        // First candidate returns contradiction → loop returns
        // immediately. The second verdict (also contradiction) is never
        // consumed.
        let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.conflicts, 1);
        assert_eq!(stats.processed, 1);
        assert_eq!(
            classifier.calls().len(),
            1,
            "first contradiction must short-circuit; second candidate not visited"
        );

        // The pending twin carries exactly ONE drift flag (against
        // whichever candidate was visited first — the merge-candidate
        // order is deterministic via cosine).
        let pending_after = facts.get_clone(&pending_id).expect("pending present");
        assert_eq!(
            pending_after.conflicts_with().len(),
            1,
            "exactly one drift flag on the pending twin"
        );
        // Sanity: the flagged id is one of the two existing facts.
        let flagged = pending_after
            .conflicts_with()
            .iter()
            .next()
            .expect("flag set");
        assert!(*flagged == a_id || *flagged == b_id);
    }

    // -----------------------------------------------------------------------
    // Exact-match short-circuit
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn exact_match_skips_sidecar_and_merges_identical_pair() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let existing = accepted("identical fact content", vec![1.0, 0.0, 0.0]);
        let existing_id = existing.id().clone();
        facts.seed(existing);
        // Pending twin has the SAME content → exact-match short-circuit.
        // Note: FactId is content-derived, so two identical-content facts
        // share the same id. We bypass that here by seeding the pending twin
        // under a different content hash via the lowercase trick (POC normalises
        // case + whitespace, so "IDENTICAL FACT CONTENT" exact-matches the
        // existing lower-case form).
        let pending_fact = pending("IDENTICAL FACT CONTENT", vec![1.0, 0.0, 0.0]);
        let pending_id = pending_fact.id().clone();
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        let classifier = ScriptedNliClassifier::new(vec![Ok(contradiction_available())]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        // Exact-match returns entailment immediately → merge committed. The
        // scripted contradiction verdict MUST NOT be consumed.
        assert_eq!(stats.merged, 1);
        assert_eq!(stats.conflicts, 0);
        assert!(
            classifier.calls().is_empty(),
            "exact-match must short-circuit before any sidecar call"
        );
        let merged = facts.get_clone(&existing_id).expect("existing present");
        assert!(merged.source_sessions().distinct_count() >= 2);
    }

    // -----------------------------------------------------------------------
    // Graceful degradation
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn sidecar_unavailable_keeps_pending_fact_gracefully() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let existing = accepted("rust is memory safe", vec![1.0, 0.0, 0.0]);
        facts.seed(existing);
        let pending_fact = pending("rust guarantees memory safety", vec![1.0, 0.0, 0.0]);
        let pending_id = pending_fact.id().clone();
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        // Every NLI call is Unavailable — the use case must not raise.
        let classifier = ScriptedNliClassifier::new(vec![Err(ProviderError::Unavailable(
            "sidecar crashed".into(),
        ))]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc
            .execute(&sid(1), &memory_key())
            .await
            .expect("graceful Ok");
        // No outcome tallied (skip does not increment any counter).
        assert_eq!(stats.finalized, 0);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.conflicts, 0);
        // The pending fact survives unchanged.
        let pending_after = facts.get_clone(&pending_id).expect("pending present");
        assert_eq!(pending_after.status(), FactStatus::Pending);
        assert!(pending_after.conflicts_with().is_empty());
    }

    #[tokio::test]
    async fn sidecar_replies_available_false_keeps_pending_fact_gracefully() {
        // The sidecar sometimes replies with its own graceful-degradation
        // placeholder (label=neutral, available=false) when the model raised
        // on a malformed input or the sidecar's stdout closed before the
        // reply landed. The use case must treat `available = false` exactly
        // like `Err(Unavailable)` — the pending fact stays pending so a
        // permanently broken sidecar cannot silently promote facts past the
        // drift-detection gate.
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let existing = accepted("rust is memory safe", vec![1.0, 0.0, 0.0]);
        facts.seed(existing);
        let pending_fact = pending("rust guarantees memory safety", vec![1.0, 0.0, 0.0]);
        let pending_id = pending_fact.id().clone();
        facts.seed(pending_fact.clone());
        sessions.seed(session_with_pending(vec![pending_id.clone()]));

        // Reply shape mirrors the "classifier unavailable" verdict produced
        // by the NLI backend on a transport/runtime failure (see
        // `ProviderError::Unavailable` mapping in `NativeNliClassifier`).
        let unavailable_verdict = NliResult {
            label: NliLabel::Neutral,
            scores: NliScores {
                entailment: 0.0,
                neutral: 1.0,
                contradiction: 0.0,
            },
            available: false,
        };
        let classifier = ScriptedNliClassifier::new(vec![Ok(unavailable_verdict)]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc
            .execute(&sid(1), &memory_key())
            .await
            .expect("graceful Ok");
        assert_eq!(stats.finalized, 0, "available=false must NOT promote");
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.conflicts, 0);
        let pending_after = facts.get_clone(&pending_id).expect("pending present");
        assert_eq!(pending_after.status(), FactStatus::Pending);
        assert!(
            pending_after.conflicts_with().is_empty(),
            "no drift flag without a real verdict"
        );
    }

    #[tokio::test]
    async fn batch_continues_after_single_pair_failure() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        // Three pending facts, two with candidates and one standalone. Each
        // candidate has a distinct content so the matcher can return a
        // deterministic verdict regardless of `HashMap` iteration order.
        let existing = accepted("shared anchor fact here", vec![1.0, 0.0, 0.0]);
        facts.seed(existing);
        // p1: similar but the matcher marks it as NLI-unavailable → skip pair.
        let p1 = pending("shared anchor fact here too", vec![1.0, 0.0, 0.0]);
        // p2: similar, matcher returns entailment → merge.
        let p2 = pending("shared anchor fact but longer", vec![1.0, 0.0, 0.0]);
        // p3: orthogonal embedding → no candidate → standalone promotion.
        let p3 = pending("totally unrelated pending fact", vec![0.0, 1.0, 0.0]);
        let p1_id = p1.id().clone();
        let p3_id = p3.id().clone();
        facts.seed(p1.clone());
        facts.seed(p2.clone());
        facts.seed(p3.clone());
        sessions.seed(session_with_pending(vec![
            p1.id().clone(),
            p2.id().clone(),
            p3.id().clone(),
        ]));

        // Order-independent matcher: keyed on the hypothesis text (the pending
        // twin) so HashMap iteration order over the pending list does not
        // change the outcome.
        let classifier = ScriptedNliClassifier::matching(|_premise, hypothesis| match hypothesis {
            "shared anchor fact here too" => Err(ProviderError::Unavailable("transient".into())),
            "shared anchor fact but longer" => Ok(entailment_available()),
            other => Err(ProviderError::InvalidResponse(format!(
                "unexpected hypothesis: {other}"
            ))),
        });
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        // One merge (p2 → existing), one finalize (p3 standalone), one skip
        // (p1 stayed pending because the sidecar was unreachable).
        assert_eq!(stats.processed, 3);
        assert_eq!(stats.merged, 1);
        assert_eq!(stats.finalized, 1);
        let p1_after = facts.get_clone(&p1_id).expect("p1 present");
        assert_eq!(p1_after.status(), FactStatus::Pending, "p1 stayed pending");
        let p3_after = facts.get_clone(&p3_id).expect("p3 present");
        // p3 standalone: single source, base confidence → still Pending.
        assert_eq!(p3_after.status(), FactStatus::Pending);
    }

    // -----------------------------------------------------------------------
    // Bookkeeping cleanup
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn finalize_clears_owned_pending_ids_after_drain() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        // Two pending facts, both owned by the session. After finalize the
        // session's pending list must be empty (both owned ids drained).
        let p1 = pending("first standalone pending fact", vec![1.0, 0.0, 0.0]);
        let p2 = pending("second standalone pending fact", vec![0.0, 1.0, 0.0]);
        let p1_id = p1.id().clone();
        let p2_id = p2.id().clone();
        facts.seed(p1);
        facts.seed(p2);
        sessions.seed(session_with_pending(vec![p1_id, p2_id]));

        let classifier = ScriptedNliClassifier::new(vec![]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(1), &memory_key()).await.unwrap();
        assert_eq!(stats.processed, 2);
        assert!(
            sessions.pending_of(&sid(1)).is_empty(),
            "owned pending ids cleared after finalize"
        );
    }

    // -----------------------------------------------------------------------
    // Stats contract
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn stats_default_is_zeroed() {
        let stats = FinalizeStats::default();
        assert_eq!(stats.processed, 0);
        assert_eq!(stats.finalized, 0);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.conflicts, 0);
        assert_eq!(stats.rejected, 0);
        assert!(stats.session_id.is_empty());
    }

    #[tokio::test]
    async fn stats_session_id_echoed_in_output() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        sessions.seed(SessionState::new(sid(7), memory_key(), ts()));
        let classifier = ScriptedNliClassifier::new(vec![]);
        let fix = Fix::new();
        let uc = build(&facts, &sessions, &classifier, &fix);

        let stats = uc.execute(&sid(7), &memory_key()).await.unwrap();
        assert_eq!(stats.session_id, sid(7).as_str());
    }
}
