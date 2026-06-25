//! 3-layer dedup pipeline for newly-extracted facts.
//!
//! `ExtractFactsFromResponse::persist_one_fact` routes each (raw, embedding)
//! pair through three layers, in order, stopping at the first hit:
//!
//! 1. **Exact `FactId` match** — cross-session confirmation (the only path
//!    a single-session Pending fact can reach the accept threshold).
//! 2. **Semantic match** — cosine ≥ `extraction.dedup_cosine_threshold`
//!    backstops the exact match when the model rephrases a fact just enough
//!    to hash to a different id.
//! 3. **No match** — store a new pending fact.
//!
//! Split out of the historical `extract_facts_from_response.rs` god-module
//! (R9); behaviour-preserving — the bodies below are the verbatim originals.

use smos_domain::{Embedding, Fact, FactId, MemoryKey, NewPendingRequest, SessionId};

use crate::errors::UseCaseError;
use crate::ports::{
    Clock, Delay, EmbeddingProvider, FactRepository, LlmExtractor, SessionRepository,
};

use super::ExtractFactsFromResponse;

impl<'a, FR, SR, EP, LE, C, D> ExtractFactsFromResponse<'a, FR, SR, EP, LE, C, D>
where
    FR: FactRepository,
    SR: SessionRepository,
    EP: EmbeddingProvider,
    LE: LlmExtractor,
    C: Clock,
    D: Delay,
{
    /// Dedup a single (raw, embedding) pair through the 3-layer flow.
    /// Returns `Some(FactId)` when a NEW fact was created; `None` when the
    /// fact confirmed an existing one (exact or semantic match).
    ///
    /// # Layer 2 vs Layer 1 confidence gap
    ///
    /// Layer 2 (semantic dedup) and Layer 1 (exact `FactId` match) both
    /// call [`Fact::confirm_cross_session`], which internally invokes
    /// `reclassify(None, cfg)` — WITHOUT an NLI verdict. As a result the
    /// `no_contradiction_bonus` (default `0.1`) is NOT applied on either
    /// path: a fact that confirms via dedup reaches at most
    /// `base (0.5) + multi_source_bonus (0.2) = 0.7`, exactly equal to the
    /// default `accept_threshold`. Only [`FinalizeSession`]'s NLI-backed
    /// merge path applies the `no_contradiction_bonus` (lifting the
    /// confirmation to `0.8`). The gap is intentional: dedup happens on
    /// every extraction cycle (cheap, synchronous), NLI only on
    /// session-end (expensive, async). Promoting via dedup at
    /// `accept_threshold` is the safe minimum; the bonus is reserved for
    /// the path that actually proved there is no contradiction.
    pub(crate) async fn persist_one_fact(
        &self,
        raw: &str,
        vector: Vec<f32>,
        memory_key: &MemoryKey,
        session_id: &SessionId,
    ) -> Result<Option<FactId>, UseCaseError> {
        let fact_id = FactId::from_content(raw);

        // Layer 1 — exact FactId match (cheap, deterministic).
        if let Some(mut existing) = self.facts.get(&fact_id, memory_key).await? {
            self.confirm_and_save(&mut existing, session_id).await?;
            return Ok(None);
        }

        // Layer 2 — semantic match (cosine >= threshold). Safety net for
        // non-deterministic extraction: a rephrased fact may hash to a
        // different FactId while its embedding is still near-identical.
        // Uses `search_for_dedup` (pending + accepted) so a still-pending
        // fact is reachable — `search_similar` is accepted-only and would
        // deadlock the cross-session confirmation that promotes it.
        let similar = self
            .facts
            .search_for_dedup(vector.clone(), memory_key, 1)
            .await?;
        if let Some(hit) = similar.into_iter().next() {
            match hit.metadata.distance {
                Some(d) => {
                    let similarity = 1.0 - d;
                    if similarity >= self.extraction_cfg.dedup_cosine_threshold
                        && let Some(mut fact) = self.facts.get(&hit.id, memory_key).await?
                    {
                        tracing::debug!(
                            raw = raw,
                            similarity = similarity,
                            matched_id = %hit.id,
                            "semantic dedup: rephrased fact matched an existing one"
                        );
                        self.confirm_and_save(&mut fact, session_id).await?;
                        return Ok(None);
                    }
                }
                None => {
                    // Distance missing — the store did not surface a cosine
                    // score (rare; only happens for adapters that forget to
                    // populate `metadata.distance`). Fail open to Layer 3
                    // rather than collapse two unrelated facts.
                    tracing::warn!(
                        raw = raw,
                        matched_id = %hit.id,
                        "semantic dedup hit carried no distance; skipping Layer 2 \
                         (create new pending fact instead)"
                    );
                }
            }
        }

        // Layer 3 — no match: store a new pending fact.
        let emb = Embedding::new(vector)?;
        let fact = Fact::new_pending(NewPendingRequest {
            content: raw,
            memory_key: memory_key.clone(),
            session: session_id.clone(),
            embedding: emb,
            extracted_at: self.clock.now(),
            base_confidence: self.confidence_cfg.base,
        })?;
        self.facts.save(&fact).await?;
        Ok(Some(fact_id))
    }

    /// Run cross-session confirmation against `fact` and persist the updated
    /// provenance when the validation gate fired. `confirm_cross_session`
    /// returns `false` when the session is already in the provenance set —
    /// in that case no save is needed (the row is unchanged).
    pub(crate) async fn confirm_and_save(
        &self,
        fact: &mut Fact,
        session_id: &SessionId,
    ) -> Result<(), UseCaseError> {
        if fact.confirm_cross_session(session_id, self.confidence_cfg)? {
            self.facts.save(fact).await?;
        }
        Ok(())
    }
}
