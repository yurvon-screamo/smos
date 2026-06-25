//! Fact persistence: embed every extracted raw fact and route each through
//! the 3-layer dedup flow in [`super::dedup`].
//!
//! Split out of the historical `extract_facts_from_response.rs` god-module
//! (R9); behaviour-preserving — the body below is the verbatim original.

use smos_domain::{FactId, MemoryKey, SessionId};

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
    /// Embed each raw fact and persist it through the 3-layer dedup flow:
    ///
    /// 1. **Exact `FactId` match** — same `SHA1(content)` already stored →
    ///    cross-session confirmation (the deterministic baseline).
    /// 2. **Semantic match** — cosine similarity ≥
    ///    [`ExtractionConfig::dedup_cosine_threshold`] against an existing
    ///    fact → cross-session confirmation. Safety net for non-deterministic
    ///    extraction: a rephrased re-observation may hash to a different
    ///    `FactId` while the embedding is still near-identical.
    /// 3. **No match** — store a new pending fact and count it.
    ///
    /// Returns the ids of newly-stored pending facts. Confirmations do not
    /// count (they update an existing fact rather than adding one) and are
    /// NOT registered on the session pending list.
    pub(crate) async fn persist_facts(
        &self,
        raw_facts: &[String],
        memory_key: &MemoryKey,
        session_id: &SessionId,
    ) -> Result<Vec<FactId>, UseCaseError> {
        let refs: Vec<&str> = raw_facts.iter().map(String::as_str).collect();
        let embeddings = self.embedder.embed_batch(&refs).await?;

        let mut new_ids = Vec::new();
        for (raw, embedding) in raw_facts.iter().zip(embeddings) {
            // Skip facts the embedder could not vectorise — they would never be
            // retrievable, so storing them is pure noise.
            let Some(vector) = embedding else { continue };
            if let Some(id) = self
                .persist_one_fact(raw, vector, memory_key, session_id)
                .await?
            {
                new_ids.push(id);
            }
        }
        Ok(new_ids)
    }
}
