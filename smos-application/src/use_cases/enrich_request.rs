//! `EnrichRequest` — full enrichment pipeline (§3, §7, §12).
//!
//! Inputs: OpenAI-shaped messages, `memory_key`, `session_id`, and references
//! to every port the pipeline needs (`FactRepository`, `SessionRepository`,
//! `EmbeddingProvider`, `RerankProvider`, `Clock`).
//!
//! Output: `Ok(messages)` — an enriched messages array (with a `<smos-memory>`
//! block prepended to the first message) or, on any recoverable failure, the
//! original messages unchanged.
//!
//! # Fail-open vs fail-closed contract
//!
//! The pipeline is **fail-open** for every port EXCEPT the reranker: an
//! embedder error, an empty vector-search result, a repo failure, or a
//! dedup hiccup all degrade to forwarding the original messages so a flaky
//! memory subsystem never breaks the user's chat. The reranker, however, is
//! **fail-closed**: a provider error or an empty rerank result propagates as
//! `Err(UseCaseError::Provider(_))` so the HTTP handler returns 503 instead
//! of silently shipping vector-order-only ranking.
//!
//! `execute` therefore returns `Result<Vec<Value>, UseCaseError>`:
//! - `Ok(messages)` — every fail-open path returns the (possibly enriched)
//!   messages array. Callers may assign the result unconditionally.
//! - `Err(UseCaseError::Provider(_))` — only the reranker. The handler maps
//!   this to HTTP 503 "SMOS provider unavailable: …".
//!
//! # Pipeline (mirrors `smos-poc/smos/enrich.py::enrich_request`)
//!
//! 1. Extract topic from the last message.
//! 2. Short-circuit when the topic (after `trim`) is below `min_topic_chars`.
//! 3. Embed the topic. `None` (and any provider error) short-circuits to the
//!    original messages.
//! 4. Vector search top-K candidates (`top_k_initial`).
//! 5. Apply pre-filters (status / validity / confidence) and heat post-filter.
//! 6. Short-circuit when no survivors remain.
//! 7. Heat boost — every survivor is rewarmed to `heat_base = 1.0` with
//!    `last_access_at = now`.
//!    - **Persona injection (§3 step 7 / §11) is deferred to a later slice**
//!      and intentionally not represented as a numbered step here. Reading
//!      `memory_key/persona.md` once per session and prepending a
//!      `[persona-...]` block will be added alongside a `PersonaRepository`
//!      port; the domain builder already accepts `memory_key` for forward
//!      compatibility.
//! 8. Rerank survivors with the cross-encoder. A provider error or an empty
//!    result propagates as `Err(UseCaseError::Provider(_))` and the request
//!    fails with HTTP 503.
//! 9. Session dedup — drop facts already injected into this session.
//! 10. Short-circuit when no new facts survived dedup.
//! 11. Build the `<smos-memory>` block from the new facts.
//! 12. Inject the block into the first message and return.

use std::collections::HashSet;

use serde_json::Value;
use smos_domain::config::{HeatConfig, RetrievalConfig};
use smos_domain::{FactId, Heat, MemoryKey, SessionId, Timestamp};

use crate::errors::{ProviderError, UseCaseError};
use crate::helpers::memory_block::{self, MemoryBlockEntry};
use crate::helpers::request_enricher;
use crate::helpers::retrieval_pipeline;
use crate::helpers::retrieval_planner::{self, RetrievalHit};
use crate::helpers::topic_extractor;
use crate::ports::{Clock, EmbeddingProvider, FactRepository, RerankProvider, SessionRepository};
use crate::types::{EnrichmentMessages, SearchHit, enrichment_messages_from_json};

/// Borrow-style bundle of every dependency the enrichment pipeline needs.
///
/// The struct holds references so a single allocation per request is enough —
/// callers build it inline at the call site and drop it right after
/// [`EnrichRequest::execute`] returns.
pub struct EnrichRequest<'a, FR, SR, EP, RP, C> {
    pub facts: &'a FR,
    pub sessions: &'a SR,
    pub embedder: &'a EP,
    pub reranker: &'a RP,
    pub clock: &'a C,
    pub retrieval_cfg: &'a RetrievalConfig,
    pub heat_cfg: &'a HeatConfig,
}

impl<'a, FR, SR, EP, RP, C> EnrichRequest<'a, FR, SR, EP, RP, C>
where
    FR: FactRepository,
    SR: SessionRepository,
    EP: EmbeddingProvider,
    RP: RerankProvider,
    C: Clock,
{
    /// Run the enrichment pipeline.
    ///
    /// Returns `Ok(messages)` — the enriched array when enrichment succeeded,
    /// or the original messages unchanged when any fail-open port short-
    /// circuited (topic too short, embedder error, no vector hits, no
    /// survivors, no new facts after dedup). The only `Err` path is the
    /// reranker (§3 step 8): a provider error or an empty rerank result
    /// propagates as [`UseCaseError::Provider`] so the HTTP handler maps it
    /// to 503.
    ///
    /// # Return paths (R7 decomposition)
    ///
    /// The body delegates the fail-open stages (steps 1–7) to the private
    /// `retrieve_survivors` method and the fail-closed + dedup stages (8–11)
    /// to the private `rerank_and_dedup` method. Every return path of the previous monolithic
    /// `execute` is preserved verbatim:
    ///
    /// 1. topic below `min_topic_chars` → `Ok(messages)`
    /// 2. embedder returned `None` → `Ok(messages)`
    /// 3. embedder returned `Err` → `Ok(messages)`
    /// 4. vector search returned `Err` → `Ok(messages)`
    /// 5. vector search returned zero hits → `Ok(messages)`
    /// 6. prefilter left zero survivors → `Ok(messages)`
    /// 7. reranker provider error → `Err(UseCaseError::Provider(_))`
    /// 8. reranker returned zero usable results → `Err(UseCaseError::Provider(_))`
    /// 9. dedup left zero new facts → `Ok(messages)`
    /// 10. happy path → `Ok(enriched_array)`
    ///
    /// # H-5 wire-shape preservation
    ///
    /// The function operates on `Vec<serde_json::Value>` **end-to-end**: the
    /// typed [`EnrichmentMessages`] projection is built once at the top for
    /// the read-only helpers (topic extraction, marker detection) and NEVER
    /// re-serialised back to JSON. The mutation step (`request_enricher::
    /// inject_value`) works on the raw `Value` directly, preserving every
    /// per-message *sibling* field of `messages[0]` (`name`, `tool_call_id`,
    /// `refusal`, unknown future OpenAI extensions) and every message in
    /// `[1..]` verbatim. The content field itself is flattened to text
    /// before prepend (so a multipart `messages[0].content` with
    /// `image_url` parts is reduced to its concatenated text parts +
    /// the prepended block) — same behaviour as the pre-H-5 pipeline,
    /// pinned by the e2e enrichment suite. Round-tripping the typed DTO
    /// would silently drop the sibling fields and break the fail-open
    /// contract for tool-calling and vision workflows.
    pub async fn execute(
        &self,
        messages: Vec<Value>,
        memory_key: &MemoryKey,
        session_id: &SessionId,
    ) -> Result<Vec<Value>, UseCaseError> {
        // Read-only typed projection: used only for topic extraction here
        // (session-marker detection runs in `HandleChatCompletion`, which
        // also uses the read-only projection). NEVER re-serialised — the
        // raw `messages: Vec<Value>` stays the source of truth for the
        // mutation + return path.
        let typed_projection = enrichment_messages_from_json(&messages);

        // Steps 1–7: topic gate, embed, vector search, prefilter+heat,
        // heat boost. Any fail-open short-circuit returns `Ok(None)`.
        let Some((topic, survivors)) = self
            .retrieve_survivors(&typed_projection, memory_key)
            .await?
        else {
            return Ok(messages);
        };

        // Steps 8–10: rerank (fail-closed) + defensive guard + session dedup.
        // An empty `new_facts` after dedup is fail-open (return original).
        let new_facts = self
            .rerank_and_dedup(&topic, &survivors, session_id, memory_key)
            .await?;
        if new_facts.is_empty() {
            return Ok(messages);
        }

        // Step 11–12 — build memory block + inject via the JSON-path entry
        // point. `inject_value` operates on the raw `Value` and mutates
        // only `messages[0].content`, preserving every other per-message
        // field the typed DTO does not model (name, tool_call_id,
        // refusal, image_url parts, …). Round-tripping the typed DTO
        // here would silently drop those fields.
        let block = build_memory_block(&new_facts, session_id, memory_key);
        let messages_value = Value::Array(messages);
        let enriched = request_enricher::inject_value(&messages_value, &block);
        match enriched {
            Value::Array(arr) => Ok(arr),
            // Defensive: `inject_value` is documented to always echo the
            // input shape (array in → array out); anything else indicates
            // a domain bug.
            other => Ok(vec![other]),
        }
    }

    /// Steps 1–7 of the pipeline: topic gate, embed, vector search,
    /// prefilter+heat, heat boost.
    ///
    /// Returns:
    /// - `Ok(None)` — fail-open short-circuit (paths 1–6): the caller MUST
    ///   return the original messages verbatim.
    /// - `Ok(Some(survivors))` — at least one survivor reached step 7;
    ///   rerank+dryheat will run next.
    /// - `Err(_)` — never returned today (every port in this stage is
    ///   fail-open), kept `Result`-shaped for symmetry with `rerank_and_dedup`
    ///   and to leave room for a future fail-closed check without churning
    ///   the caller's `?` chain.
    async fn retrieve_survivors(
        &self,
        typed_projection: &EnrichmentMessages,
        memory_key: &MemoryKey,
    ) -> Result<Option<(String, Vec<RetrievalHit>)>, UseCaseError> {
        // Step 1 + 2 — short-circuit on empty / too-short topic. POC parity:
        // `if len(topic.strip()) < min_topic_chars`. Trimming prevents
        // whitespace-only topics (e.g. `"   "`) from passing the gate and
        // producing a garbage embedding downstream.
        let topic = topic_extractor::extract_from_messages(typed_projection);
        let trimmed_len = topic.trim().chars().count();
        if trimmed_len < self.retrieval_cfg.min_topic_chars {
            tracing::debug!(
                chars = trimmed_len,
                "enrichment skipped: topic below min_topic_chars"
            );
            return Ok(None);
        }

        // Step 3 — embed. None and errors are both fail-open.
        let embedding = match self.embedder.embed(&topic).await {
            Ok(Some(v)) => v,
            Ok(None) => {
                tracing::warn!("embedder returned None; skipping enrichment (fail-open)");
                return Ok(None);
            }
            Err(e) => {
                tracing::warn!(error = %e, "embedder error; skipping enrichment (fail-open)");
                return Ok(None);
            }
        };

        // Step 4 — vector search. The repo owns the search algorithm
        // (HNSW + brute-force fallback); we just hand over the embedding.
        let hits = match self
            .facts
            .search_similar(embedding, memory_key, self.retrieval_cfg.top_k_initial)
            .await
        {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!(error = %e, "vector search failed; skipping enrichment (fail-open)");
                return Ok(None);
            }
        };
        if hits.is_empty() {
            tracing::info!(memory_key = %memory_key, "no vector hits; skipping enrichment");
            return Ok(None);
        }

        // Step 5 — pre-filter + heat post-filter (pure domain).
        let now = self.clock.now();
        let survivors = prefilter(hits, self.retrieval_cfg, self.heat_cfg, now);

        // Step 6 — short-circuit when no survivors remain.
        if survivors.is_empty() {
            return Ok(None);
        }

        // Step 7 — heat boost. Best-effort: a failure here is logged but does
        // not abort enrichment (the rerank/dedup still works on stale heat).
        self.boost_heat(&survivors, memory_key, now).await;

        Ok(Some((topic, survivors)))
    }

    /// Steps 8–10 of the pipeline: rerank (fail-closed), defensive guard,
    /// session dedup.
    ///
    /// Returns:
    /// - `Ok(new_facts)` — possibly empty; the caller decides whether to
    ///   short-circuit (empty) or build the memory block (non-empty).
    /// - `Err(UseCaseError::Provider(_))` — reranker provider error or an
    ///   empty/zero-result rerank response (paths 7–8). Propagates to the
    ///   HTTP handler as 503.
    async fn rerank_and_dedup(
        &self,
        topic: &str,
        survivors: &[RetrievalHit],
        session_id: &SessionId,
        memory_key: &MemoryKey,
    ) -> Result<Vec<RetrievalHit>, UseCaseError> {
        // Step 8 — rerank. Fail-closed: a provider error or an empty result
        // propagates as `UseCaseError::Provider(_)` so the HTTP handler
        // returns 503. See the module docs for rationale.
        let ranked_facts = self.rerank_survivors(topic, survivors).await?;

        // Step 8b — defensive guard: `rerank_survivors` already returns Err
        // when the provider responds with zero results OR when none of the
        // returned indices map back to survivors. The guard exists only so
        // a future port implementation that produces an empty `Vec` through
        // a different code path STILL honours the fail-closed contract
        // rather than silently building an empty `<smos-memory>` block.
        if ranked_facts.is_empty() {
            return Err(UseCaseError::Provider(ProviderError::InvalidResponse(
                "reranker returned no usable results".to_string(),
            )));
        }

        // Step 9–10 — session dedup (atomic via SessionRepository::dedup_and_mark).
        // Fail-open: on error, no new facts are surfaced (empty Vec).
        Ok(self
            .dedup_against_session(&ranked_facts, session_id, memory_key)
            .await)
    }

    /// Heat boost: every survivor gets `heat_base = 1.0`, `last_access_at = now`.
    ///
    /// Errors are logged and swallowed because heat is best-effort — a failure
    /// to rewarm does not break the pipeline.
    async fn boost_heat(&self, survivors: &[RetrievalHit], memory_key: &MemoryKey, now: Timestamp) {
        let ids: Vec<FactId> = survivors.iter().map(|h| h.id.clone()).collect();
        // `Heat::MAX` is a `const` (= `1.0`); no runtime validation needed.
        if let Err(e) = self
            .facts
            .update_heat_batch(&ids, memory_key, Heat::MAX, now)
            .await
        {
            tracing::warn!(error = %e, "heat boost failed (best-effort); continuing");
        }
    }

    /// Rerank survivors with the cross-encoder. Fail-closed: a provider
    /// error or an empty result returns `Err` and the request fails with
    /// HTTP 503 instead of silently shipping vector-order-only ranking.
    ///
    /// Delegates to [`retrieval_pipeline::rerank_hits`] so the read-only
    /// `RetrieveFacts` use case shares the exact same rerank path; the score
    /// carried by [`retrieval_pipeline::RankedHit`] is dropped here because
    /// the live enrichment pipeline only consumes the ordered hits.
    async fn rerank_survivors(
        &self,
        topic: &str,
        survivors: &[RetrievalHit],
    ) -> Result<Vec<RetrievalHit>, ProviderError> {
        let ranked = retrieval_pipeline::rerank_hits(
            topic,
            survivors,
            self.reranker,
            self.retrieval_cfg.top_k_final,
        )
        .await?;
        Ok(ranked.into_iter().map(|r| r.hit).collect())
    }

    /// Atomic dedup against the session's `injected_facts` set. Returns only
    /// the facts that are new to this session.
    async fn dedup_against_session(
        &self,
        ranked_facts: &[RetrievalHit],
        session_id: &SessionId,
        memory_key: &MemoryKey,
    ) -> Vec<RetrievalHit> {
        let candidate_ids: Vec<FactId> = ranked_facts.iter().map(|f| f.id.clone()).collect();
        let new_ids: HashSet<FactId> = match self
            .sessions
            .dedup_and_mark(session_id, memory_key, &candidate_ids)
            .await
        {
            Ok(ids) => ids.into_iter().collect(),
            Err(e) => {
                tracing::warn!(error = %e, "dedup_and_mark failed; skipping injection (fail-open)");
                return Vec::new();
            }
        };
        if new_ids.is_empty() {
            return Vec::new();
        }
        ranked_facts
            .iter()
            .filter(|f| new_ids.contains(&f.id))
            .cloned()
            .collect()
    }
}

// Module-private free functions keep the `impl` block under the size limit and
// make the pure pieces individually testable without spinning up ports.

/// Convert `SearchHit` rows (the adapter DTO) into the domain's `RetrievalHit`
/// projection, then apply pre-filters + heat post-filter.
fn prefilter(
    hits: Vec<SearchHit>,
    retrieval_cfg: &RetrievalConfig,
    heat_cfg: &HeatConfig,
    now: Timestamp,
) -> Vec<RetrievalHit> {
    let retrieval_hits: Vec<RetrievalHit> = hits
        .into_iter()
        .filter_map(retrieval_pipeline::hit_to_retrieval)
        .collect();
    retrieval_planner::prefilter_and_heat(&retrieval_hits, retrieval_cfg, heat_cfg, now)
}

/// Render the `<smos-memory>` block from the new facts.
fn build_memory_block(
    facts: &[RetrievalHit],
    session_id: &SessionId,
    memory_key: &MemoryKey,
) -> String {
    let entries: Vec<MemoryBlockEntry<'_>> = facts
        .iter()
        .map(|f| MemoryBlockEntry {
            id: &f.id,
            document: f.document.as_str(),
        })
        .collect();
    memory_block::build(entries, session_id, memory_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessageDto, EnrichmentMessages, MessageContent};
    use smos_domain::{FactStatus, MemoryKey, SessionId};

    fn user_msg(content: &str) -> ChatMessageDto {
        ChatMessageDto {
            role: "user".into(),
            content: MessageContent::Text(content.into()),
            tool_calls: None,
        }
    }

    #[test]
    fn extract_topic_from_string_content() {
        let msgs: EnrichmentMessages = vec![user_msg("hello world")];
        assert_eq!(topic_extractor::extract_from_messages(&msgs), "hello world");
    }

    #[test]
    fn extract_topic_returns_empty_when_no_messages() {
        let msgs: EnrichmentMessages = Vec::new();
        assert_eq!(topic_extractor::extract_from_messages(&msgs), "");
    }

    #[test]
    fn extract_topic_returns_empty_when_missing_content() {
        let msg = ChatMessageDto {
            role: "user".into(),
            content: MessageContent::Text(String::new()),
            tool_calls: None,
        };
        let msgs: EnrichmentMessages = vec![msg];
        assert_eq!(topic_extractor::extract_from_messages(&msgs), "");
    }

    #[test]
    fn extract_topic_flattens_multipart() {
        let msg = ChatMessageDto {
            role: "user".into(),
            content: MessageContent::Multipart(vec![
                crate::types::ContentPart {
                    kind: "text".into(),
                    text: "alpha".into(),
                },
                crate::types::ContentPart {
                    kind: "image_url".into(),
                    text: String::new(),
                },
                crate::types::ContentPart {
                    kind: "text".into(),
                    text: "beta".into(),
                },
            ]),
            tool_calls: None,
        };
        let msgs: EnrichmentMessages = vec![msg];
        assert_eq!(topic_extractor::extract_from_messages(&msgs), "alpha beta");
    }

    // The pure helpers (`parse_fact_status`, `parse_iso_timestamp`,
    // `hit_to_retrieval`) and their unit tests now live in
    // `crate::helpers::retrieval_pipeline`; the rerank path is covered there
    // and shared with `RetrieveFacts`, so this module no longer duplicates
    // them.

    // -----------------------------------------------------------------------
    // build_memory_block — format smoke
    // -----------------------------------------------------------------------

    #[test]
    fn build_memory_block_includes_session_and_fact_lines() {
        let session = SessionId::from_raw("sess_0123456789ab").expect("session");
        let key = MemoryKey::from_raw("origa").expect("key");
        let facts = vec![RetrievalHit {
            id: FactId::from_raw("fact_0123456789abcdef").expect("fact"),
            document: "hello world".into(),
            memory_key: key.clone(),
            status: FactStatus::Accepted,
            confidence: smos_domain::Confidence::new(0.9).unwrap(),
            valid_until: None,
            heat_base: Heat::MAX,
            last_access_at: Timestamp::from_unix_secs(1_700_000_000).unwrap(),
        }];
        let block = build_memory_block(&facts, &session, &key);
        assert!(block.contains("<smos-memory"));
        assert!(block.contains("hello world"));
    }

    // -----------------------------------------------------------------------
    // prefilter — empty input yields empty output
    // -----------------------------------------------------------------------

    #[test]
    fn prefilter_returns_empty_for_empty_input() {
        let cfg = RetrievalConfig::default();
        let heat = HeatConfig::default();
        let now = Timestamp::from_unix_secs(1_700_000_000).unwrap();
        assert!(prefilter(Vec::new(), &cfg, &heat, now).is_empty());
    }
}

// ===========================================================================
// execute — orchestrator-level unit tests (the pipeline end-to-end through
// the public entry point). The helper-level tests above cover the pure
// pieces; these three pin the fail-open / fail-closed CONTRACT of
// `EnrichRequest::execute`, which until now was only exercised by the
// adapter-layer e2e suite.
// ===========================================================================
#[cfg(test)]
mod execute_tests {
    use super::*;
    use crate::ports::{EmbeddingProvider, RerankProvider};
    use crate::testkit::{ConstantEmbedder, FixedClock, InMemoryFacts, InMemorySessions};
    use crate::types::{RerankResult, SearchHit, SearchHitMetadata};
    use serde_json::json;

    // ---- Local doubles for the two ports the testkit does not cover ----
    //
    // The testkit ships no None-returning embedder and no reranker at all
    // (finalize / extract / import use neither), so these minimal fakes fill
    // exactly the two gaps the execute contract needs.

    /// Always returns `Ok(None)` — drives the fail-open embedder path (step 3).
    struct NoneEmbedder;
    impl EmbeddingProvider for NoneEmbedder {
        async fn embed(&self, _text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
            Ok(None)
        }
    }

    /// Always returns an empty rerank result — drives the fail-closed path
    /// (step 8): an empty response must surface as `UseCaseError::Provider`.
    struct EmptyReranker;
    impl RerankProvider for EmptyReranker {
        async fn rerank(
            &self,
            _query: &str,
            _documents: &[String],
            _top_k: usize,
        ) -> Result<Vec<RerankResult>, ProviderError> {
            Ok(Vec::new())
        }
    }

    fn now_ts() -> Timestamp {
        Timestamp::from_unix_secs(1_700_000_000).unwrap()
    }
    fn key() -> MemoryKey {
        MemoryKey::from_raw("origa").unwrap()
    }
    fn sid() -> SessionId {
        SessionId::from_raw("sess_0123456789ab").unwrap()
    }

    /// A SearchHit that clears the retrieval pre-filter + heat post-filter
    /// (accepted, no tombstone, confidence above the 0.7 floor, max heat with
    /// `last_access_at == now` so there is no decay). Scripting this hit into
    /// `InMemoryFacts::search_similar` pushes `EnrichRequest` past step 7 so
    /// the reranker stage actually runs.
    fn survivable_hit() -> SearchHit {
        SearchHit {
            id: FactId::from_content("an accepted fact about rust ownership"),
            document: "an accepted fact about rust ownership".to_string(),
            memory_key: key(),
            metadata: SearchHitMetadata {
                status: "accepted".into(),
                confidence: 0.85,
                valid_until: None,
                heat_base: 1.0,
                last_access_at: 1_700_000_000.0,
                distance: Some(0.1),
                created_at: None,
                conflicts_with: Vec::new(),
            },
        }
    }

    /// 1. topic below `min_topic_chars` (default 3) short-circuits before the
    ///    embedder is even consulted, returning the original messages verbatim.
    #[tokio::test]
    async fn enrich_skips_when_topic_below_min_chars() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker = EmptyReranker;
        let clock = FixedClock(now_ts());
        let retrieval = RetrievalConfig::default();
        let heat = HeatConfig::default();
        let uc = EnrichRequest {
            facts: &facts,
            sessions: &sessions,
            embedder: &embedder,
            reranker: &reranker,
            clock: &clock,
            retrieval_cfg: &retrieval,
            heat_cfg: &heat,
        };
        // "ok" is 2 chars < min_topic_chars (3) -> step-2 short-circuit.
        let original = vec![json!({"role": "user", "content": "ok"})];
        let out = uc
            .execute(original.clone(), &key(), &sid())
            .await
            .expect("ok");
        assert_eq!(
            out, original,
            "topic below min_topic_chars returns the messages unchanged (fail-open)"
        );
    }

    /// 2. embedder `None` is fail-open (step 3): the pipeline returns the
    ///    original messages rather than erroring or injecting an empty block.
    #[tokio::test]
    async fn enrich_fail_opens_when_embedder_returns_none() {
        let facts = InMemoryFacts::default();
        let sessions = InMemorySessions::default();
        let embedder = NoneEmbedder;
        let reranker = EmptyReranker;
        let clock = FixedClock(now_ts());
        let retrieval = RetrievalConfig::default();
        let heat = HeatConfig::default();
        let uc = EnrichRequest {
            facts: &facts,
            sessions: &sessions,
            embedder: &embedder,
            reranker: &reranker,
            clock: &clock,
            retrieval_cfg: &retrieval,
            heat_cfg: &heat,
        };
        let original =
            vec![json!({"role": "user", "content": "explain rust ownership and borrowing"})];
        let out = uc
            .execute(original.clone(), &key(), &sid())
            .await
            .expect("ok");
        assert_eq!(
            out, original,
            "embedder None must fail-open to the original messages (no <smos-memory> block)"
        );
    }

    /// 3. an empty rerank result is fail-closed (step 8): once survivors reach
    ///    the reranker, a provider that returns nothing propagates as
    ///    `Err(UseCaseError::Provider(_))` so the HTTP handler maps it to 503
    ///    rather than silently shipping vector-order-only ranking.
    #[tokio::test]
    async fn enrich_fail_closes_with_provider_err_when_reranker_returns_empty() {
        let facts = InMemoryFacts::default();
        // Script a survivor so the pipeline reaches step 8 instead of
        // short-circuiting at "no vector hits".
        facts.script_search_hits(vec![survivable_hit()]);
        let sessions = InMemorySessions::default();
        let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker = EmptyReranker;
        let clock = FixedClock(now_ts());
        let retrieval = RetrievalConfig::default();
        let heat = HeatConfig::default();
        let uc = EnrichRequest {
            facts: &facts,
            sessions: &sessions,
            embedder: &embedder,
            reranker: &reranker,
            clock: &clock,
            retrieval_cfg: &retrieval,
            heat_cfg: &heat,
        };
        let original =
            vec![json!({"role": "user", "content": "explain rust ownership and borrowing"})];
        let result = uc.execute(original, &key(), &sid()).await;
        assert!(
            matches!(result, Err(UseCaseError::Provider(_))),
            "an empty rerank result must fail-closed as UseCaseError::Provider (HTTP 503)"
        );
    }
}
