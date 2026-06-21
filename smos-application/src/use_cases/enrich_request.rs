//! `EnrichRequest` — full enrichment pipeline (§3, §7, §12).
//!
//! Inputs: OpenAI-shaped messages, `memory_key`, `session_id`, and references
//! to every port the pipeline needs (`FactRepository`, `SessionRepository`,
//! `EmbeddingProvider`, `RerankProvider`, `Clock`).
//!
//! Output: an enriched messages array (with a `<smos-memory>` block prepended
//! to the first message) or, on any recoverable failure, the original messages
//! unchanged. The use case NEVER blocks a request because of memory
//! unavailability (§12 fail-open) — that contract is enforced here, not in the
//! port adapters, so callers cannot accidentally break it. The signature is
//! infallible (`Vec<Value>`, not `Result`) precisely because every port-level
//! error is already fail-open to the original messages inside `execute`.
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
//! 8. Rerank survivors with the cross-encoder (REQUIRED for production
//!    quality; an empty result or provider error degrades to top-N survivors
//!    and logs at WARN — never silently).
//! 9. Fallback to top-N survivors when reranking returned nothing
//!    (degraded mode, logged at WARN).
//! 10. Session dedup — drop facts already injected into this session.
//! 11. Short-circuit when no new facts survived dedup.
//! 12. Build the `<smos-memory>` block from the new facts.
//! 13. Inject the block into the first message and return.

use std::collections::HashSet;

use serde_json::Value;
use smos_domain::config::{HeatConfig, RetrievalConfig};
use smos_domain::{FactId, FactStatus, Heat, MemoryKey, SessionId, Timestamp};

use crate::helpers::memory_block::{self, MemoryBlockEntry};
use crate::helpers::request_enricher;
use crate::helpers::retrieval_planner::{self, RetrievalHit};
use crate::helpers::topic_extractor;
use crate::ports::{Clock, EmbeddingProvider, FactRepository, RerankProvider, SessionRepository};
use crate::types::{SearchHit, enrichment_messages_from_json};

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
    /// Always returns a messages array (fail-open per §12). On any recoverable
    /// failure the original `messages` are returned unchanged; the only way to
    /// *lose* the original messages here would be a bug inside this function,
    /// which the infallible signature makes impossible to express at the
    /// type level. [`HandleChatCompletion`] therefore consumes the request's
    /// messages via `std::mem::take` and assigns the enriched result back
    /// unconditionally — no `Err` arm to forget to repopulate.
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
    ) -> Vec<Value> {
        // Read-only typed projection: used only for topic extraction here
        // (session-marker detection runs in `HandleChatCompletion`, which
        // also uses the read-only projection). NEVER re-serialised — the
        // raw `messages: Vec<Value>` stays the source of truth for the
        // mutation + return path.
        let typed_projection = enrichment_messages_from_json(&messages);

        // Step 1 + 2 — short-circuit on empty / too-short topic. POC parity:
        // `if len(topic.strip()) < min_topic_chars`. Trimming prevents
        // whitespace-only topics (e.g. `"   "`) from passing the gate and
        // producing a garbage embedding downstream.
        let topic = topic_extractor::extract_from_messages(&typed_projection);
        let trimmed_len = topic.trim().chars().count();
        if trimmed_len < self.retrieval_cfg.min_topic_chars {
            tracing::debug!(
                chars = trimmed_len,
                "enrichment skipped: topic below min_topic_chars"
            );
            return messages;
        }

        // Step 3 — embed. None and errors are both fail-open.
        let embedding = match self.embedder.embed(&topic).await {
            Ok(Some(v)) => v,
            Ok(None) => {
                tracing::warn!("embedder returned None; skipping enrichment (fail-open)");
                return messages;
            }
            Err(e) => {
                tracing::warn!(error = %e, "embedder error; skipping enrichment (fail-open)");
                return messages;
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
                return messages;
            }
        };
        if hits.is_empty() {
            tracing::info!(memory_key = %memory_key, "no vector hits; skipping enrichment");
            return messages;
        }

        // Step 5 — pre-filter + heat post-filter (pure domain).
        let now = self.clock.now();
        let survivors = prefilter(hits, self.retrieval_cfg, self.heat_cfg, now);

        // Step 6 — short-circuit when no survivors remain.
        if survivors.is_empty() {
            return messages;
        }

        // Step 7 — heat boost. Best-effort: a failure here is logged but does
        // not abort enrichment (the rerank/dedup still works on stale heat).
        self.boost_heat(&survivors, memory_key, now).await;

        // Step 8 — rerank. Step 9 — empty result falls back to top-N survivors.
        let ranked_facts = self.rerank_survivors(&topic, &survivors).await;

        // Step 10 — short-circuit when reranking produced nothing usable.
        if ranked_facts.is_empty() {
            return messages;
        }

        // Step 11 — session dedup (atomic via SessionRepository::dedup_and_mark).
        let new_facts = self
            .dedup_against_session(&ranked_facts, session_id, memory_key)
            .await;

        // Step 12 — short-circuit when no new facts survived dedup.
        if new_facts.is_empty() {
            return messages;
        }

        // Step 13 — build memory block + inject via the JSON-path entry
        // point. `inject_value` operates on the raw `Value` and mutates
        // only `messages[0].content`, preserving every other per-message
        // field the typed DTO does not model (name, tool_call_id,
        // refusal, image_url parts, …). Round-tripping the typed DTO
        // here would silently drop those fields.
        let block = build_memory_block(&new_facts, session_id, memory_key);
        let messages_value = Value::Array(messages);
        let enriched = request_enricher::inject_value(&messages_value, &block);
        match enriched {
            Value::Array(arr) => arr,
            // Defensive: `inject_value` is documented to always echo the
            // input shape (array in → array out); anything else indicates
            // a domain bug.
            other => vec![other],
        }
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

    /// Rerank survivors with the cross-encoder; on empty result fall back to
    /// the first `top_k_final` survivors in retrieval order (parity with POC
    /// `enrich.py::_rerank_candidates` fallback branch).
    ///
    /// The reranker is REQUIRED for production-quality enrichment (§3 step 8).
    /// Both failure modes — `Err` from the provider and an empty `Vec` — fall
    /// back to top-N survivors so the pipeline stays fail-open and never
    /// blocks a request, but each branch logs at WARN with an explicit
    /// "degraded mode" hint so an operator running without the reranker sees
    /// the drop in retrieval quality in the logs rather than silently
    /// shipping vector-order-only ranking.
    async fn rerank_survivors(&self, topic: &str, survivors: &[RetrievalHit]) -> Vec<RetrievalHit> {
        let documents: Vec<String> = survivors.iter().map(|s| s.document.clone()).collect();
        let ranked = match self
            .reranker
            .rerank(topic, &documents, self.retrieval_cfg.top_k_final)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "reranker unavailable; using survivor fallback \
                     (degraded mode) — check llama.cpp server"
                );
                return survivors
                    .iter()
                    .take(self.retrieval_cfg.top_k_final)
                    .cloned()
                    .collect();
            }
        };
        if ranked.is_empty() {
            tracing::warn!(
                "reranker returned empty results; using survivor fallback \
                 (degraded mode) — verify the reranker model and query"
            );
            return survivors
                .iter()
                .take(self.retrieval_cfg.top_k_final)
                .cloned()
                .collect();
        }
        ranked
            .into_iter()
            .filter_map(|r| survivors.get(r.index).cloned())
            .collect()
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
    let retrieval_hits: Vec<RetrievalHit> = hits.into_iter().filter_map(hit_to_retrieval).collect();
    retrieval_planner::prefilter_and_heat(&retrieval_hits, retrieval_cfg, heat_cfg, now)
}

/// Map a single `SearchHit` to a `RetrievalHit`. Drops rows whose typed fields
/// cannot be reconstructed (status / confidence / heat) so a corrupt row never
/// poisons the pipeline — it is logged and skipped.
fn hit_to_retrieval(hit: SearchHit) -> Option<RetrievalHit> {
    let status = match parse_fact_status(&hit.metadata.status) {
        Some(s) => s,
        None => {
            tracing::warn!(status = %hit.metadata.status, "unparseable status; dropping hit");
            return None;
        }
    };
    let confidence = match smos_domain::Confidence::new(hit.metadata.confidence) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "out-of-range confidence; dropping hit");
            return None;
        }
    };
    let heat = match Heat::new(hit.metadata.heat_base) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(error = %e, "out-of-range heat_base; dropping hit");
            return None;
        }
    };
    let last_access_at = match Timestamp::from_unix_secs(hit.metadata.last_access_at as i64) {
        Ok(ts) => ts,
        Err(e) => {
            tracing::warn!(error = %e, "out-of-range last_access_at; dropping hit");
            return None;
        }
    };
    // `valid_until` is stored as an ISO-8601 string by the adapter; an absent
    // tombstone (`None`) means the fact is still current. Parse failures are
    // logged and treated as `None` so a corrupt row never blocks retrieval.
    let valid_until = hit
        .metadata
        .valid_until
        .as_deref()
        .and_then(parse_iso_timestamp);
    Some(RetrievalHit {
        id: hit.id,
        document: hit.document,
        memory_key: hit.memory_key,
        status,
        confidence,
        valid_until,
        heat_base: heat,
        last_access_at,
    })
}

/// Map a wire-formatted status string to a `FactStatus`. Compares against
/// each canonical lowercase token (`FactStatus::as_str`) so the wire contract
/// has a single source of truth in the domain. Returns `None` for unknown
/// values (logged by the caller).
fn parse_fact_status(s: &str) -> Option<FactStatus> {
    [
        FactStatus::Pending,
        FactStatus::Accepted,
        FactStatus::Rejected,
    ]
    .into_iter()
    .find(|candidate| s == candidate.as_str())
}

/// Parse an ISO-8601 string into a `Timestamp` via the `time` crate's
/// `Rfc3339` parser. Returns `None` on any failure.
fn parse_iso_timestamp(s: &str) -> Option<Timestamp> {
    use time::OffsetDateTime;
    let odt = OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()?;
    Timestamp::from_unix_secs(odt.unix_timestamp()).ok()
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
    use smos_domain::{MemoryKey, SessionId};

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

    // -----------------------------------------------------------------------
    // parse_fact_status — wire-format → enum mapping
    // -----------------------------------------------------------------------

    #[test]
    fn parse_fact_status_recognises_canonical_tokens() {
        assert_eq!(parse_fact_status("pending"), Some(FactStatus::Pending));
        assert_eq!(parse_fact_status("accepted"), Some(FactStatus::Accepted));
        assert_eq!(parse_fact_status("rejected"), Some(FactStatus::Rejected));
    }

    #[test]
    fn parse_fact_status_rejects_unknown_tokens() {
        assert_eq!(parse_fact_status("invalid"), None);
        assert_eq!(parse_fact_status(""), None);
    }

    #[test]
    fn parse_fact_status_is_case_sensitive() {
        // The wire contract is the lowercase token emitted by `as_str`; a
        // case mismatch is treated as unknown so the row is dropped rather
        // than silently re-interpreted.
        assert_eq!(parse_fact_status("Accepted"), None);
        assert_eq!(parse_fact_status("ACCEPTED"), None);
    }

    // -----------------------------------------------------------------------
    // parse_iso_timestamp — Rfc3339 → Timestamp mapping
    // -----------------------------------------------------------------------

    #[test]
    fn parse_iso_timestamp_accepts_rfc3339_utc() {
        let ts = parse_iso_timestamp("2025-06-18T12:00:00Z").expect("valid rfc3339");
        assert_eq!(ts.as_unix_secs(), 1_750_248_000);
    }

    #[test]
    fn parse_iso_timestamp_accepts_offset_form() {
        let ts = parse_iso_timestamp("2025-06-18T12:00:00+00:00").expect("valid offset");
        assert_eq!(ts.as_unix_secs(), 1_750_248_000);
    }

    #[test]
    fn parse_iso_timestamp_rejects_malformed_strings() {
        assert_eq!(parse_iso_timestamp("not a date"), None);
        assert_eq!(parse_iso_timestamp(""), None);
        assert_eq!(parse_iso_timestamp("2025-06-18"), None);
    }

    // -----------------------------------------------------------------------
    // hit_to_retrieval — SearchHit → RetrievalHit projection
    // -----------------------------------------------------------------------

    fn sample_hit(
        status: &str,
        confidence: f32,
        heat_base: f32,
        last_access_at: f32,
        valid_until: Option<&str>,
    ) -> SearchHit {
        SearchHit {
            id: FactId::from_raw("fact_0123456789abcdef").expect("fact id"),
            document: "doc".into(),
            memory_key: MemoryKey::from_raw("origa").expect("memory key"),
            metadata: crate::types::SearchHitMetadata {
                status: status.into(),
                confidence,
                valid_until: valid_until.map(str::to_string),
                heat_base,
                last_access_at,
                distance: Some(0.1),
            },
        }
    }

    #[test]
    fn hit_to_retrieval_maps_well_formed_hit() {
        let hit = sample_hit("accepted", 0.85, 1.0, 1_700_000_000.0, None);
        let r = hit_to_retrieval(hit).expect("mapped");
        assert_eq!(r.status, FactStatus::Accepted);
        assert!((r.confidence.value() - 0.85).abs() < 1e-6);
        assert!((r.heat_base.value() - 1.0).abs() < 1e-6);
        assert_eq!(r.last_access_at.as_unix_secs(), 1_700_000_000);
        assert!(r.valid_until.is_none());
    }

    #[test]
    fn hit_to_retrieval_carries_valid_until_tombstone() {
        let hit = sample_hit(
            "accepted",
            0.9,
            0.5,
            1_700_000_000.0,
            Some("2025-12-31T00:00:00Z"),
        );
        let r = hit_to_retrieval(hit).expect("mapped");
        assert!(r.valid_until.is_some());
    }

    #[test]
    fn hit_to_retrieval_drops_hit_with_unknown_status() {
        let hit = sample_hit("weird", 0.9, 1.0, 1_700_000_000.0, None);
        assert!(hit_to_retrieval(hit).is_none());
    }

    #[test]
    fn hit_to_retrieval_drops_hit_with_out_of_range_confidence() {
        // 1.5 is outside [0,1]; Confidence::new rejects it.
        let hit = sample_hit("accepted", 1.5, 1.0, 1_700_000_000.0, None);
        assert!(hit_to_retrieval(hit).is_none());
    }

    #[test]
    fn hit_to_retrieval_drops_hit_with_out_of_range_heat() {
        let hit = sample_hit("accepted", 0.9, 2.0, 1_700_000_000.0, None);
        assert!(hit_to_retrieval(hit).is_none());
    }

    #[test]
    fn hit_to_retrieval_drops_hit_with_out_of_range_last_access_at() {
        // `f32::INFINITY` saturates to `i64::MAX` on `as i64` cast; that
        // overflows the `OffsetDateTime` year range and the typed timestamp
        // rejects it, so the row is dropped.
        let hit = sample_hit("accepted", 0.9, 1.0, f32::INFINITY, None);
        assert!(hit_to_retrieval(hit).is_none());
    }

    #[test]
    fn hit_to_retrieval_treats_malformed_valid_until_as_none() {
        // A corrupt tombstone string must not poison the row — it is logged
        // and the fact stays current (None tombstone).
        let hit = sample_hit("accepted", 0.9, 1.0, 1_700_000_000.0, Some("not-a-date"));
        let r = hit_to_retrieval(hit).expect("mapped despite malformed valid_until");
        assert!(r.valid_until.is_none());
    }

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
