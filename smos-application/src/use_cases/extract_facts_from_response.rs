//! `ExtractFactsFromResponse` — async fact-extraction pipeline (§4, §12).
//!
//! Runs entirely AFTER the client has received the response (the proxy spawns
//! it as a background `tokio::spawn` once `[DONE]` is reached), so extraction
//! latency never reaches the user. Extracted facts are stored as `Pending` and
//! handed to session-end processing (§5) for batch NLI + finalize.
//!
//! Extraction failure is non-fatal: the response is already gone, so the
//! spawn wrapper logs and skips (§12 — 3 retries with exponential backoff,
//! then give up gracefully).
//!
//! # Pipeline (mirrors `smos-poc/smos/response_pipeline.py::process_response_async`)
//!
//! 1. Kill-switch: `enable_response_extraction = false` → return 0 immediately.
//! 2. Strip SMOS-internal noise (session marker, `<smos-memory>` block, bare
//!    `sess_<id>`) via `noise_filter::clean` so the extractor never turns
//!    control metadata into a "fact".
//! 3. Append formatted tool calls to the input (facts may live in tool results
//!    — e.g. file content returned by a `read` tool).
//! 4. Short-circuit when the combined input is below `MIN_INPUT_CHARS` (short
//!    replies like "ok" carry no extractable signal).
//! 5. Retry the extractor up to 3 times with exponential backoff (1 s, 2 s —
//!    sleeps BETWEEN attempts, never after the last). `Unavailable` (model
//!    unreachable) skips gracefully; other errors retry.
//! 6. Embed each extracted fact (`embed_batch`) and persist it through the
//!    3-layer dedup flow:
//!    1. **Exact `FactId` match** — cross-session confirmation (the only path
//!       a single-session Pending fact can reach the accept threshold).
//!    2. **Semantic match** — cosine ≥ `extraction.dedup_cosine_threshold`
//!       backstops the exact match when the model rephrases a fact just
//!       enough to hash to a different id (non-deterministic extraction
//!       safety net).
//!    3. **No match** — store a new pending fact.
//! 7. Register newly-stored fact ids on the session's pending list.

use std::time::Duration;

use smos_domain::chat::ToolCall;
use smos_domain::config::{ConfidenceConfig, ExtractionConfig};
use smos_domain::{Embedding, Fact, FactId, MemoryKey, SessionId};

use crate::errors::{ProviderError, UseCaseError};
use crate::helpers::noise_filter;
use crate::ports::{
    Clock, Delay, EmbeddingProvider, FactRepository, LlmExtractor, SessionRepository,
};

/// Minimum combined input length (chars) below which extraction is skipped.
/// Matches the POC `_MIN_INPUT_CHARS = 15`: short replies ("ok", "done") carry
/// no extractable signal and waste a model round-trip.
pub const MIN_INPUT_CHARS: usize = 15;

/// Extraction attempts (§4 step 5). Backoff sleeps happen BETWEEN attempts,
/// never after the final one, so a permanently-failing model does not add a
/// stall beyond the last retry.
const EXTRACTION_ATTEMPTS: u32 = 3;

/// Backoff schedule: 1 s after attempt 1, 2 s after attempt 2 (no sleep after
/// attempt 3). Mirrors the POC `2 ** attempt` schedule.
const BACKOFF: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(2)];

/// Borrow-style bundle of every dependency the extraction pipeline needs.
///
/// Built inline at the spawn site (the adapter hands it owned clones of the
/// concrete adapters + a [`Delay`] impl); dropped right after
/// [`ExtractFactsFromResponse::execute`] returns.
pub struct ExtractFactsFromResponse<'a, FR, SR, EP, LE, C, D> {
    pub facts: &'a FR,
    pub sessions: &'a SR,
    pub embedder: &'a EP,
    pub extractor: &'a LE,
    pub clock: &'a C,
    pub delay: &'a D,
    pub confidence_cfg: &'a ConfidenceConfig,
    /// Semantic-dedup safety net for `persist_facts` step 2. Backstops the
    /// exact `FactId` match when the model rephrases a fact just enough to
    /// hash to a different id while the embedding is still near-identical.
    pub extraction_cfg: &'a ExtractionConfig,
    /// Kill-switch from `config.server.enable_response_extraction`. `false`
    /// short-circuits the whole pipeline to a no-op.
    pub enable_response_extraction: bool,
}

impl<'a, FR, SR, EP, LE, C, D> ExtractFactsFromResponse<'a, FR, SR, EP, LE, C, D>
where
    FR: FactRepository,
    SR: SessionRepository,
    EP: EmbeddingProvider,
    LE: LlmExtractor,
    C: Clock,
    D: Delay,
{
    /// Run the extraction pipeline. Returns the number of newly-stored pending
    /// facts (cross-session confirmations do not count — they update an
    /// existing fact rather than adding one).
    pub async fn execute(
        &self,
        content: &str,
        tool_calls: &[ToolCall],
        memory_key: &MemoryKey,
        session_id: &SessionId,
    ) -> Result<usize, UseCaseError> {
        // Step 1 — kill-switch.
        if !self.enable_response_extraction {
            return Ok(0);
        }

        // Steps 2 + 3 — clean noise, append tool calls.
        let mut input = noise_filter::clean(content);
        input.push_str(&format_tool_calls(tool_calls));

        // Step 4 — short-circuit on too-short input.
        if input.trim().chars().count() < MIN_INPUT_CHARS {
            tracing::debug!(
                len = input.len(),
                "extraction skipped: input below MIN_INPUT_CHARS"
            );
            return Ok(0);
        }

        // Step 5 — extract with retries.
        let raw_facts = self.extract_with_retries(&input, tool_calls).await?;
        if raw_facts.is_empty() {
            return Ok(0);
        }

        // Steps 6 + 7 — persist + register pending.
        let new_ids = self
            .persist_facts(&raw_facts, memory_key, session_id)
            .await?;
        if !new_ids.is_empty() {
            self.sessions.add_pending(session_id, &new_ids).await?;
        }
        Ok(new_ids.len())
    }

    /// Call the extractor up to [`EXTRACTION_ATTEMPTS`] times with exponential
    /// backoff between attempts. Returns the first non-empty fact list, or an
    /// empty list when every attempt came back empty / the model is down.
    async fn extract_with_retries(
        &self,
        input: &str,
        tool_calls: &[ToolCall],
    ) -> Result<Vec<String>, UseCaseError> {
        for attempt in 0..EXTRACTION_ATTEMPTS {
            match self.extractor.extract_facts(input, tool_calls).await {
                Ok(facts) if !facts.is_empty() => return Ok(facts),
                Ok(_) => self.maybe_sleep(attempt).await,
                // Unreachable model: retrying will not help, skip gracefully.
                Err(ProviderError::Unavailable(msg)) => {
                    tracing::warn!(error = %msg, "extractor unavailable; skipping (graceful)");
                    return Ok(Vec::new());
                }
                Err(e) if attempt + 1 < EXTRACTION_ATTEMPTS => {
                    tracing::warn!(attempt = attempt + 1, error = %e, "extraction failed; retrying");
                    self.maybe_sleep(attempt).await;
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(Vec::new())
    }

    /// Sleep the backoff duration for `attempt` (0-based), only between
    /// attempts — never after the final one. Delegates to the [`Delay`] port
    /// so the application layer stays runtime-agnostic.
    async fn maybe_sleep(&self, attempt: u32) {
        if let Some(delay) = BACKOFF.get(attempt as usize) {
            self.delay.delay(*delay).await;
        }
    }

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
    async fn persist_facts(
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
    async fn persist_one_fact(
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
        let fact = Fact::new_pending(
            raw,
            memory_key.clone(),
            session_id.clone(),
            emb,
            self.clock.now(),
            self.confidence_cfg.base,
        )?;
        self.facts.save(&fact).await?;
        Ok(Some(fact_id))
    }

    /// Run cross-session confirmation against `fact` and persist the updated
    /// provenance when the validation gate fired. `confirm_cross_session`
    /// returns `false` when the session is already in the provenance set —
    /// in that case no save is needed (the row is unchanged).
    async fn confirm_and_save(
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

/// Render tool calls as readable text appended to the extraction input.
///
/// Lets the extractor lift facts out of tool results (e.g. file content
/// returned by a `read_file` call). Mirrors the POC `_build_extraction_input`
/// "Tool calls:" trailer.
pub fn format_tool_calls(tool_calls: &[ToolCall]) -> String {
    if tool_calls.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n\nTool calls:");
    for call in tool_calls {
        out.push_str(&format!("\n- {}({})", call.name, call.arguments));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SearchHit, SearchHitMetadata};
    use smos_domain::{FactStatus, Heat, Timestamp};
    use std::collections::HashMap;
    use std::sync::Mutex;

    // ---- Fakes (classicist style: in-memory repos, scripted providers) ----

    #[derive(Clone)]
    struct FixedClock(Timestamp);
    impl Clock for FixedClock {
        fn now(&self) -> Timestamp {
            self.0
        }
    }

    /// No-op delay: retry backoff runs instantaneously in unit tests so the
    /// suite never pays the real 1 s + 2 s sleeps. Timing is verified
    /// end-to-end via the E2E suite against the real `TokioDelay` adapter.
    #[derive(Clone, Copy)]
    struct NoOpDelay;
    impl Delay for NoOpDelay {
        async fn delay(&self, _duration: Duration) {}
    }

    struct ScriptedExtractor {
        results: Mutex<Vec<Result<Vec<String>, ProviderError>>>,
        calls: Mutex<u32>,
    }
    impl ScriptedExtractor {
        fn new(results: Vec<Result<Vec<String>, ProviderError>>) -> Self {
            Self {
                results: Mutex::new(results),
                calls: Mutex::new(0),
            }
        }
        /// Number of times `extract_facts` was invoked.
        fn call_count(&self) -> u32 {
            *self.calls.lock().unwrap()
        }
    }
    impl LlmExtractor for ScriptedExtractor {
        async fn extract_facts(
            &self,
            _content: &str,
            _tool_calls: &[ToolCall],
        ) -> Result<Vec<String>, ProviderError> {
            *self.calls.lock().unwrap() += 1;
            let mut guard = self.results.lock().unwrap();
            if guard.is_empty() {
                Ok(Vec::new())
            } else {
                guard.remove(0)
            }
        }
    }

    struct ConstantEmbedder(Vec<f32>);
    impl EmbeddingProvider for ConstantEmbedder {
        async fn embed(&self, _text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
            Ok(Some(self.0.clone()))
        }
    }

    /// Embedding provider that records every `embed` call and returns a
    /// deterministic 1024-dim vector unique to the input text. Used to
    /// verify the extraction pipeline hands distinct embeddings to
    /// distinct facts (so Layer 2 dedup makes the right call). The
    /// default `embed_batch` implementation loops `embed`, so every call
    /// is recorded without an extra override.
    struct RecordingEmbedder {
        calls: std::sync::Arc<Mutex<Vec<String>>>,
    }
    impl RecordingEmbedder {
        fn new() -> (Self, std::sync::Arc<Mutex<Vec<String>>>) {
            let calls = std::sync::Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    calls: calls.clone(),
                },
                calls,
            )
        }
        fn vector_for(text: &str) -> Vec<f32> {
            // Stable, content-derived 1024-dim one-hot-ish vector: hash
            // the text into a single u64, use it as the index of a
            // single non-zero dimension. Different inputs ⇒ different
            // indices ⇒ cosine similarity 0 across distinct hashes.
            let hash = text
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            let mut vec = vec![0.0; 1024];
            vec[(hash as usize) % 1024] = 1.0;
            vec
        }
    }
    impl EmbeddingProvider for RecordingEmbedder {
        async fn embed(&self, text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
            self.calls.lock().unwrap().push(text.to_string());
            Ok(Some(Self::vector_for(text)))
        }
    }

    #[derive(Default, Clone)]
    struct InMemoryFacts {
        store: std::sync::Arc<Mutex<HashMap<String, Fact>>>,
        /// Optional scripted `search_for_dedup` response (semantic-dedup
        /// tests only). Empty by default so Layer 2 stays inert for
        /// exact-match + new-fact tests.
        ///
        /// `search_similar` (accepted-only) is intentionally left returning
        /// an empty Vec so the fake mirrors the production
        /// `SurrealStore` contract — Layer 2 MUST go through
        /// `search_for_dedup`, never `search_similar`, otherwise tests pass
        /// against the fake but fail against the real store (circular
        /// deadlock on pending facts).
        dedup_hits: std::sync::Arc<Mutex<Vec<SearchHit>>>,
    }
    impl InMemoryFacts {
        fn script_dedup_hits(&self, hits: Vec<SearchHit>) {
            *self.dedup_hits.lock().unwrap() = hits;
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
            Ok(self.store.lock().unwrap().get(id.as_str()).cloned())
        }
        async fn list_accepted(
            &self,
            _mk: &MemoryKey,
        ) -> Result<Vec<Fact>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn list_pending(
            &self,
            _mk: &MemoryKey,
        ) -> Result<Vec<Fact>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn list_memory_keys_for_session(
            &self,
            _session_id: &SessionId,
        ) -> Result<Vec<MemoryKey>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn search_similar(
            &self,
            _e: Vec<f32>,
            _mk: &MemoryKey,
            _l: usize,
        ) -> Result<Vec<SearchHit>, crate::errors::RepoError> {
            // Mirrors the SurrealStore contract: search_similar is
            // accepted-only. Tests that exercise Layer 2 go through
            // `search_for_dedup` below.
            Ok(Vec::new())
        }
        async fn search_for_dedup(
            &self,
            _e: Vec<f32>,
            _mk: &MemoryKey,
            _l: usize,
        ) -> Result<Vec<SearchHit>, crate::errors::RepoError> {
            Ok(self.dedup_hits.lock().unwrap().clone())
        }
        async fn update_heat_batch(
            &self,
            _ids: &[FactId],
            _mk: &MemoryKey,
            _h: Heat,
            _t: Timestamp,
        ) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
    }

    #[derive(Default, Clone)]
    struct RecordingSessions {
        pending: std::sync::Arc<Mutex<Vec<FactId>>>,
    }
    impl SessionRepository for RecordingSessions {
        async fn add_pending(
            &self,
            _id: &SessionId,
            fact_ids: &[FactId],
        ) -> Result<(), crate::errors::RepoError> {
            self.pending
                .lock()
                .unwrap()
                .extend(fact_ids.iter().cloned());
            Ok(())
        }
        async fn get_or_create(
            &self,
            _i: &SessionId,
            _m: &MemoryKey,
        ) -> Result<smos_domain::SessionState, crate::errors::RepoError> {
            unreachable!("not used by extraction")
        }
        async fn collect_expired(
            &self,
            _t: Duration,
        ) -> Result<Vec<(SessionId, smos_domain::SessionState)>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn snapshot_all(
            &self,
        ) -> Result<Vec<(SessionId, smos_domain::SessionState)>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn remove_pending_owned(
            &self,
            _i: &SessionId,
            _o: &[FactId],
        ) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
        async fn clear_session(&self, _i: &SessionId) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
        async fn dedup_and_mark(
            &self,
            _i: &SessionId,
            _m: &MemoryKey,
            _c: &[FactId],
        ) -> Result<Vec<FactId>, crate::errors::RepoError> {
            Ok(Vec::new())
        }
        async fn save(
            &self,
            _i: &SessionId,
            _s: &smos_domain::SessionState,
        ) -> Result<(), crate::errors::RepoError> {
            Ok(())
        }
    }

    fn mk() -> MemoryKey {
        MemoryKey::from_raw("proj").unwrap()
    }
    fn sid(tag: u8) -> SessionId {
        SessionId::from_raw(&format!("sess_{:012x}", tag as u64)).unwrap()
    }
    fn cfg() -> ConfidenceConfig {
        ConfidenceConfig::default()
    }
    fn extraction_cfg() -> ExtractionConfig {
        ExtractionConfig::default()
    }
    fn clock() -> FixedClock {
        FixedClock(Timestamp::from_unix_secs(1_700_000_000).unwrap())
    }

    #[allow(clippy::too_many_arguments)]
    fn build<'a>(
        facts: &'a InMemoryFacts,
        sessions: &'a RecordingSessions,
        extractor: &'a ScriptedExtractor,
        embedder: &'a ConstantEmbedder,
        clock: &'a FixedClock,
        cfg: &'a ConfidenceConfig,
        extraction_cfg: &'a ExtractionConfig,
    ) -> ExtractFactsFromResponse<
        'a,
        InMemoryFacts,
        RecordingSessions,
        ConstantEmbedder,
        ScriptedExtractor,
        FixedClock,
        NoOpDelay,
    > {
        ExtractFactsFromResponse {
            facts,
            sessions,
            embedder,
            extractor,
            clock,
            delay: &NO_OP_DELAY,
            confidence_cfg: cfg,
            extraction_cfg,
            enable_response_extraction: true,
        }
    }

    /// Singleton no-op delay — every unit test reuses it so the retry loop
    /// never actually sleeps.
    static NO_OP_DELAY: NoOpDelay = NoOpDelay;

    /// Shared fixture: embedder + clock + confidence config owned by the test
    /// so the returned use case can borrow them for its whole lifetime.
    struct Fix {
        embedder: ConstantEmbedder,
        clock: FixedClock,
        cfg: ConfidenceConfig,
        extraction_cfg: ExtractionConfig,
    }
    impl Fix {
        fn new() -> Self {
            Self {
                embedder: ConstantEmbedder(vec![0.1, 0.2, 0.3]),
                clock: clock(),
                cfg: cfg(),
                extraction_cfg: extraction_cfg(),
            }
        }
    }

    #[test]
    fn format_tool_calls_renders_name_and_arguments() {
        let calls = vec![ToolCall {
            name: "read_file".into(),
            arguments: smos_domain::chat::ToolArguments::from_json(r#"{"path":"auth.rs"}"#),
        }];
        assert_eq!(
            format_tool_calls(&calls),
            "\n\nTool calls:\n- read_file({\"path\":\"auth.rs\"})"
        );
    }

    #[test]
    fn format_tool_calls_empty_returns_empty_string() {
        assert_eq!(format_tool_calls(&[]), "");
    }

    #[tokio::test]
    async fn execute_disabled_returns_zero_without_calling_extractor() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        let extractor = ScriptedExtractor::new(vec![Err(ProviderError::Unavailable(
            "must not be called".into(),
        ))]);
        let fix = Fix::new();
        let mut uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );
        uc.enable_response_extraction = false;

        let n = uc
            .execute("TTL=10 prevents refresh loop", &[], &mk(), &sid(1))
            .await
            .unwrap();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn execute_short_input_returns_zero_without_calling_extractor() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        let extractor = ScriptedExtractor::new(vec![Err(ProviderError::Unavailable(
            "must not be called".into(),
        ))]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        // "ok" is 2 chars < MIN_INPUT_CHARS (15).
        let n = uc.execute("ok", &[], &mk(), &sid(1)).await.unwrap();
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn execute_saves_new_pending_fact_and_registers_it() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        let extractor = ScriptedExtractor::new(vec![Ok(vec![
            "TTL=10 prevents the token refresh loop".to_string(),
        ])]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let n = uc
            .execute("we changed TTL to 10 to stop the loop", &[], &mk(), &sid(1))
            .await
            .unwrap();

        assert_eq!(n, 1);
        let fact = facts
            .store
            .lock()
            .unwrap()
            .get(FactId::from_content("TTL=10 prevents the token refresh loop").as_str())
            .cloned()
            .expect("fact saved");
        assert_eq!(fact.status(), FactStatus::Pending);
        assert_eq!(
            sessions.pending.lock().unwrap().len(),
            1,
            "fact registered on session pending list"
        );
    }

    #[tokio::test]
    async fn execute_unavailable_extractor_skips_gracefully() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        // A single Unavailable result: the use case must return Ok(0)
        // immediately WITHOUT retrying. `call_count == 1` (not 3) is the
        // invariant that proves the early-exit on Unavailable.
        let extractor = ScriptedExtractor::new(vec![Err(ProviderError::Unavailable(
            "connection refused".into(),
        ))]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let n = uc
            .execute("some real content long enough", &[], &mk(), &sid(1))
            .await
            .unwrap();
        assert_eq!(n, 0);
        assert_eq!(
            extractor.call_count(),
            1,
            "Unavailable must skip retries — extractor called exactly once"
        );
        assert!(
            facts.store.lock().unwrap().is_empty(),
            "no fact persisted on graceful skip"
        );
    }

    #[tokio::test]
    async fn execute_retries_on_request_failed_then_succeeds() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        let extractor = ScriptedExtractor::new(vec![
            Err(ProviderError::RequestFailed("500".into())),
            Err(ProviderError::RequestFailed("500".into())),
            Ok(vec!["auth.rs uses JWT for tokens".to_string()]),
        ]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let n = uc
            .execute("the auth module uses JWT", &[], &mk(), &sid(1))
            .await
            .unwrap();
        assert_eq!(n, 1);
    }

    #[tokio::test]
    async fn execute_gives_up_after_all_attempts_fail() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        let extractor = ScriptedExtractor::new(vec![
            Err(ProviderError::RequestFailed("500".into())),
            Err(ProviderError::RequestFailed("500".into())),
            Err(ProviderError::RequestFailed("500".into())),
        ]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let result = uc
            .execute("content long enough to pass gate", &[], &mk(), &sid(1))
            .await;
        assert!(result.is_err(), "final failure propagates as Err");
        assert!(facts.store.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn execute_strips_smos_noise_before_extraction() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        let extractor = ScriptedExtractor::new(vec![Ok(vec!["a clean fact".to_string()])]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let content = "real content about the deployment\n<!-- smos:sess_abcdef012345 -->\n<smos-memory session=\"s\">x</smos-memory>";
        let n = uc.execute(content, &[], &mk(), &sid(1)).await.unwrap();
        assert_eq!(n, 1);
    }

    #[tokio::test]
    async fn execute_cross_session_confirms_existing_fact() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();

        // Seed a fact from session 1.
        let first = Fact::new_pending(
            "shared fact content here",
            mk(),
            sid(1),
            Embedding::new(vec![1.0]).unwrap(),
            Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap();
        let fid = first.id().clone();
        facts
            .store
            .lock()
            .unwrap()
            .insert(fid.as_str().to_string(), first);

        let extractor =
            ScriptedExtractor::new(vec![Ok(vec!["shared fact content here".to_string()])]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        // Same fact observed from session 2 → confirmation, not a new fact.
        let n = uc
            .execute("shared fact content here", &[], &mk(), &sid(2))
            .await
            .unwrap();
        assert_eq!(n, 0, "confirmation does not count as a new fact");

        let confirmed = facts
            .store
            .lock()
            .unwrap()
            .get(fid.as_str())
            .cloned()
            .expect("fact still present");
        assert_eq!(
            confirmed.source_sessions().distinct_count(),
            2,
            "provenance grew to two sessions"
        );
        assert!(
            sessions.pending.lock().unwrap().is_empty(),
            "confirmation must not register on the pending list"
        );
    }

    // ---- Layer 2 — semantic dedup safety net ----

    /// Build a `SearchHit` whose `metadata.distance` corresponds to the given
    /// cosine similarity. The store reports cosine distance, so
    /// `distance = 1.0 - similarity` (Layer 2 inverts it back).
    fn hit_for(fact: &Fact, similarity: f32, mk: MemoryKey) -> SearchHit {
        let metadata = SearchHitMetadata {
            status: "pending".into(),
            confidence: 0.5,
            valid_until: None,
            heat_base: 1.0,
            last_access_at: 1_700_000_000.0,
            distance: Some(1.0 - similarity),
        };
        SearchHit {
            id: fact.id().clone(),
            document: fact.content().to_string(),
            memory_key: mk,
            metadata,
        }
    }

    /// Rephrased re-observation (different SHA1) is caught at Layer 2 via
    /// cosine similarity and routed through cross-session confirmation
    /// instead of leaving the fact stuck at single-source confidence.
    #[tokio::test]
    async fn persist_facts_layer2_semantic_match_confirms_existing_fact() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();

        // Seed an existing fact from session 1 under one phrasing.
        let stored = Fact::new_pending(
            "the token cache uses TTL=60 to avoid stale entries",
            mk(),
            sid(1),
            Embedding::new(vec![1.0]).unwrap(),
            Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap();
        let stored_id = stored.id().clone();
        facts
            .store
            .lock()
            .unwrap()
            .insert(stored_id.as_str().to_string(), stored);

        // The extractor rephrased the same concept differently → its FactId
        // will differ, so Layer 1 (exact match) misses. Layer 2 must catch
        // it because the scripted `search_similar` returns the stored fact
        // with similarity 0.98 (above the 0.95 threshold).
        facts.script_dedup_hits(vec![hit_for(
            &facts
                .store
                .lock()
                .unwrap()
                .get(stored_id.as_str())
                .cloned()
                .expect("seeded fact"),
            0.98,
            mk(),
        )]);

        let rephrased = "token cache TTL is 60 to prevent stale entries";
        let extractor = ScriptedExtractor::new(vec![Ok(vec![rephrased.to_string()])]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let n = uc.execute(rephrased, &[], &mk(), &sid(2)).await.unwrap();

        assert_eq!(
            n, 0,
            "semantic duplicate must confirm, not create a new fact"
        );
        let confirmed = facts
            .store
            .lock()
            .unwrap()
            .get(stored_id.as_str())
            .cloned()
            .expect("seeded fact still present");
        assert_eq!(
            confirmed.source_sessions().distinct_count(),
            2,
            "semantic match grows provenance to two sessions"
        );
        assert!(
            facts
                .store
                .lock()
                .unwrap()
                .get(FactId::from_content(rephrased).as_str())
                .is_none(),
            "no new fact id created for the rephrased variant"
        );
        assert!(
            sessions.pending.lock().unwrap().is_empty(),
            "semantic confirmation must not register on the pending list"
        );
    }

    /// Below the cosine threshold the semantic layer must NOT collapse two
    /// different phrasings: the new fact is stored as a separate pending
    /// entry (Layer 3 fallback).
    #[tokio::test]
    async fn persist_facts_layer2_below_threshold_creates_new_fact() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();

        let stored = Fact::new_pending(
            "auth module uses Argon2id for password hashing",
            mk(),
            sid(1),
            Embedding::new(vec![1.0]).unwrap(),
            Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap();
        let stored_id = stored.id().clone();
        facts
            .store
            .lock()
            .unwrap()
            .insert(stored_id.as_str().to_string(), stored);

        // Similarity 0.80 < 0.95 threshold → Layer 2 must NOT fire.
        facts.script_dedup_hits(vec![hit_for(
            &facts
                .store
                .lock()
                .unwrap()
                .get(stored_id.as_str())
                .cloned()
                .expect("seeded fact"),
            0.80,
            mk(),
        )]);

        let new_content = "TLS handshake failure in the upstream pool";
        let extractor = ScriptedExtractor::new(vec![Ok(vec![new_content.to_string()])]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let n = uc.execute(new_content, &[], &mk(), &sid(2)).await.unwrap();

        assert_eq!(n, 1, "below-threshold similarity must create a new fact");
        let new_id = FactId::from_content(new_content);
        assert!(
            facts.store.lock().unwrap().contains_key(new_id.as_str()),
            "new fact persisted under its own FactId"
        );
        assert_eq!(
            sessions.pending.lock().unwrap().len(),
            1,
            "new fact registered on the pending list"
        );
    }

    /// `metadata.distance = None` (store did not surface a distance) must
    /// NOT collapse two phrasings even when the underlying row would
    /// otherwise match — Layer 2 cannot make a decision without a distance,
    /// so it falls through to Layer 3 (create new). This guards against
    /// silent dedup when a future adapter forgets to populate distance.
    #[tokio::test]
    async fn persist_facts_layer2_missing_distance_falls_through_to_new_fact() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();

        let stored = Fact::new_pending(
            "config reload triggers a graceful drain",
            mk(),
            sid(1),
            Embedding::new(vec![1.0]).unwrap(),
            Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap();
        let stored_id = stored.id().clone();
        facts
            .store
            .lock()
            .unwrap()
            .insert(stored_id.as_str().to_string(), stored);

        let mut hit = hit_for(
            &facts
                .store
                .lock()
                .unwrap()
                .get(stored_id.as_str())
                .cloned()
                .expect("seeded fact"),
            1.0,
            mk(),
        );
        hit.metadata.distance = None;
        facts.script_dedup_hits(vec![hit]);

        let new_content = "config reload drains gracefully on SIGHUP";
        let extractor = ScriptedExtractor::new(vec![Ok(vec![new_content.to_string()])]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let n = uc.execute(new_content, &[], &mk(), &sid(2)).await.unwrap();

        assert_eq!(
            n, 1,
            "missing distance must not collapse — fall through to new fact"
        );
    }

    /// Tunable threshold: when the operator lowers
    /// `dedup_cosine_threshold`, an above-threshold hit at 0.85 (which the
    /// default 0.95 would have rejected) now confirms the existing fact.
    /// Confirms the config field actually flows into the dedup decision.
    #[tokio::test]
    async fn persist_facts_layer2_threshold_lowered_collapses_0_85_pair() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();

        let stored = Fact::new_pending(
            "indexer batches at most 1024 documents per commit",
            mk(),
            sid(1),
            Embedding::new(vec![1.0]).unwrap(),
            Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap();
        let stored_id = stored.id().clone();
        facts
            .store
            .lock()
            .unwrap()
            .insert(stored_id.as_str().to_string(), stored);

        // Similarity 0.85 — above the operator-lowered 0.80 threshold.
        facts.script_dedup_hits(vec![hit_for(
            &facts
                .store
                .lock()
                .unwrap()
                .get(stored_id.as_str())
                .cloned()
                .expect("seeded fact"),
            0.85,
            mk(),
        )]);

        let rephrased = "the indexer caps batches at 1024 documents";
        let extractor = ScriptedExtractor::new(vec![Ok(vec![rephrased.to_string()])]);
        let mut fix = Fix::new();
        fix.extraction_cfg = ExtractionConfig {
            dedup_cosine_threshold: 0.80,
        };
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let n = uc.execute(rephrased, &[], &mk(), &sid(2)).await.unwrap();

        assert_eq!(
            n, 0,
            "lowered threshold collapses the 0.85 pair via semantic match"
        );
        let confirmed = facts
            .store
            .lock()
            .unwrap()
            .get(stored_id.as_str())
            .cloned()
            .expect("seeded fact still present");
        assert_eq!(
            confirmed.source_sessions().distinct_count(),
            2,
            "semantic collapse grows provenance"
        );
    }

    // -----------------------------------------------------------------------
    // RecordingEmbedder — verify distinct facts get distinct vectors
    // -----------------------------------------------------------------------

    /// Build a use case backed by a `RecordingEmbedder`. The default
    /// `embed_batch` loops `embed`, so every fact handed to the pipeline
    /// produces one recorded call.
    #[allow(clippy::too_many_arguments)]
    fn build_with_recording_embedder<'a>(
        facts: &'a InMemoryFacts,
        sessions: &'a RecordingSessions,
        extractor: &'a ScriptedExtractor,
        embedder: &'a RecordingEmbedder,
        clock: &'a FixedClock,
        cfg: &'a ConfidenceConfig,
        extraction_cfg: &'a ExtractionConfig,
    ) -> ExtractFactsFromResponse<
        'a,
        InMemoryFacts,
        RecordingSessions,
        RecordingEmbedder,
        ScriptedExtractor,
        FixedClock,
        NoOpDelay,
    > {
        ExtractFactsFromResponse {
            facts,
            sessions,
            embedder,
            extractor,
            clock,
            delay: &NO_OP_DELAY,
            confidence_cfg: cfg,
            extraction_cfg,
            enable_response_extraction: true,
        }
    }

    /// The extraction pipeline MUST hand distinct embeddings to distinct
    /// extracted facts — otherwise Layer 2 dedup would collapse two
    /// unrelated facts (cosine ~1) and silently lose data. The
    /// `RecordingEmbedder` returns a content-derived one-hot vector so
    /// two distinct facts end up with cosine similarity 0; this test
    /// pins that contract by checking the recorded calls + the resulting
    /// store state.
    #[tokio::test]
    async fn recording_embedder_yields_distinct_vectors_for_distinct_facts() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        let extractor = ScriptedExtractor::new(vec![Ok(vec![
            "alpha configuration directive".to_string(),
            "beta configuration directive".to_string(),
        ])]);
        let (embedder, calls) = RecordingEmbedder::new();
        let clock = clock();
        let cfg = cfg();
        let extraction_cfg = extraction_cfg();
        let uc = build_with_recording_embedder(
            &facts,
            &sessions,
            &extractor,
            &embedder,
            &clock,
            &cfg,
            &extraction_cfg,
        );

        let n = uc
            .execute("content covering both directives", &[], &mk(), &sid(1))
            .await
            .unwrap();
        assert_eq!(n, 2, "two distinct facts persisted");
        assert_eq!(
            calls.lock().unwrap().len(),
            2,
            "embedder called once per extracted fact"
        );
        // Two distinct FactIds in the store → no collapse happened.
        let id_a = FactId::from_content("alpha configuration directive");
        let id_b = FactId::from_content("beta configuration directive");
        assert!(facts.store.lock().unwrap().contains_key(id_a.as_str()));
        assert!(facts.store.lock().unwrap().contains_key(id_b.as_str()));
    }

    // -----------------------------------------------------------------------
    // Empty raw fact in extraction batch — must not crash
    // -----------------------------------------------------------------------

    /// An empty string in the extracted facts list surfaces as `Err` —
    /// the pipeline propagates the underlying domain failure rather than
    /// silently dropping the empty entry or persisting a malformed fact.
    /// The whole batch fails: the call site (background extraction task)
    /// logs the error and the facts that would have been persisted in
    /// the same batch are lost too. A future refactor that filters
    /// empty raw facts BEFORE the `Fact::new_pending` constructor would
    /// change this test from `is_err()` to `n == 1`; that change is
    /// intentional and the test should be updated alongside it.
    #[tokio::test]
    async fn execute_propagates_err_when_batch_contains_empty_raw_fact() {
        let facts = InMemoryFacts::default();
        let sessions = RecordingSessions::default();
        // Mix one empty + one real fact in the extractor output.
        let extractor = ScriptedExtractor::new(vec![Ok(vec![
            String::new(),
            "real fact that should still persist".to_string(),
        ])]);
        let fix = Fix::new();
        let uc = build(
            &facts,
            &sessions,
            &extractor,
            &fix.embedder,
            &fix.clock,
            &fix.cfg,
            &fix.extraction_cfg,
        );

        let result = uc
            .execute(
                "content long enough to clear MIN_INPUT_CHARS",
                &[],
                &mk(),
                &sid(1),
            )
            .await;
        assert!(
            result.is_err(),
            "empty raw fact must surface as Err (the only safe non-silent path)"
        );
    }
}
