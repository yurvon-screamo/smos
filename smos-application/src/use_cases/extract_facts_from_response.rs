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
use smos_domain::{MemoryKey, SessionId};

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

pub mod dedup;
pub mod persist;

#[cfg(test)]
mod tests;
