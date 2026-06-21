//! `LlmExtractor` port — parse facts out of LLM responses.
//!
//! The extraction use case pre-combines the assistant `response_content` with
//! formatted tool calls (see `extract_facts_from_response::format_tool_calls`)
//! so the adapter receives a single ready-to-prompt string. The `tool_calls`
//! parameter is carried alongside for adapters that want structured access to
//! the calls (e.g. a future extractor that special-cases tool results), but
//! the canonical production adapter treats `response_content` as the complete
//! input. The provider adapter is responsible for prompt construction; this
//! trait only models the (response, tool_calls) → facts call, returning
//! canonical English statements ready for `Fact::new_pending`.

use smos_domain::chat::ToolCall;

use crate::errors::ProviderError;

/// Fact-extraction boundary (LLM-driven, prompt-based).
pub trait LlmExtractor {
    /// Extract zero or more canonical-English fact strings from an assistant
    /// response. `response_content` is the combined extraction input (assistant
    /// text + formatted tool calls); `tool_calls` is the raw structured calls
    /// for adapters that prefer to handle them directly.
    async fn extract_facts(
        &self,
        response_content: &str,
        tool_calls: &[ToolCall],
    ) -> Result<Vec<String>, ProviderError>;
}
