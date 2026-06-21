//! `NoopExtractor` — `LlmExtractor` that always returns an empty fact list.
//!
//! Kept as a public test utility (and as the simplest possible reference
//! implementation of the port) after Slice-5 wired the real
//! [`OllamaExtractor`](crate::OllamaExtractor) into the production binary and
//! the test `AppState`. Test suites that do not care about extraction still
//! use this to keep their wiring minimal.

use smos_application::errors::ProviderError;
use smos_application::ports::LlmExtractor;
use smos_domain::chat::ToolCall;

/// No-op extractor: always returns an empty fact list.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopExtractor;

impl LlmExtractor for NoopExtractor {
    async fn extract_facts(
        &self,
        _response_content: &str,
        _tool_calls: &[ToolCall],
    ) -> Result<Vec<String>, ProviderError> {
        Ok(Vec::new())
    }
}
