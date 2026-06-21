//! `LlmUpstream` port — single-call LLM proxy (slice 3+).
//!
//! The upstream abstracts the OpenAI-compatible HTTP endpoint. `complete`
//! returns either a fully-buffered JSON response (non-streaming callers) or a
//! byte stream (streaming callers). Slice-3's HTTP adapter implements this.

use crate::errors::UpstreamError;
use crate::types::{ChatRequest, ChatResponse};

/// OpenAI-compatible chat-completion boundary.
pub trait LlmUpstream {
    /// Submit `request` and return either a buffered JSON body or a byte
    /// stream, depending on `request.stream` (the OpenAI streaming flag).
    async fn complete(&self, request: ChatRequest) -> Result<ChatResponse, UpstreamError>;
}
