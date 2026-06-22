//! `LlmUpstream` port — single-call LLM proxy (slice 3+).
//!
//! The upstream abstracts the OpenAI-compatible HTTP endpoint. `complete`
//! returns either a fully-buffered JSON response (non-streaming callers) or a
//! byte stream (streaming callers). Slice-3's HTTP adapter implements this.
//!
//! # Provider routing
//!
//! `complete` takes a `provider_name` argument so the same trait surface can
//! back the multi-provider routing introduced by the persons system: each
//! chat-completion request resolves a `[persons.X]` entry, which names a
//! `[[providers]]` entry, and the adapter uses the name to look up the
//! concrete URL + auth header. Single-provider stubs ignore the argument.

use crate::errors::UpstreamError;
use crate::types::{ChatRequest, ChatResponse};

/// OpenAI-compatible chat-completion boundary.
pub trait LlmUpstream {
    /// Submit `request` to the provider identified by `provider_name` and
    /// return either a buffered JSON body or a byte stream, depending on
    /// `request.stream` (the OpenAI streaming flag).
    ///
    /// Single-provider implementations ignore `provider_name` and forward to
    /// the only configured endpoint; multi-provider implementations look the
    /// name up in their internal map. An unknown name surfaces as
    /// [`UpstreamError::ConnectFailed`] so the HTTP layer maps it to 502.
    async fn complete(
        &self,
        provider_name: &str,
        request: ChatRequest,
    ) -> Result<ChatResponse, UpstreamError>;
}
