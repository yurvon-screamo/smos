//! Concrete `EmbeddingProvider` and `RerankProvider` adapters.
//!
//! - [`ollama`] implements `EmbeddingProvider` + `LlmExtractor` against
//!   OpenAI-compatible `/v1/embeddings` and `/v1/chat/completions` endpoints
//!   (served by `llama-server`). The module name is historical — the wire
//!   protocol is OpenAI-compatible, not Ollama-native.
//! - [`llama_cpp`] implements `RerankProvider` against the llama.cpp
//!   `/v1/rerank` endpoint.
//!
//! ## Fail-open vs fail-closed layering
//!
//! The two adapters diverge deliberately:
//!
//! - **Embeddings are fail-open.** The embedding adapter translates HTTP-level
//!   failures into `Ok(None)`. The `EnrichRequest` use case treats that as
//!   "skip enrichment, forward the original messages" so a flaky embedder
//!   never blocks a chat request.
//! - **Reranker is fail-closed.** The llama.cpp adapter surfaces HTTP-level
//!   failures as `Err(ProviderError::…)`, and the use case propagates that
//!   as `Err(UseCaseError::Provider(_))` → HTTP 503. Silent vector-order-only
//!   ranking is not exposed to the client. See
//!   `smos-application/src/use_cases/enrich_request.rs` for the rationale.
//!
//! > **Note:** the use case is the source of truth for both policies. The
//! > embedder's `Ok(None)` conversion exists so transient network blips do
//! > not even reach the use case's `Err` arm; the reranker's `Err`
//! > conversion exists so the 503 body carries the real root cause
//! > ("reranker timeout after 60s" vs the generic "reranker returned empty
//! > results" the use case emits when the server itself responds with
//! > nothing).

pub mod llama_cpp;
pub mod noop;
pub mod ollama;

pub use llama_cpp::LlamaCppReranker;
pub use noop::NoopExtractor;
pub use ollama::{OllamaEmbedding, OllamaExtractor};
