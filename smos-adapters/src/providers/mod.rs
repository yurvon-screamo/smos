//! Concrete `EmbeddingProvider` and `RerankProvider` adapters.
//!
//! - [`ollama`] implements `EmbeddingProvider` against the Ollama
//!   `/api/embeddings` (single-prompt) endpoint.
//! - [`llama_cpp`] implements `RerankProvider` against the llama.cpp
//!   `/v1/rerank` endpoint.
//!
//! ## Fail-open layering
//!
//! Adapters translate HTTP-level failures into recoverable shapes
//! (`Ok(None)` for embeddings, `Ok(vec![])` for rerank). This is a SECONDARY
//! defense — the PRIMARY fail-open lives in the `EnrichRequest` use case
//! (`smos-application/src/use_cases/enrich_request.rs`), which guards both
//! `Ok(None)` / `Ok(vec![])` AND any `Err` a future adapter might emit. The
//! adapter-level conversion exists so transient network blips do not even
//! reach the use case's `Err` arm and so a buggy mock / upgraded server
//! cannot inject a hard failure into the request path.
//!
//! > **Note:** the use case MUST remain the source of truth for the fail-open
//! > policy. Removing the use-case guards and relying solely on the adapter
//! > behaviour would break the §12 contract — adapters are allowed to evolve
//! > (e.g. start returning `Err` for quota errors) without breaking the
//! > pipeline.

pub mod llama_cpp;
pub mod noop;
pub mod ollama;

pub use llama_cpp::LlamaCppReranker;
pub use noop::NoopExtractor;
pub use ollama::{OllamaEmbedding, OllamaExtractor};
