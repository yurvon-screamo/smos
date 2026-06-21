//! Ollama-backed adapters.
//!
//! - [`OllamaEmbedding`] implements `EmbeddingProvider` against the Ollama
//!   `/api/embeddings` (single-prompt) endpoint (Jina v5).
//! - [`OllamaExtractor`] implements `LlmExtractor` against `/api/chat`
//!   (Qwen3.5-2B), wired in Slice-5 for post-response fact extraction.

mod ollama_client;
mod ollama_embedding;
mod ollama_extractor;

pub use ollama_embedding::OllamaEmbedding;
pub use ollama_extractor::OllamaExtractor;
