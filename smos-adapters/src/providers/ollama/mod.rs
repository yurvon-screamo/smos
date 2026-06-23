//! OpenAI-compatible adapters backed by `llama-server`.
//!
//! - [`OllamaEmbedding`] implements `EmbeddingProvider` against the
//!   OpenAI-compatible `/v1/embeddings` endpoint (Jina v5 GGUF).
//! - [`OllamaExtractor`] implements `LlmExtractor` against
//!   `/v1/chat/completions` (Nemotron-3-Nano-4B), wired in Slice-5 for
//!   post-response fact extraction.
//!
//! The module path is historical (`providers::ollama`) — the struct names
//! `OllamaExtractor` / `OllamaEmbedding` are kept as stable public API
//! surface while the wire protocol they speak is now OpenAI-compatible
//! (served by `llama-server`).

mod ollama_client;
mod ollama_embedding;
mod ollama_extractor;

pub use ollama_embedding::OllamaEmbedding;
pub use ollama_extractor::OllamaExtractor;
