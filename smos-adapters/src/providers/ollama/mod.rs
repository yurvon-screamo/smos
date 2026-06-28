//! OpenAI-compatible adapters backed by `llama-server`.
//!
//! - [`OllamaEmbedding`] implements `EmbeddingProvider` against the
//!   OpenAI-compatible `/v1/embeddings` endpoint (Jina v5 GGUF).
//! - [`OllamaExtractor`] implements `LlmExtractor` against
//!   `/v1/chat/completions` (Qwen3.5-2B-MTP), wired in Slice-5 for
//!   post-response fact extraction.
//!
//! The module path and the `Ollama*` struct names are historical — they
//! predate the switch to `llama-server`. Both are kept as a stable public
//! API surface; the wire protocol these adapters speak is OpenAI-compatible
//! (served by `llama-server`), not a vendor-native API.

mod ollama_client;
mod ollama_embedding;
mod ollama_extractor;

pub use ollama_embedding::OllamaEmbedding;
pub use ollama_extractor::OllamaExtractor;
