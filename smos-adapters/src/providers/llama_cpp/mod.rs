//! `llama_cpp` reranker provider module.
//!
//! Implements [`crate::providers::RerankProvider`] (re-exported here for
//! convenience) against the OpenAI-compatible `/v1/rerank` endpoint exposed by
//! `llama-server` when started with a reranker model.

mod llama_cpp_reranker;

pub use llama_cpp_reranker::LlamaCppReranker;
