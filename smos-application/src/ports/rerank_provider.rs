//! `RerankProvider` port — cross-encoder re-ordering of candidate documents.

use crate::errors::ProviderError;
use crate::types::RerankResult;

/// Reranker model boundary (Jina v2 reranker, Cohere, BGE-reranker, …).
pub trait RerankProvider {
    /// Re-score `documents` against `query` and return the top-`top_k` hits
    /// ordered by descending relevance score. Each hit preserves the original
    /// document index so the caller can map back to its source fact.
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<RerankResult>, ProviderError>;
}
