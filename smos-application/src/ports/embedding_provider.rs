//! `EmbeddingProvider` port — text → vector.
//!
//! `embed_batch` has a default implementation that loops `embed` per call:
//! providers with native batch endpoints (Ollama `/api/embed`, OpenAI
//! `/v1/embeddings` with `input: []`) override it for fewer round-trips.

use crate::errors::ProviderError;

/// Embedding model boundary (Jina v5, Ollama, OpenAI, …).
pub trait EmbeddingProvider {
    /// Embed a single text. `None` is returned when the provider cannot
    /// produce an embedding (e.g. input is empty after normalisation).
    async fn embed(&self, text: &str) -> Result<Option<Vec<f32>>, ProviderError>;

    /// Embed many texts. Default loops `embed`; override for batch endpoints.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Option<Vec<f32>>>, ProviderError> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            out.push(self.embed(text).await?);
        }
        Ok(out)
    }
}
