//! `OllamaEmbedding` — `EmbeddingProvider` against the Ollama single-prompt
//! `/api/embeddings` endpoint (POC parity with `smos/embeddings.py`).
//!
//! The endpoint accepts `{"model": ..., "prompt": "..."}` and returns
//! `{"embedding": [f32; dim]}`. HTTP-level failures are translated to
//! `Ok(None)` so the upstream `EnrichRequest` use case can apply its fail-open
//! policy; only request-body serialisation failures surface as `Err`
//! (those indicate a code bug, not a transient outage).

use std::sync::Arc;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use smos_application::errors::ProviderError;
use smos_application::ports::EmbeddingProvider;

use crate::config::EmbeddingConfig;
use crate::providers::ollama::ollama_client::build_client;

/// Ollama-backed embedding adapter (Jina v5 by default).
#[derive(Clone)]
pub struct OllamaEmbedding {
    client: Client,
    config: Arc<EmbeddingConfig>,
}

impl OllamaEmbedding {
    /// Build the adapter with a fresh pooled HTTP client sized to the config's
    /// timeout. Construction does NOT contact the server — the first request
    /// is the first network call.
    pub fn new(config: Arc<EmbeddingConfig>) -> Result<Self, ProviderError> {
        let client = build_client(config.timeout_seconds)?;
        Ok(Self { client, config })
    }

    fn embeddings_url(&self) -> String {
        format!("{}/api/embeddings", self.config.url.trim_end_matches('/'))
    }

    /// Read-only access to the configured dimensions. Exposed for tests that
    /// want to seed embeddings matching the adapter's vector index.
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }
}

#[derive(Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    embedding: Vec<f32>,
}

impl EmbeddingProvider for OllamaEmbedding {
    async fn embed(&self, text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
        if text.trim().is_empty() {
            // Avoid a needless round-trip on empty input; the upstream pipeline
            // treats short topics as "skip enrichment" anyway.
            return Ok(None);
        }
        let body = EmbeddingsRequest {
            model: &self.config.model,
            prompt: text,
        };
        let response = match self
            .client
            .post(self.embeddings_url())
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                if e.is_timeout() {
                    tracing::warn!(error = %e, "ollama embeddings timeout (fail-open)");
                } else {
                    tracing::warn!(error = %e, "ollama embeddings send failed (fail-open)");
                }
                return Ok(None);
            }
        };
        if !response.status().is_success() {
            tracing::warn!(
                status = response.status().as_u16(),
                "ollama embeddings non-2xx (fail-open)"
            );
            return Ok(None);
        }
        let parsed: EmbeddingsResponse = match response.json().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "ollama embeddings body decode failed (fail-open)");
                return Ok(None);
            }
        };
        if parsed.embedding.is_empty() {
            tracing::warn!("ollama returned empty embedding (fail-open)");
            return Ok(None);
        }
        Ok(Some(parsed.embedding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(url: &str) -> Arc<EmbeddingConfig> {
        Arc::new(EmbeddingConfig {
            url: url.into(),
            model: "m".into(),
            ..EmbeddingConfig::default()
        })
    }

    #[test]
    fn embeddings_url_strips_trailing_slash_and_appends_path() {
        let embed = OllamaEmbedding::new(cfg("http://ollama:11434/")).expect("build");
        assert_eq!(embed.embeddings_url(), "http://ollama:11434/api/embeddings");
    }

    #[test]
    fn embeddings_url_for_plain_base() {
        let embed = OllamaEmbedding::new(cfg("http://ollama:11434")).expect("build");
        assert_eq!(embed.embeddings_url(), "http://ollama:11434/api/embeddings");
    }

    #[test]
    fn dimensions_exposes_configured_value() {
        let embed = OllamaEmbedding::new(cfg("http://ollama:11434")).expect("build");
        assert_eq!(embed.dimensions(), EmbeddingConfig::default().dimensions);
    }
}
