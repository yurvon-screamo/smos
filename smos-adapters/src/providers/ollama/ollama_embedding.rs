//! `OllamaEmbedding` — `EmbeddingProvider` against an OpenAI-compatible
//! `/v1/embeddings` endpoint backed by `llama-server` (Jina v5 by default).
//!
//! The endpoint accepts `{"model": ..., "input": "..."}` and returns
//! `{"data": [{"embedding": [f32; dim]}]}`. HTTP-level failures are
//! translated to `Ok(None)` so the upstream `EnrichRequest` use case can apply
//! its fail-open policy; only request-body serialisation failures surface as
//! `Err` (those indicate a code bug, not a transient outage).

use std::sync::Arc;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use smos_application::errors::ProviderError;
use smos_application::ports::EmbeddingProvider;

use crate::config::EmbeddingConfig;
use crate::providers::ollama::ollama_client::build_client;

/// OpenAI-compatible embedding adapter backed by `llama-server`
/// (Jina v5 by default).
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
        format!("{}/v1/embeddings", self.config.url.trim_end_matches('/'))
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
    input: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingsData>,
}

#[derive(Deserialize)]
struct EmbeddingsData {
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
            input: text,
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
                    tracing::warn!(error = %e, "embeddings timeout (fail-open)");
                } else {
                    tracing::warn!(error = %e, "embeddings send failed (fail-open)");
                }
                return Ok(None);
            }
        };
        if !response.status().is_success() {
            tracing::warn!(
                status = response.status().as_u16(),
                "embeddings non-2xx (fail-open)"
            );
            return Ok(None);
        }
        let parsed: EmbeddingsResponse = match response.json().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "embeddings body decode failed (fail-open)");
                return Ok(None);
            }
        };
        let Some(first) = parsed.data.into_iter().next() else {
            tracing::warn!("embeddings response had no data items (fail-open)");
            return Ok(None);
        };
        if first.embedding.is_empty() {
            tracing::warn!("llama-server returned empty embedding (fail-open)");
            return Ok(None);
        }
        Ok(Some(first.embedding))
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
        let embed = OllamaEmbedding::new(cfg("http://llama:28081/")).expect("build");
        assert_eq!(embed.embeddings_url(), "http://llama:28081/v1/embeddings");
    }

    #[test]
    fn embeddings_url_for_plain_base() {
        let embed = OllamaEmbedding::new(cfg("http://llama:28081")).expect("build");
        assert_eq!(embed.embeddings_url(), "http://llama:28081/v1/embeddings");
    }

    #[test]
    fn dimensions_exposes_configured_value() {
        let embed = OllamaEmbedding::new(cfg("http://llama:28081")).expect("build");
        assert_eq!(embed.dimensions(), EmbeddingConfig::default().dimensions);
    }
}
