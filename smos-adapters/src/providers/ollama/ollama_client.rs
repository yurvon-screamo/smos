//! Shared HTTP helpers for Ollama-style requests.
//!
//! One `reqwest::Client` per adapter is the right granularity: the client pools
//! connections and amortises TLS handshakes across requests. The builder is
//! configured with the per-section timeout (`EmbeddingConfig::timeout_seconds`
//! for the embedder, `LlmExtractionConfig::timeout_seconds` for the extractor).

use std::time::Duration;

use reqwest::Client;
use smos_application::errors::ProviderError;

/// Build a pooled `reqwest::Client` configured with the supplied timeout.
///
/// Free function so the embedding adapter and the extraction adapter can
/// share the same builder shape without forcing a shared owner.
pub fn build_client(timeout_seconds: u64) -> Result<Client, ProviderError> {
    Client::builder()
        .timeout(Duration::from_secs(timeout_seconds))
        .build()
        .map_err(|e| ProviderError::Unavailable(format!("reqwest client build failed: {e}")))
}
