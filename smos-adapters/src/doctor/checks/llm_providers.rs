//! LLM/embedding connectivity + required-models check + reranker probe.
//!
//! Two public entry points:
//! - [`check_llm_extractions`] / [`check_embeddings`] — `GET {url}/api/tags`,
//!   list models, match against the configured extraction + embedding
//!   expectations respectively.
//! - [`check_reranker`] — probe the llama.cpp reranker URL. The reranker is
//!   REQUIRED for production-quality enrichment; an unreachable endpoint
//!   downgrades the enrich pipeline to vector-order-only ranking (degraded
//!   mode), so the probe emits WARN — not FAIL, because SMOS still serves
//!   requests without it — with an explicit remediation hint.
//!
//! The match logic lives in [`super::super::models`]; this module owns the
//! HTTP IO and the row construction.

use std::time::Duration;

use reqwest::Client;

use super::super::models::{ExpectedModel, match_expected_models};
use super::super::types::CheckResult;
use crate::config::{EmbeddingConfig, LlmExtractionConfig, RerankerConfig};

/// Ollama `/api/tags` response shape — only the fields the doctor reads.
/// Extra fields returned by the server are silently ignored by serde.
#[derive(Debug, serde::Deserialize)]
struct TagsResponse {
    models: Vec<TagsModel>,
}

#[derive(Debug, serde::Deserialize)]
struct TagsModel {
    name: String,
}

/// Build the expected-model list from the SMOS extraction + embedding
/// configs. One row per role. Exposed for tests that exercise the doctor
/// helper without spinning up Ollama.
#[cfg(test)]
fn expected_models_from_config(
    extraction: &LlmExtractionConfig,
    embedding: &EmbeddingConfig,
) -> Vec<ExpectedModel> {
    vec![
        ExpectedModel::new("extraction model", &extraction.model),
        ExpectedModel::new("embedding model", &embedding.model),
    ]
}

/// Probe the LLM extraction endpoint and emit one row per expected model +
/// one connectivity row.
///
/// `timeout` bounds each HTTP request so a wedged backend that accepts the
/// TCP handshake but never responds surfaces as FAIL instead of hanging
/// the doctor.
pub async fn check_llm_extractions(
    client: &Client,
    extraction: &LlmExtractionConfig,
    timeout: Duration,
) -> Vec<CheckResult> {
    check_one_endpoint(
        client,
        &extraction.url,
        &extraction.model,
        "extraction",
        timeout,
    )
    .await
}

/// Probe the embedding endpoint and emit one row per expected model + one
/// connectivity row. Kept separate from [`check_llm_extractions`] because the
/// two sections may point at different hosts.
pub async fn check_embeddings(
    client: &Client,
    embedding: &EmbeddingConfig,
    timeout: Duration,
) -> Vec<CheckResult> {
    check_one_endpoint(
        client,
        &embedding.url,
        &embedding.model,
        "embedding",
        timeout,
    )
    .await
}

async fn check_one_endpoint(
    client: &Client,
    base_url: &str,
    model: &str,
    role: &'static str,
    timeout: Duration,
) -> Vec<CheckResult> {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let mut results = Vec::new();

    let response = client.get(&url).timeout(timeout).send().await;
    let body = match response {
        Ok(r) if r.status().is_success() => r.bytes().await.ok(),
        _ => None,
    };

    let Some(bytes) = body else {
        results.push(
            CheckResult::fail(
                format!("Ollama connectivity ({role})"),
                format!("url: {base_url}"),
            )
            .with_recommendation("start `ollama serve`"),
        );
        results.push(
            CheckResult::fail(
                format!("Required model ({role}): {model}"),
                "Ollama unreachable",
            )
            .with_recommendation(format!("ollama pull {model}")),
        );
        return results;
    };

    let parsed: Result<TagsResponse, _> = serde_json::from_slice(&bytes);
    let Ok(parsed) = parsed else {
        results.push(
            CheckResult::fail(
                format!("Ollama connectivity ({role})"),
                "response was not valid JSON",
            )
            .with_recommendation("check Ollama version (>=0.1.x)"),
        );
        return results;
    };

    let names: Vec<String> = parsed.models.into_iter().map(|m| m.name).collect();
    let count = names.len();
    results.push(CheckResult::pass(
        format!("Ollama connectivity ({role})"),
        format!("url: {base_url}\navailable models: {count}"),
    ));

    let role_label: &'static str = match role {
        "extraction" => "extraction model",
        "embedding" => "embedding model",
        other => other,
    };
    let expected = vec![ExpectedModel::new(role_label, model)];
    for (m, hit) in match_expected_models(&expected, &names) {
        let name = format!("Required model: {}", m.configured);
        if hit {
            results.push(CheckResult::pass(name, format!("role: {}", m.role)));
        } else {
            results.push(
                CheckResult::fail(name, "not pulled")
                    .with_recommendation(format!("ollama pull {}", m.configured)),
            );
        }
    }
    results
}

/// Doctor check name for the reranker probe. Shared between the live probe
/// ([`check_reranker`]) and the TLS-init-failure fallback
/// ([`super::http_client_unavailable_rows`]) so the operator sees a stable
/// row identifier across every code path that emits a reranker result.
pub(crate) const RERANKER_CHECK_NAME: &str =
    "Reranker (REQUIRED for production-quality enrichment)";

/// Recommendation emitted when the reranker is unreachable from the live
/// probe. The wording reinforces the REQUIRED-for-enrichment contract so
/// the operator understands the quality cost of leaving it down.
const RERANKER_UNREACHABLE_HINT: &str = "reranker REQUIRED for production-quality enrichment — start the \
     llama.cpp reranker server; without it the enrich pipeline runs \
     in degraded mode (vector-order-only ranking)";

/// Probe the reranker. WARN on any failure — the reranker is REQUIRED for
/// production-quality enrichment, but SMOS keeps serving requests without it
/// (degraded mode: vector-order-only ranking). A WARN rather than FAIL keeps
/// the doctor non-blocking for dev setups while making the production-quality
/// expectation loud.
///
/// `timeout` bounds the health probe so an unreachable reranker surfaces
/// as WARN instead of stalling the doctor.
pub async fn check_reranker(
    client: &Client,
    config: &RerankerConfig,
    timeout: Duration,
) -> CheckResult {
    let url = format!("{}/health", config.url.trim_end_matches('/'));
    match client.get(&url).timeout(timeout).send().await {
        Ok(r) if r.status().is_success() => CheckResult::pass(
            RERANKER_CHECK_NAME,
            format!("url: {}\nmodel: {}", config.url, config.model),
        ),
        Ok(r) => CheckResult::warn(
            RERANKER_CHECK_NAME,
            format!("url: {}\nHTTP {}", config.url, r.status()),
        )
        .with_recommendation(RERANKER_UNREACHABLE_HINT),
        Err(_) => CheckResult::warn(
            RERANKER_CHECK_NAME,
            format!("url: {}\nunreachable", config.url),
        )
        .with_recommendation(RERANKER_UNREACHABLE_HINT),
    }
}

/// Test helper: classify the first model row from an `/api/tags` JSON body.
/// Exposed so the unit tests can verify the parser shape without spinning
/// up Ollama.
#[cfg(test)]
pub(crate) fn parse_tags_for_test(body: &[u8]) -> Option<Vec<String>> {
    let parsed: TagsResponse = serde_json::from_slice(body).ok()?;
    Some(parsed.models.into_iter().map(|m| m.name).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extraction() -> LlmExtractionConfig {
        LlmExtractionConfig {
            url: "http://localhost:11434".into(),
            model: "qwen3.5:2b".into(),
            ..LlmExtractionConfig::default()
        }
    }

    fn embedding() -> EmbeddingConfig {
        EmbeddingConfig {
            url: "http://localhost:11434".into(),
            model: "hf.co/jinaai/jinaai-embeddings-v5:latest".into(),
            ..EmbeddingConfig::default()
        }
    }

    #[test]
    fn expected_models_from_config_lists_both_roles() {
        let expected = expected_models_from_config(&extraction(), &embedding());
        assert_eq!(expected.len(), 2);
        assert_eq!(expected[0].role, "extraction model");
        assert_eq!(expected[1].role, "embedding model");
    }

    #[test]
    fn parse_tags_for_test_handles_minimal_body() {
        let body = br#"{"models":[{"name":"granite4.1:3b"},{"name":"qwen3.5:2b"}]}"#;
        let names = parse_tags_for_test(body).expect("parsed");
        assert_eq!(
            names,
            vec!["granite4.1:3b".to_string(), "qwen3.5:2b".to_string()]
        );
    }

    #[test]
    fn parse_tags_for_test_returns_none_on_invalid_body() {
        assert!(parse_tags_for_test(b"not json").is_none());
        assert!(parse_tags_for_test(br#"{"no_models_key":[]}"#).is_none());
    }
}
