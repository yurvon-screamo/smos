//! LLM/embedding connectivity + reranker probes for the doctor.
//!
//! Three public entry points, each emitting one [`CheckResult`] row:
//! - [`check_llm_extractions`] — `GET {extraction_url}/health`.
//! - [`check_embeddings`] — `GET {embedding_url}/health`.
//! - [`check_reranker`] — `GET {reranker_url}/health`.
//!
//! The probes answer "is something listening on the configured port and
//! answering HTTP?" — they do NOT inspect the model loaded on the server
//! (llama.cpp exposes no standard `/v1/models` shape we can match against
//! the configured `model` id, and an unloaded-model failure surfaces fast
//! enough on the first real request). The reranker is a hard dependency:
//! an unreachable endpoint makes every chat-completion request fail with
//! HTTP 503, so its probe emits FAIL — not WARN — so an operator who runs
//! `smos doctor` sees the dependency before the first request fails in
//! production.

use std::time::Duration;

use reqwest::Client;

use super::super::types::CheckResult;
use crate::config::{EmbeddingConfig, LlmExtractionConfig, RerankerConfig};

/// Build the `/health` URL for `base_url` (trailing-slash safe).
fn health_url(base_url: &str) -> String {
    format!("{}/health", base_url.trim_end_matches('/'))
}

/// Probe the LLM extraction server (`/health`). Emits one PASS / FAIL row
/// carrying the configured URL so the operator can confirm which `[llm_extraction]`
/// entry was actually validated.
///
/// `timeout` bounds the request so a wedged backend that accepts the TCP
/// handshake but never responds surfaces as FAIL instead of hanging the
/// doctor.
pub async fn check_llm_extractions(
    client: &Client,
    extraction: &LlmExtractionConfig,
    timeout: Duration,
) -> Vec<CheckResult> {
    vec![probe_role(client, &extraction.url, "extraction", timeout).await]
}

/// Probe the embedding server (`/health`). Kept separate from
/// [`check_llm_extractions`] because the two sections may point at different
/// hosts.
pub async fn check_embeddings(
    client: &Client,
    embedding: &EmbeddingConfig,
    timeout: Duration,
) -> Vec<CheckResult> {
    vec![probe_role(client, &embedding.url, "embedding", timeout).await]
}

/// One role probe. Returns a single row whose name embeds the role label so
/// the doctor output stays readable when multiple roles are checked.
async fn probe_role(
    client: &Client,
    base_url: &str,
    role: &'static str,
    timeout: Duration,
) -> CheckResult {
    let url = health_url(base_url);
    match client.get(&url).timeout(timeout).send().await {
        Ok(response) => {
            let status = response.status();
            if status.is_success() {
                CheckResult::pass(
                    format!("llama-server connectivity ({role})"),
                    format!("url: {base_url}"),
                )
            } else {
                CheckResult::fail(
                    format!("llama-server connectivity ({role})"),
                    format!("url: {base_url}\nHTTP {}", status),
                )
                .with_recommendation(format!(
                    "start `llama-server` for the {role} role on {base_url}"
                ))
            }
        }
        Err(_) => CheckResult::fail(
            format!("llama-server connectivity ({role})"),
            format!("url: {base_url}\nunreachable"),
        )
        .with_recommendation(format!(
            "start `llama-server` for the {role} role on {base_url}, or enable \
             [llama_cpp] auto_launch = true in config.toml"
        )),
    }
}

/// Doctor check name for the reranker probe. Shared between the live probe
/// ([`check_reranker`]) and the TLS-init-failure fallback
/// ([`super::http_client_unavailable_rows`]) so the operator sees a stable
/// row identifier across every code path that emits a reranker result.
pub(crate) const RERANKER_CHECK_NAME: &str = "Reranker";

/// Recommendation emitted when the reranker is unreachable from the live
/// probe. Reminds the operator that the reranker is a hard dependency: every
/// chat-completion request returns HTTP 503 until the reranker is back
/// online.
const RERANKER_UNREACHABLE_HINT: &str = "start the llama.cpp reranker server; \
     every chat-completion request fails with HTTP 503 while it is down";

/// Probe the reranker. FAIL on any failure — the reranker is a hard
/// dependency: a provider error or an empty rerank result makes
/// `EnrichRequest::execute` return `Err(UseCaseError::Provider(_))`, and
/// the HTTP handler maps that to 503 on every chat-completion request. A
/// FAIL (rather than WARN) makes the dependency loud in `smos doctor` so
/// an operator catches a down reranker before the first user-facing 503.
///
/// `timeout` bounds the health probe so an unreachable reranker surfaces
/// as FAIL instead of stalling the doctor.
pub async fn check_reranker(
    client: &Client,
    config: &RerankerConfig,
    timeout: Duration,
) -> CheckResult {
    let url = health_url(&config.url);
    match client.get(&url).timeout(timeout).send().await {
        Ok(r) if r.status().is_success() => CheckResult::pass(
            RERANKER_CHECK_NAME,
            format!("url: {}\nmodel: {}", config.url, config.model),
        ),
        Ok(r) => CheckResult::fail(
            RERANKER_CHECK_NAME,
            format!("url: {}\nHTTP {}", config.url, r.status()),
        )
        .with_recommendation(RERANKER_UNREACHABLE_HINT),
        Err(_) => CheckResult::fail(
            RERANKER_CHECK_NAME,
            format!("url: {}\nunreachable", config.url),
        )
        .with_recommendation(RERANKER_UNREACHABLE_HINT),
    }
}
