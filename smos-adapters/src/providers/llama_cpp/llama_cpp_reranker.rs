//! `LlamaCppReranker` — `RerankProvider` against the llama.cpp `/v1/rerank`
//! endpoint.
//!
//! Request body mirrors the OpenAI-style rerank shape used by the POC:
//! `{"model": ..., "query": ..., "documents": [...], "top_n": k}`. The
//! response carries a `results` array; each entry's `index` references the
//! original `documents` slice so callers can map back to their source facts.
//!
//! HTTP-level failures are translated to `Ok(vec![])` (empty result) so the
//! upstream `EnrichRequest` use case can apply its top-N-survivors fallback;
//! only serialisation failures surface as `Err` (a code bug, not a transient
//! outage).

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use smos_application::errors::ProviderError;
use smos_application::ports::RerankProvider;
use smos_application::types::RerankResult;

use crate::config::RerankerConfig;

/// llama.cpp-backed reranker adapter (Qwen3-Reranker by default).
#[derive(Clone)]
pub struct LlamaCppReranker {
    client: Client,
    config: Arc<RerankerConfig>,
}

impl LlamaCppReranker {
    /// Build the adapter with a fresh pooled HTTP client sized to the config's
    /// timeout. Construction does NOT contact the server.
    pub fn new(config: Arc<RerankerConfig>) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| ProviderError::Unavailable(format!("reqwest client build failed: {e}")))?;
        Ok(Self { client, config })
    }

    fn rerank_url(&self) -> String {
        format!("{}/v1/rerank", self.config.url.trim_end_matches('/'))
    }
}

#[derive(Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [String],
    top_n: usize,
}

#[derive(Deserialize)]
struct RerankResponse {
    #[serde(default)]
    results: Vec<RerankResponseItem>,
}

#[derive(Deserialize)]
struct RerankResponseItem {
    index: usize,
    /// llama.cpp exposes the cross-encoder logit as `relevance_score` on
    /// recent builds; older builds used `score`. We accept either so the
    /// adapter does not break on a server upgrade.
    #[serde(default)]
    relevance_score: Option<f32>,
    #[serde(default)]
    score: Option<f32>,
}

impl RerankResponseItem {
    fn score_or_zero(&self) -> f32 {
        self.relevance_score.or(self.score).unwrap_or(0.0)
    }
}

impl RerankProvider for LlamaCppReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<RerankResult>, ProviderError> {
        if documents.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let body = RerankRequest {
            model: &self.config.model,
            query,
            documents,
            top_n: top_k,
        };
        let response = match self.client.post(self.rerank_url()).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                if e.is_timeout() {
                    tracing::warn!(error = %e, "reranker timeout (fail-open: empty result)");
                } else {
                    tracing::warn!(error = %e, "reranker send failed (fail-open: empty result)");
                }
                return Ok(Vec::new());
            }
        };
        if !response.status().is_success() {
            tracing::warn!(
                status = response.status().as_u16(),
                "reranker non-2xx (fail-open: empty result)"
            );
            return Ok(Vec::new());
        }
        let parsed: RerankResponse = match response.json().await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "reranker body decode failed (fail-open: empty result)");
                return Ok(Vec::new());
            }
        };
        Ok(materialise_results(parsed, documents, top_k))
    }
}

/// Convert the raw rerank response into `RerankResult`s ordered by descending
/// score, capped at `top_k`. Out-of-range indices (server bug) are dropped
/// rather than panicking, and duplicate indices are de-duplicated (kept first
/// occurrence) so a malformed server response cannot inject the same document
/// twice.
fn materialise_results(
    parsed: RerankResponse,
    documents: &[String],
    top_k: usize,
) -> Vec<RerankResult> {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut items: Vec<RerankResult> = parsed
        .results
        .into_iter()
        .filter_map(|r| {
            let doc = documents.get(r.index)?;
            if !seen.insert(r.index) {
                return None;
            }
            Some(RerankResult {
                index: r.index,
                score: r.score_or_zero(),
                document: doc.clone(),
            })
        })
        .collect();
    // Sort by score descending (stable so equal scores preserve server order).
    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items.truncate(top_k);
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(url: &str) -> Arc<RerankerConfig> {
        Arc::new(RerankerConfig {
            url: url.into(),
            model: "rr".into(),
            timeout_seconds: 2,
        })
    }

    #[test]
    fn rerank_url_strips_trailing_slash_and_appends_path() {
        let r = LlamaCppReranker::new(cfg("http://rr:8181/")).expect("build");
        assert_eq!(r.rerank_url(), "http://rr:8181/v1/rerank");
    }

    #[test]
    fn rerank_url_for_plain_base() {
        let r = LlamaCppReranker::new(cfg("http://rr:8181")).expect("build");
        assert_eq!(r.rerank_url(), "http://rr:8181/v1/rerank");
    }

    #[test]
    fn materialise_sorts_by_score_descending() {
        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let parsed = RerankResponse {
            results: vec![
                RerankResponseItem {
                    index: 0,
                    relevance_score: Some(0.1),
                    score: None,
                },
                RerankResponseItem {
                    index: 1,
                    relevance_score: Some(0.9),
                    score: None,
                },
                RerankResponseItem {
                    index: 2,
                    relevance_score: Some(0.5),
                    score: None,
                },
            ],
        };
        let out = materialise_results(parsed, &docs, 3);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].index, 1);
        assert_eq!(out[1].index, 2);
        assert_eq!(out[2].index, 0);
    }

    #[test]
    fn materialise_truncates_to_top_k() {
        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let parsed = RerankResponse {
            results: vec![
                RerankResponseItem {
                    index: 0,
                    relevance_score: Some(0.1),
                    score: None,
                },
                RerankResponseItem {
                    index: 1,
                    relevance_score: Some(0.9),
                    score: None,
                },
            ],
        };
        let out = materialise_results(parsed, &docs, 1);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].index, 1);
    }

    #[test]
    fn materialise_drops_out_of_range_indices() {
        let docs = vec!["only".to_string()];
        let parsed = RerankResponse {
            results: vec![
                RerankResponseItem {
                    index: 0,
                    relevance_score: Some(0.5),
                    score: None,
                },
                RerankResponseItem {
                    index: 7,
                    relevance_score: Some(0.99),
                    score: None,
                },
            ],
        };
        let out = materialise_results(parsed, &docs, 5);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].index, 0);
    }

    #[test]
    fn materialise_falls_back_to_score_when_relevance_score_absent() {
        let docs = vec!["a".to_string()];
        let parsed = RerankResponse {
            results: vec![RerankResponseItem {
                index: 0,
                relevance_score: None,
                score: Some(0.42),
            }],
        };
        let out = materialise_results(parsed, &docs, 5);
        assert_eq!(out.len(), 1);
        assert!((out[0].score - 0.42).abs() < 1e-6);
    }

    #[test]
    fn materialise_deduplicates_repeated_indices() {
        let docs = vec!["a".to_string(), "b".to_string()];
        let parsed = RerankResponse {
            results: vec![
                RerankResponseItem {
                    index: 1,
                    relevance_score: Some(0.9),
                    score: None,
                },
                RerankResponseItem {
                    index: 1, // duplicate — must be dropped
                    relevance_score: Some(0.99),
                    score: None,
                },
                RerankResponseItem {
                    index: 0,
                    relevance_score: Some(0.5),
                    score: None,
                },
            ],
        };
        let out = materialise_results(parsed, &docs, 5);
        assert_eq!(out.len(), 2, "duplicate index should be collapsed");
        assert_eq!(out[0].index, 1, "highest score first");
        assert_eq!(out[1].index, 0);
    }
}
