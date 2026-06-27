//! `RetrieveFacts` — read-only retrieval use case (BEAM benchmark contract).
//!
//! Returns the reranked accepted facts for a query WITHOUT any of the
//! state-mutating steps the live [`EnrichRequest`] pipeline runs: no session
//! dedup, no heat boost on write, no message injection. It is the retrieval
//! surface the BEAM long-term-memory harness calls via `smos search`.
//!
//! # Fail-closed (differs from `EnrichRequest`)
//!
//! [`EnrichRequest`] is fail-open for every port except the reranker because
//! it sits on the hot proxy path — a flaky memory subsystem must never break
//! the user's chat. This use case is NOT on the hot path: it is a benchmark /
//! operator tool whose entire purpose is to report what SMOS would retrieve.
//! Silent empty output would mislead the benchmark into scoring SMOS as
//! "remembers nothing". Every provider failure (embed None/Err, vector-search
//! Err, reranker Err/empty) therefore propagates as [`UseCaseError`]; only the
//! "nothing matched" outcomes (short query, zero vector hits, zero
//! post-filter survivors) return `Ok(vec![])`.
//!
//! # Ranking parity
//!
//! The rerank + projection path delegates to
//! [`crate::helpers::retrieval_pipeline`], the SAME helpers
//! [`EnrichRequest`] uses, so the ranking is identical to the live pipeline's
//! pre-dedup survivors by construction — pinned by `ranking_matches_enrich`.

use std::collections::HashMap;

use smos_domain::config::{HeatConfig, RetrievalConfig};
use smos_domain::{FactId, MemoryKey};

use crate::errors::{ProviderError, UseCaseError};
use crate::helpers::retrieval_pipeline;
use crate::helpers::retrieval_planner;
use crate::ports::{Clock, EmbeddingProvider, FactRepository, RerankProvider};
use crate::types::SearchHit;

/// One retrieval result: the full vector-search DTO plus the cross-encoder
/// relevance score that ordered it. `score` is the raw `RerankResult.score`
/// (higher = more relevant); it is NOT the cosine `distance` (lower = more
/// similar) carried by [`SearchHit::metadata`], and the two MUST NOT be
/// conflated.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredSearchHit {
    pub hit: SearchHit,
    pub score: f32,
}

/// Borrow-style bundle of every dependency the read-only retrieval pipeline
/// needs (ADR-0001 record-struct constructor convention — no positional
/// `new`). Deliberately omits `SessionRepository`: this use case performs no
/// session dedup so reranked facts are NEVER hidden by a prior injection.
pub struct RetrieveFacts<'a, FR, EP, RP, C> {
    pub facts: &'a FR,
    pub embedder: &'a EP,
    pub reranker: &'a RP,
    pub clock: &'a C,
    pub retrieval_cfg: &'a RetrievalConfig,
    pub heat_cfg: &'a HeatConfig,
}

impl<'a, FR, EP, RP, C> RetrieveFacts<'a, FR, EP, RP, C>
where
    FR: FactRepository,
    EP: EmbeddingProvider,
    RP: RerankProvider,
    C: Clock,
{
    /// Retrieve and rerank accepted facts for `query` under `memory_key`.
    ///
    /// `top_k_override` (when set) replaces `retrieval_cfg.top_k_final` as the
    /// rerank depth; `None` keeps the configured default.
    ///
    /// Returns the reranked hits in descending relevance order, each paired
    /// with its cross-encoder score. See the module docs for the fail-closed
    /// contract.
    pub async fn execute(
        &self,
        query: &str,
        memory_key: &MemoryKey,
        top_k_override: Option<usize>,
    ) -> Result<Vec<ScoredSearchHit>, UseCaseError> {
        // Short / empty query — "nothing to search for", not a failure.
        if query.trim().chars().count() < self.retrieval_cfg.min_topic_chars {
            return Ok(Vec::new());
        }

        let embedding = self.embed_query(query).await?;
        let hits = self
            .facts
            .search_similar(embedding, memory_key, self.retrieval_cfg.top_k_initial)
            .await?;
        if hits.is_empty() {
            return Ok(Vec::new());
        }

        // Keep the original DTOs (with created_at / conflicts_with / distance)
        // so they can be recovered after the RetrievalHit projection, which
        // intentionally drops those fields.
        let lookup = build_lookup(&hits);
        let survivors = self.prefilter_survivors(hits);
        if survivors.is_empty() {
            return Ok(Vec::new());
        }

        let top_k = top_k_override.unwrap_or(self.retrieval_cfg.top_k_final);
        let ranked =
            retrieval_pipeline::rerank_hits(query, &survivors, self.reranker, top_k).await?;
        if ranked.is_empty() {
            return Err(reranker_unusable());
        }

        Ok(ranked
            .into_iter()
            .filter_map(|r| {
                lookup.get(&r.hit.id).map(|hit| ScoredSearchHit {
                    hit: hit.clone(),
                    score: r.score,
                })
            })
            .collect())
    }

    /// Embed the query. Fail-CLOSED: `None` and `Err` both surface as
    /// [`UseCaseError::Provider`] (see module docs).
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, UseCaseError> {
        match self.embedder.embed(query).await {
            Ok(Some(v)) => Ok(v),
            Ok(None) => Err(UseCaseError::Provider(ProviderError::InvalidResponse(
                "embedder returned None for the query".to_string(),
            ))),
            Err(e) => Err(UseCaseError::Provider(e)),
        }
    }

    /// Project to `RetrievalHit` and apply the hard pre-filters + heat
    /// post-filter. Pure delegation so the survivor set matches
    /// [`EnrichRequest`]'s step-5 output verbatim.
    fn prefilter_survivors(&self, hits: Vec<SearchHit>) -> Vec<retrieval_planner::RetrievalHit> {
        let now = self.clock.now();
        let retrieval_hits: Vec<retrieval_planner::RetrievalHit> = hits
            .into_iter()
            .filter_map(retrieval_pipeline::hit_to_retrieval)
            .collect();
        retrieval_planner::prefilter_and_heat(
            &retrieval_hits,
            self.retrieval_cfg,
            self.heat_cfg,
            now,
        )
    }
}

/// Index the vector-search hits by FactId so the post-rerank mapping back to
/// the full DTO is O(1) and preserves the SearchHit-only fields the
/// `RetrievalHit` projection drops.
fn build_lookup(hits: &[SearchHit]) -> HashMap<FactId, SearchHit> {
    hits.iter().map(|h| (h.id.clone(), h.clone())).collect()
}

fn reranker_unusable() -> UseCaseError {
    UseCaseError::Provider(ProviderError::InvalidResponse(
        "reranker returned no usable results".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{ConstantEmbedder, FixedClock, InMemoryFacts, InMemorySessions};
    use crate::types::{RerankResult, SearchHitMetadata};
    use crate::use_cases::EnrichRequest;
    use smos_domain::config::{HeatConfig, RetrievalConfig};
    use smos_domain::{FactStatus, Heat, Timestamp};

    // Minimal local doubles for the embedder failure modes the testkit does
    // not cover (ConstantEmbedder always succeeds). Parity-shaped with the
    // local doubles in `enrich_request::execute_tests`.

    struct NoneEmbedder;
    impl crate::ports::EmbeddingProvider for NoneEmbedder {
        async fn embed(&self, _text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
            Ok(None)
        }
    }

    struct ErrorEmbedder;
    impl crate::ports::EmbeddingProvider for ErrorEmbedder {
        async fn embed(&self, _text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
            Err(ProviderError::Unavailable("embedder down".into()))
        }
    }

    fn now_ts() -> Timestamp {
        Timestamp::from_unix_secs(1_700_000_000).unwrap()
    }

    fn key() -> MemoryKey {
        MemoryKey::from_raw("origa").unwrap()
    }

    fn sid() -> smos_domain::SessionId {
        smos_domain::SessionId::from_raw("sess_0123456789ab").unwrap()
    }

    fn rcfg() -> RetrievalConfig {
        RetrievalConfig::default()
    }

    fn hcfg() -> HeatConfig {
        HeatConfig::default()
    }

    /// A SearchHit that clears the pre-filter + heat post-filter, with a
    /// content-derived FactId so distinct documents map to distinct ids.
    fn survivable_hit(document: &str, mk: &MemoryKey) -> SearchHit {
        SearchHit {
            id: FactId::from_content(document),
            document: document.to_string(),
            memory_key: mk.clone(),
            metadata: SearchHitMetadata {
                status: "accepted".into(),
                confidence: 0.85,
                valid_until: None,
                heat_base: 1.0,
                last_access_at: 1_700_000_000.0,
                distance: Some(0.1),
                created_at: Some("2025-06-18T12:00:00Z".into()),
                conflicts_with: vec!["fact_c0ffeec0ffeec0f".into()],
            },
        }
    }

    /// Deterministic, document-keyed reranker: scores `gamma` > `alpha` >
    /// `beta` and honours `top_k`, so the ranking is stable regardless of the
    /// survivor iteration order. Shared by the parity test and the
    /// truncation test.
    fn deterministic_reranker() -> crate::testkit::ScriptedReranker {
        crate::testkit::ScriptedReranker::matching(|_q, docs, top_k| {
            let mut scored: Vec<(usize, String, f32)> = docs
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let s = if d.contains("gamma") {
                        0.95
                    } else if d.contains("alpha") {
                        0.80
                    } else {
                        0.50
                    };
                    (i, d.clone(), s)
                })
                .collect();
            scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            Ok(scored
                .into_iter()
                .take(top_k)
                .map(|(i, d, s)| RerankResult {
                    index: i,
                    score: s,
                    document: d,
                })
                .collect())
        })
    }

    // -----------------------------------------------------------------------
    // Short / empty query → empty (NOT an error)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn short_query_returns_empty() {
        let facts = InMemoryFacts::default();
        let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker = deterministic_reranker();
        let uc = RetrieveFacts {
            facts: &facts,
            embedder: &embedder,
            reranker: &reranker,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        // "ok" is 2 chars < min_topic_chars (3).
        let out = uc.execute("ok", &key(), None).await.expect("ok");
        assert!(out.is_empty(), "short query returns an empty array");
    }

    #[tokio::test]
    async fn empty_query_returns_empty() {
        let facts = InMemoryFacts::default();
        let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker = deterministic_reranker();
        let uc = RetrieveFacts {
            facts: &facts,
            embedder: &embedder,
            reranker: &reranker,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let out = uc.execute("   ", &key(), None).await.expect("ok");
        assert!(
            out.is_empty(),
            "whitespace-only query returns an empty array"
        );
    }

    // -----------------------------------------------------------------------
    // Provider failures → fail-closed (Err)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn embed_none_fail_closes() {
        let facts = InMemoryFacts::default();
        let embedder = NoneEmbedder;
        let reranker = deterministic_reranker();
        let uc = RetrieveFacts {
            facts: &facts,
            embedder: &embedder,
            reranker: &reranker,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let result = uc.execute("explain rust ownership", &key(), None).await;
        assert!(
            matches!(result, Err(UseCaseError::Provider(_))),
            "embedder None must fail-closed (search is not fail-open)"
        );
    }

    #[tokio::test]
    async fn embed_err_fail_closes() {
        let facts = InMemoryFacts::default();
        let embedder = ErrorEmbedder;
        let reranker = deterministic_reranker();
        let uc = RetrieveFacts {
            facts: &facts,
            embedder: &embedder,
            reranker: &reranker,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let result = uc.execute("explain rust ownership", &key(), None).await;
        assert!(
            matches!(result, Err(UseCaseError::Provider(_))),
            "embedder Err must fail-closed"
        );
    }

    #[tokio::test]
    async fn reranker_empty_fail_closes() {
        let mk = key();
        let facts = InMemoryFacts::default();
        facts.script_search_hits(vec![survivable_hit("alpha fact", &mk)]);
        let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        // Empty FIFO script → first call returns Ok(vec![]), the legitimate
        // "provider found nothing" shape, which the use case treats as
        // fail-closed (parity with EnrichRequest step 8).
        let reranker = crate::testkit::ScriptedReranker::new(Vec::new());
        let uc = RetrieveFacts {
            facts: &facts,
            embedder: &embedder,
            reranker: &reranker,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let result = uc.execute("explain rust ownership", &mk, None).await;
        assert!(
            matches!(result, Err(UseCaseError::Provider(_))),
            "empty rerank result must fail-closed"
        );
    }

    // -----------------------------------------------------------------------
    // No matches → Ok(empty) (not an error)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn no_vector_hits_returns_empty() {
        let facts = InMemoryFacts::default();
        let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker = deterministic_reranker();
        let uc = RetrieveFacts {
            facts: &facts,
            embedder: &embedder,
            reranker: &reranker,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let out = uc
            .execute("explain rust ownership", &key(), None)
            .await
            .expect("ok");
        assert!(out.is_empty(), "no vector hits → empty array, not error");
    }

    // -----------------------------------------------------------------------
    // Truncation + field propagation
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn rerank_truncates_to_top_k_override() {
        let mk = key();
        let facts = InMemoryFacts::default();
        facts.script_search_hits(vec![
            survivable_hit("alpha fact", &mk),
            survivable_hit("beta fact", &mk),
            survivable_hit("gamma fact", &mk),
        ]);
        let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker = deterministic_reranker();
        let uc = RetrieveFacts {
            facts: &facts,
            embedder: &embedder,
            reranker: &reranker,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let out = uc
            .execute("explain rust ownership", &mk, Some(2))
            .await
            .expect("ok");
        assert_eq!(out.len(), 2, "top_k_override caps the result count");
        // Deterministic order: gamma (0.95) > alpha (0.80).
        assert!(out[0].hit.document.contains("gamma"));
        assert!(out[1].hit.document.contains("alpha"));
        // Scores are the reranker's, descending.
        assert!(out[0].score > out[1].score);
    }

    #[tokio::test]
    async fn created_at_and_conflicts_with_propagate() {
        let mk = key();
        let facts = InMemoryFacts::default();
        facts.script_search_hits(vec![survivable_hit("alpha fact", &mk)]);
        let embedder = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker = deterministic_reranker();
        let uc = RetrieveFacts {
            facts: &facts,
            embedder: &embedder,
            reranker: &reranker,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let out = uc
            .execute("explain rust ownership", &mk, None)
            .await
            .expect("ok");
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].hit.metadata.created_at.as_deref(),
            Some("2025-06-18T12:00:00Z"),
            "created_at survives the RetrievalHit round-trip via the lookup"
        );
        assert_eq!(
            out[0].hit.metadata.conflicts_with,
            vec!["fact_c0ffeec0ffeec0f"],
            "conflicts_with survives the RetrievalHit round-trip via the lookup"
        );
    }

    // -----------------------------------------------------------------------
    // PARITY: ranking matches EnrichRequest's pre-dedup survivors
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn ranking_matches_enrich_request() {
        let mk = key();
        let hits = vec![
            survivable_hit("alpha fact", &mk),
            survivable_hit("beta fact", &mk),
            survivable_hit("gamma fact", &mk),
        ];

        // --- RetrieveFacts ---
        let facts_r = InMemoryFacts::default();
        facts_r.script_search_hits(hits.clone());
        let embedder_r = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker_r = deterministic_reranker();
        let retrieve = RetrieveFacts {
            facts: &facts_r,
            embedder: &embedder_r,
            reranker: &reranker_r,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let scored = retrieve
            .execute("explain rust ownership", &mk, None)
            .await
            .expect("ok");
        let retrieve_ids: Vec<String> = scored
            .iter()
            .map(|s| s.hit.id.as_str().to_string())
            .collect();

        // --- EnrichRequest (fresh session → no dedup removes anything) ---
        let facts_e = InMemoryFacts::default();
        facts_e.script_search_hits(hits.clone());
        let sessions_e = InMemorySessions::default();
        let embedder_e = ConstantEmbedder(vec![0.1, 0.2, 0.3]);
        let reranker_e = deterministic_reranker();
        let enrich = EnrichRequest {
            facts: &facts_e,
            sessions: &sessions_e,
            embedder: &embedder_e,
            reranker: &reranker_e,
            clock: &FixedClock(now_ts()),
            retrieval_cfg: &rcfg(),
            heat_cfg: &hcfg(),
        };
        let messages =
            vec![serde_json::json!({"role": "user", "content": "explain rust ownership"})];
        let enriched = enrich.execute(messages, &mk, &sid()).await.expect("ok");
        let enrich_ids = extract_fact_ids_from_block(&enriched);

        assert_eq!(
            retrieve_ids, enrich_ids,
            "RetrieveFacts ranking must equal EnrichRequest's pre-dedup survivor order"
        );
    }

    /// Pull the `[fact_id]` tokens out of the injected `<smos-memory>` block,
    /// preserving order. The block is prepended to `messages[0].content`; the
    /// only lines that start with `[` are the fact lines, so the scan is
    /// unambiguous.
    fn extract_fact_ids_from_block(messages: &[serde_json::Value]) -> Vec<String> {
        let content = messages
            .first()
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        content
            .lines()
            .filter_map(|line| {
                let line = line.trim();
                let rest = line.strip_prefix('[')?;
                let close = rest.find(']')?;
                Some(rest[..close].to_string())
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Sanity: the survivable_hit fixture actually survives the pre-filter
    // -----------------------------------------------------------------------

    #[test]
    fn survivable_hit_fixture_passes_prefilter() {
        use crate::helpers::retrieval_pipeline::hit_to_retrieval;
        use crate::helpers::retrieval_planner::prefilter_and_heat;
        let mk = key();
        let hit = survivable_hit("alpha fact", &mk);
        let projected = hit_to_retrieval(hit).expect("maps");
        assert_eq!(projected.status, FactStatus::Accepted);
        let survivors = prefilter_and_heat(&[projected], &rcfg(), &hcfg(), now_ts());
        assert_eq!(survivors.len(), 1, "fixture must clear pre + heat filter");
        // Pin the heat fields so a future fixture change does not silently
        // break the "survivable" guarantee the rerank tests rely on.
        assert_eq!(survivors[0].heat_base, Heat::MAX);
    }
}
