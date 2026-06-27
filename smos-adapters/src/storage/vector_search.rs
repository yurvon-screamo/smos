use smos_application::{errors::RepoError, types::SearchHit};
use smos_domain::{Fact, FactId, FactStatus, MemoryKey};

use super::rows::{FactRow, SearchSimilarRow};
use super::surreal_store::SurrealStore;

/// Status set a vector search is allowed to return.
///
/// [`VectorSearchScope::Retrieval`] backs §3 enrichment: it must see only
/// accepted facts so an unconfirmed pending claim never leaks into a chat
/// context. [`VectorSearchScope::Dedup`] backs the extraction pipeline's
/// Layer 2 safety net: it must additionally include pending facts because
/// the cross-session confirmation that promotes them past the accept
/// threshold can only fire if the search finds them.
#[derive(Clone, Copy)]
pub(crate) enum VectorSearchScope {
    Retrieval,
    Dedup,
}

impl VectorSearchScope {
    /// SQL fragment for the equality-prefiltered brute-force pass.
    fn status_predicate(&self) -> &'static str {
        match self {
            Self::Retrieval => "status = 'accepted'",
            Self::Dedup => "(status = 'accepted' OR status = 'pending')",
        }
    }

    /// Rust predicate mirroring [`Self::status_predicate`] for the
    /// post-filtered HNSW pass.
    fn allows_status(&self, status: &str) -> bool {
        match self {
            Self::Retrieval => status == "accepted",
            Self::Dedup => status == "accepted" || status == "pending",
        }
    }
}

impl SurrealStore {
    /// Two-stage vector search (HNSW + brute-force fallback) shared by
    /// [`FactRepository::search_similar`] and
    /// [`FactRepository::search_for_dedup`]. The scope controls the status
    /// predicate applied in both passes.
    ///
    /// 1. Pull `over_fetch` nearest neighbours from the HNSW index WITHOUT
    ///    equality pre-filters. The AC0 spike proved that combining the
    ///    KNN operator with `memory_key = $mk AND status = 'accepted'`
    ///    returns zero rows on SurrealDB 2.6 — the planner can't fold
    ///    equality predicates into the HNSW traversal. Issuing the KNN
    ///    alone and post-filtering in Rust is the validated workaround.
    ///
    /// 2. Filter the HNSW candidates by memory_key + status + valid_until.
    ///
    /// 3. ALWAYS run the brute-force cosine scan as well, regardless of
    ///    how many hits the HNSW pass returned. The HNSW index is
    ///    approximate: under skewed namespaces or post-deletion churn it
    ///    can miss a true neighbour that an exact cosine scan finds. The
    ///    brute-force pass is the correctness backstop; the cost is one
    ///    extra round-trip per search (acceptable for the SMOS workload —
    ///    every search is per-request, not per-token).
    ///
    /// 4. Merge the two passes (dedup by FactId, prefer the smaller
    ///    `distance`), sort ascending, and truncate to `limit`.
    pub(crate) async fn vector_search(
        &self,
        embedding: Vec<f32>,
        memory_key: &MemoryKey,
        limit: usize,
        scope: VectorSearchScope,
    ) -> Result<Vec<SearchHit>, RepoError> {
        let over_fetch = (limit * 4).max(limit + 8);
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();

        let hnsw_hits = self
            .search_similar_hnsw(&embedding_f64, over_fetch, memory_key, scope)
            .await?;
        let bf_hits = self
            .search_similar_bruteforce(&embedding_f64, memory_key, limit, scope)
            .await?;

        // Dedup by FactId, keeping the hit with the smaller distance on
        // collisions. HNSW and brute-force often return the same FactId
        // with slightly different distances; preferring the smaller one
        // surfaces the most similar fact regardless of which pass found it.
        let mut merged: Vec<SearchHit> = Vec::with_capacity(hnsw_hits.len() + bf_hits.len());
        let mut seen: std::collections::HashMap<FactId, usize> =
            std::collections::HashMap::with_capacity(hnsw_hits.len() + bf_hits.len());
        for hit in hnsw_hits.into_iter().chain(bf_hits) {
            match seen.get(&hit.id) {
                None => {
                    seen.insert(hit.id.clone(), merged.len());
                    merged.push(hit);
                }
                Some(&idx) => {
                    let incumbent = &merged[idx];
                    let incumbent_dist = incumbent.metadata.distance.unwrap_or(f32::INFINITY);
                    let new_dist = hit.metadata.distance.unwrap_or(f32::INFINITY);
                    if new_dist < incumbent_dist {
                        merged[idx] = hit;
                    }
                }
            }
        }
        merged.sort_by(|a, b| {
            a.metadata
                .distance
                .partial_cmp(&b.metadata.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(limit);
        Ok(merged)
    }

    /// HNSW-backed KNN pass — no equality pre-filter, post-filtered in Rust.
    /// Returns up to `over_fetch` hits filtered by `memory_key`,
    /// `scope.allows_status`, and `valid_until = NONE`.
    pub(crate) async fn search_similar_hnsw(
        &self,
        embedding_f64: &[f64],
        over_fetch: usize,
        memory_key: &MemoryKey,
        scope: VectorSearchScope,
    ) -> Result<Vec<SearchHit>, RepoError> {
        // The KNN operator `<|K,EF|>` requires literal integers — SurrealQL's
        // parser rejects a bound parameter in that position. We interpolate
        // the values directly (they are derived from `limit`, which is
        // application-controlled, so this is safe).
        let sql = format!(
            "SELECT id, content, memory_key, status, confidence,
                    valid_until, heat_base, last_access_at, extracted_at, conflicts_with,
                    vector::distance::knn() AS distance
             FROM fact
             WHERE embedding <|{over_fetch}, 64|> $embedding
             ORDER BY distance;"
        );
        let mut res = self
            .db
            .query(&sql)
            .bind(("embedding", embedding_f64.to_vec()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "search_similar_hnsw")?;
        let rows: Vec<SearchSimilarRow> = res.take(0).map_err(Self::map_db_error)?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r.to_hit(memory_key))
            .filter(|h| scope.allows_status(&h.metadata.status))
            .filter(|h| h.metadata.valid_until.is_none())
            .collect())
    }

    /// Brute-force cosine pass with equality pre-filters. Slower than HNSW
    /// but immune to the planner limitation that breaks KNN + filter.
    ///
    /// Returns `distance = 1.0 - similarity` so the metric is consistent with
    /// the HNSW pass (smaller distance = more similar) and the merge-sort in
    /// `vector_search` orders both passes by the same key.
    pub(crate) async fn search_similar_bruteforce(
        &self,
        embedding_f64: &[f64],
        memory_key: &MemoryKey,
        limit: usize,
        scope: VectorSearchScope,
    ) -> Result<Vec<SearchHit>, RepoError> {
        // Inline the status predicate so the planner can fold it together
        // with `memory_key` / `valid_until` into one index seek.
        let sql = format!(
            "SELECT id, content, memory_key, status, confidence,
                    valid_until, heat_base, last_access_at, extracted_at, conflicts_with,
                    (1.0 - vector::similarity::cosine(embedding, $embedding)) AS distance
             FROM fact
             WHERE memory_key = $mk AND {status_pred} AND valid_until = NONE
             ORDER BY distance ASC
             LIMIT $limit;",
            status_pred = scope.status_predicate()
        );
        let mut res = self
            .db
            .query(&sql)
            .bind(("mk", memory_key.as_str().to_string()))
            .bind(("embedding", embedding_f64.to_vec()))
            .bind(("limit", limit as i64))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "search_similar_bruteforce")?;
        let rows: Vec<SearchSimilarRow> = res.take(0).map_err(Self::map_db_error)?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r.to_hit(memory_key))
            .collect())
    }
}

/// Internal helper for `list_accepted` / `list_pending`.
impl SurrealStore {
    pub(crate) async fn list_by_status(
        &self,
        memory_key: &MemoryKey,
        status: FactStatus,
    ) -> Result<Vec<Fact>, RepoError> {
        let mut res = self
            .db
            .query(
                "SELECT * FROM fact
                 WHERE memory_key = $mk AND status = $status;",
            )
            .bind(("mk", memory_key.as_str().to_string()))
            .bind(("status", status.as_str().to_string()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "query")?;
        let rows: Vec<FactRow> = res.take(0).map_err(Self::map_db_error)?;
        rows.into_iter()
            .map(|r| {
                // Reconstruct the FactId from the row's content. The Fact
                // aggregate's invariant `id == FactId::from_content(content)`
                // is enforced by `Fact::rehydrate`, so this matches the row's
                // Surreal record id (`fact:<fact_id_string>`) by construction.
                let fact_id = FactId::from_content(&r.content);
                r.to_fact(fact_id)
            })
            .collect()
    }
}
