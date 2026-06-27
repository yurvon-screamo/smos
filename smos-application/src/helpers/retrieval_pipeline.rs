//! Retrieval pipeline — shared rerank + projection helpers (§3 step 8).
//!
//! Extracted from `enrich_request` so the read-only `RetrieveFacts` use case
//! (the BEAM benchmark adapter contract) re-uses the EXACT rerank + projection
//! path the live enrichment pipeline runs. Both use cases call [`rerank_hits`]
//! and [`hit_to_retrieval`] here, so their ranking is identical by
//! construction — pinned by the parity test in `retrieve_facts::tests`.

use smos_domain::enums::FactStatus;
use smos_domain::{Confidence, Heat, Timestamp};

use crate::errors::ProviderError;
use crate::helpers::retrieval_planner::RetrievalHit;
use crate::ports::RerankProvider;
use crate::types::SearchHit;

/// One rerank-ordered survivor with its cross-encoder relevance score.
///
/// `score` is the raw `RerankResult.score` (higher = more relevant). It is
/// carried alongside the hit so the read-only search surface can surface it
/// to the caller without re-running the reranker; the live enrichment path
/// ignores it (it only needs the ordered hits).
#[derive(Debug, Clone, PartialEq)]
pub struct RankedHit {
    pub hit: RetrievalHit,
    pub score: f32,
}

/// Rerank survivors with the cross-encoder and return them ordered by
/// descending relevance, each paired with its score.
///
/// Fail-closed contract (mirrors the former `EnrichRequest::rerank_survivors`
/// verbatim): a provider error OR a provider response with zero results
/// propagates as `Err(ProviderError::…)`. An `Ok(_)` is returned only when the
/// provider responded with at least one index that maps back to a survivor;
/// the filter-map can still yield an empty `Vec` when every returned index is
/// out of range, and the caller's defensive guard converts that to `Err` too.
pub async fn rerank_hits<R: RerankProvider>(
    topic: &str,
    survivors: &[RetrievalHit],
    reranker: &R,
    top_k: usize,
) -> Result<Vec<RankedHit>, ProviderError> {
    let documents: Vec<String> = survivors.iter().map(|s| s.document.clone()).collect();
    let ranked = reranker
        .rerank(topic, &documents, top_k)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "reranker unavailable; request will fail with 503");
            e
        })?;
    if ranked.is_empty() {
        tracing::error!("reranker returned empty results; request will fail with 503");
        return Err(ProviderError::InvalidResponse(
            "reranker returned empty results".to_string(),
        ));
    }
    Ok(ranked
        .into_iter()
        .filter_map(|r| {
            survivors.get(r.index).cloned().map(|hit| RankedHit {
                hit,
                score: r.score,
            })
        })
        .collect())
}

/// Map a single `SearchHit` to a `RetrievalHit`. Drops rows whose typed fields
/// cannot be reconstructed (status / confidence / heat) so a corrupt row never
/// poisons the pipeline — it is logged and skipped.
///
/// The SearchHit-only fields (`distance`, `created_at`, `conflicts_with`) are
/// intentionally NOT carried into the `RetrievalHit` projection: the live
/// enrichment pipeline never reads them, and the read-only search surface
/// recovers them from the original `SearchHit` via a FactId lookup so the
/// projection stays minimal.
pub fn hit_to_retrieval(hit: SearchHit) -> Option<RetrievalHit> {
    let status = match parse_fact_status(&hit.metadata.status) {
        Some(s) => s,
        None => {
            tracing::warn!(status = %hit.metadata.status, "unparseable status; dropping hit");
            return None;
        }
    };
    let confidence = match Confidence::new(hit.metadata.confidence) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, "out-of-range confidence; dropping hit");
            return None;
        }
    };
    let heat = match Heat::new(hit.metadata.heat_base) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(error = %e, "out-of-range heat_base; dropping hit");
            return None;
        }
    };
    let last_access_at = match Timestamp::from_unix_secs(hit.metadata.last_access_at as i64) {
        Ok(ts) => ts,
        Err(e) => {
            tracing::warn!(error = %e, "out-of-range last_access_at; dropping hit");
            return None;
        }
    };
    // `valid_until` is stored as an ISO-8601 string by the adapter; an absent
    // tombstone (`None`) means the fact is still current. Parse failures are
    // logged and treated as `None` so a corrupt row never blocks retrieval.
    let valid_until = hit
        .metadata
        .valid_until
        .as_deref()
        .and_then(parse_iso_timestamp);
    Some(RetrievalHit {
        id: hit.id,
        document: hit.document,
        memory_key: hit.memory_key,
        status,
        confidence,
        valid_until,
        heat_base: heat,
        last_access_at,
    })
}

/// Map a wire-formatted status string to a `FactStatus`. Compares against
/// each canonical lowercase token (`FactStatus::as_str`) so the wire contract
/// has a single source of truth in the domain. Returns `None` for unknown
/// values (logged by the caller).
pub fn parse_fact_status(s: &str) -> Option<FactStatus> {
    [
        FactStatus::Pending,
        FactStatus::Accepted,
        FactStatus::Rejected,
    ]
    .into_iter()
    .find(|candidate| s == candidate.as_str())
}

/// Parse an ISO-8601 string into a `Timestamp` via the `time` crate's
/// `Rfc3339` parser. Returns `None` on any failure.
pub fn parse_iso_timestamp(s: &str) -> Option<Timestamp> {
    use time::OffsetDateTime;
    let odt = OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()?;
    Timestamp::from_unix_secs(odt.unix_timestamp()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SearchHitMetadata;
    use smos_domain::{FactId, MemoryKey};

    fn sample_hit(
        status: &str,
        confidence: f32,
        heat_base: f32,
        last_access_at: f32,
        valid_until: Option<&str>,
    ) -> SearchHit {
        SearchHit {
            id: FactId::from_raw("fact_0123456789abcdef").expect("fact id"),
            document: "doc".into(),
            memory_key: MemoryKey::from_raw("origa").expect("memory key"),
            metadata: SearchHitMetadata {
                status: status.into(),
                confidence,
                valid_until: valid_until.map(str::to_string),
                heat_base,
                last_access_at,
                distance: Some(0.1),
                created_at: None,
                conflicts_with: Vec::new(),
            },
        }
    }

    // -----------------------------------------------------------------------
    // parse_fact_status — wire-format → enum mapping
    // -----------------------------------------------------------------------

    #[test]
    fn parse_fact_status_recognises_canonical_tokens() {
        assert_eq!(parse_fact_status("pending"), Some(FactStatus::Pending));
        assert_eq!(parse_fact_status("accepted"), Some(FactStatus::Accepted));
        assert_eq!(parse_fact_status("rejected"), Some(FactStatus::Rejected));
    }

    #[test]
    fn parse_fact_status_rejects_unknown_tokens() {
        assert_eq!(parse_fact_status("invalid"), None);
        assert_eq!(parse_fact_status(""), None);
    }

    #[test]
    fn parse_fact_status_is_case_sensitive() {
        // The wire contract is the lowercase token emitted by `as_str`; a
        // case mismatch is treated as unknown so the row is dropped rather
        // than silently re-interpreted.
        assert_eq!(parse_fact_status("Accepted"), None);
        assert_eq!(parse_fact_status("ACCEPTED"), None);
    }

    // -----------------------------------------------------------------------
    // parse_iso_timestamp — Rfc3339 → Timestamp mapping
    // -----------------------------------------------------------------------

    #[test]
    fn parse_iso_timestamp_accepts_rfc3339_utc() {
        let ts = parse_iso_timestamp("2025-06-18T12:00:00Z").expect("valid rfc3339");
        assert_eq!(ts.as_unix_secs(), 1_750_248_000);
    }

    #[test]
    fn parse_iso_timestamp_accepts_offset_form() {
        let ts = parse_iso_timestamp("2025-06-18T12:00:00+00:00").expect("valid offset");
        assert_eq!(ts.as_unix_secs(), 1_750_248_000);
    }

    #[test]
    fn parse_iso_timestamp_rejects_malformed_strings() {
        assert_eq!(parse_iso_timestamp("not a date"), None);
        assert_eq!(parse_iso_timestamp(""), None);
        assert_eq!(parse_iso_timestamp("2025-06-18"), None);
    }

    // -----------------------------------------------------------------------
    // hit_to_retrieval — SearchHit → RetrievalHit projection
    // -----------------------------------------------------------------------

    #[test]
    fn hit_to_retrieval_maps_well_formed_hit() {
        let hit = sample_hit("accepted", 0.85, 1.0, 1_700_000_000.0, None);
        let r = hit_to_retrieval(hit).expect("mapped");
        assert_eq!(r.status, FactStatus::Accepted);
        assert!((r.confidence.value() - 0.85).abs() < 1e-6);
        assert!((r.heat_base.value() - 1.0).abs() < 1e-6);
        assert_eq!(r.last_access_at.as_unix_secs(), 1_700_000_000);
        assert!(r.valid_until.is_none());
    }

    #[test]
    fn hit_to_retrieval_carries_valid_until_tombstone() {
        let hit = sample_hit(
            "accepted",
            0.9,
            0.5,
            1_700_000_000.0,
            Some("2025-12-31T00:00:00Z"),
        );
        let r = hit_to_retrieval(hit).expect("mapped");
        assert!(r.valid_until.is_some());
    }

    #[test]
    fn hit_to_retrieval_drops_hit_with_unknown_status() {
        let hit = sample_hit("weird", 0.9, 1.0, 1_700_000_000.0, None);
        assert!(hit_to_retrieval(hit).is_none());
    }

    #[test]
    fn hit_to_retrieval_drops_hit_with_out_of_range_confidence() {
        // 1.5 is outside [0,1]; Confidence::new rejects it.
        let hit = sample_hit("accepted", 1.5, 1.0, 1_700_000_000.0, None);
        assert!(hit_to_retrieval(hit).is_none());
    }

    #[test]
    fn hit_to_retrieval_drops_hit_with_out_of_range_heat() {
        let hit = sample_hit("accepted", 0.9, 2.0, 1_700_000_000.0, None);
        assert!(hit_to_retrieval(hit).is_none());
    }

    #[test]
    fn hit_to_retrieval_drops_hit_with_out_of_range_last_access_at() {
        // `f32::INFINITY` saturates to `i64::MAX` on `as i64` cast; that
        // overflows the `OffsetDateTime` year range and the typed timestamp
        // rejects it, so the row is dropped.
        let hit = sample_hit("accepted", 0.9, 1.0, f32::INFINITY, None);
        assert!(hit_to_retrieval(hit).is_none());
    }

    #[test]
    fn hit_to_retrieval_treats_malformed_valid_until_as_none() {
        // A corrupt tombstone string must not poison the row — it is logged
        // and the fact stays current (None tombstone).
        let hit = sample_hit("accepted", 0.9, 1.0, 1_700_000_000.0, Some("not-a-date"));
        let r = hit_to_retrieval(hit).expect("mapped despite malformed valid_until");
        assert!(r.valid_until.is_none());
    }
}
