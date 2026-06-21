//! Retrieval planner — pure pre-filter and heat post-filter (§3 step 4).
//!
//! Mirrors the POC `_prefilter_and_heat`: given a list of raw vector-search
//! hits, drop everything that fails a hard pre-filter (status, validity window,
//! confidence) or whose decayed live heat is at/below the minimum threshold.

use smos_domain::config::{HeatConfig, RetrievalConfig};
use smos_domain::enums::FactStatus;
use smos_domain::{Confidence, FactId, Heat, MemoryKey, Timestamp};

/// Minimal projection of a vector-search hit that the pre-filter consumes.
///
/// The adapter layer builds these from ChromaDB / SurrealDB rows; the
/// application layer never touches the DB driver directly.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalHit {
    pub id: FactId,
    pub document: String,
    pub memory_key: MemoryKey,
    pub status: FactStatus,
    pub confidence: Confidence,
    pub valid_until: Option<Timestamp>,
    pub heat_base: Heat,
    pub last_access_at: Timestamp,
}

impl RetrievalHit {
    /// Apply the hard pre-filters (status / validity / confidence) to `self`.
    ///
    /// `valid_until` is a tombstone marker (§6 frontmatter): a non-null value
    /// means the fact has been deprecated at finalize time. The presence of the
    /// tombstone is the filter — the wall-clock time of the tombstone itself is
    /// not consulted, because by convention facts are tombstoned only when
    /// already expired.
    pub fn passes_prefilters(&self, cfg: &RetrievalConfig) -> bool {
        if self.status != FactStatus::Accepted {
            return false;
        }
        if self.valid_until.is_some() {
            return false;
        }
        if self.confidence.value() < cfg.min_confidence {
            return false;
        }
        true
    }

    /// Live heat computed from `heat_base` + `last_access_at`.
    ///
    /// Delegates to the canonical [`Heat::decay`] formula in the domain layer
    /// so the decay curve has a single source of truth.
    pub fn heat_live(&self, decay_rate: f32, now: Timestamp) -> f32 {
        Heat::decay(self.heat_base, self.last_access_at, now, decay_rate)
    }
}

/// Apply hard pre-filters then the heat post-filter, preserving order.
pub fn prefilter_and_heat(
    hits: &[RetrievalHit],
    cfg: &RetrievalConfig,
    heat_cfg: &HeatConfig,
    now: Timestamp,
) -> Vec<RetrievalHit> {
    hits.iter()
        .filter(|h| h.passes_prefilters(cfg))
        .filter(|h| h.heat_live(heat_cfg.decay_rate, now) > heat_cfg.min_threshold)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> Timestamp {
        Timestamp::from_unix_secs(1_700_000_000).unwrap()
    }

    fn key() -> MemoryKey {
        MemoryKey::from_raw("origa").unwrap()
    }

    fn fid(tag: &str) -> FactId {
        FactId::from_content(tag)
    }

    fn hit(status: FactStatus) -> RetrievalHit {
        RetrievalHit {
            id: fid("hit"),
            document: "doc".to_string(),
            memory_key: key(),
            status,
            confidence: Confidence::new(0.8).unwrap(),
            valid_until: None,
            heat_base: Heat::new(1.0).unwrap(),
            last_access_at: now(),
        }
    }

    fn rcfg() -> RetrievalConfig {
        RetrievalConfig::default()
    }

    fn hcfg() -> HeatConfig {
        HeatConfig::default()
    }

    #[test]
    fn accepted_hit_with_high_confidence_passes() {
        let hits = vec![hit(FactStatus::Accepted)];
        let out = prefilter_and_heat(&hits, &rcfg(), &hcfg(), now());
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn pending_hit_is_filtered_out() {
        let hits = vec![hit(FactStatus::Pending)];
        let out = prefilter_and_heat(&hits, &rcfg(), &hcfg(), now());
        assert!(out.is_empty());
    }

    #[test]
    fn rejected_hit_is_filtered_out() {
        let hits = vec![hit(FactStatus::Rejected)];
        let out = prefilter_and_heat(&hits, &rcfg(), &hcfg(), now());
        assert!(out.is_empty());
    }

    #[test]
    fn hit_with_valid_until_is_filtered_out() {
        let mut h = hit(FactStatus::Accepted);
        h.valid_until = Some(now());
        let out = prefilter_and_heat(&[h], &rcfg(), &hcfg(), now());
        assert!(out.is_empty());
    }

    #[test]
    fn hit_below_min_confidence_is_filtered_out() {
        let mut h = hit(FactStatus::Accepted);
        h.confidence = Confidence::new(0.5).unwrap();
        let out = prefilter_and_heat(&[h], &rcfg(), &hcfg(), now());
        assert!(out.is_empty());
    }

    #[test]
    fn hit_with_stale_heat_is_filtered_out() {
        let mut h = hit(FactStatus::Accepted);
        h.last_access_at = Timestamp::from_unix_secs(now().as_unix_secs() - 1000 * 3600).unwrap();
        let out = prefilter_and_heat(&[h], &rcfg(), &hcfg(), now());
        assert!(out.is_empty());
    }

    #[test]
    fn preserves_order_of_survivors() {
        let hits = vec![
            RetrievalHit {
                id: fid("a"),
                document: "first".to_string(),
                memory_key: key(),
                status: FactStatus::Accepted,
                confidence: Confidence::new(0.9).unwrap(),
                valid_until: None,
                heat_base: Heat::new(1.0).unwrap(),
                last_access_at: now(),
            },
            RetrievalHit {
                id: fid("b"),
                document: "second".to_string(),
                memory_key: key(),
                status: FactStatus::Accepted,
                confidence: Confidence::new(0.95).unwrap(),
                valid_until: None,
                heat_base: Heat::new(1.0).unwrap(),
                last_access_at: now(),
            },
        ];
        let out = prefilter_and_heat(&hits, &rcfg(), &hcfg(), now());
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, fid("a"));
        assert_eq!(out[1].id, fid("b"));
    }
}
