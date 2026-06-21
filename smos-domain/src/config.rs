//! Pure configuration value objects.
//!
//! These structs hold tunable thresholds and bonuses only. Loading values from
//! environment variables or files is the responsibility of an adapter (slice 3);
//! the domain layer merely consumes the values.

use serde::{Deserialize, Serialize};

/// Confidence formula coefficients and gate thresholds (§5.4, §9).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceConfig {
    /// Baseline confidence assigned to every freshly-extracted fact.
    pub base: f32,
    /// Bonus when 2+ distinct sessions have observed the fact.
    pub multi_source_bonus: f32,
    /// Bonus when NLI ran and did not flag a contradiction.
    pub no_contradiction_bonus: f32,
    /// Confidence at/above which a fact is promoted to `Accepted`.
    pub accept_threshold: f32,
    /// Confidence at/above which a fact stays `Pending` (below this: `Rejected`).
    pub pending_threshold: f32,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            base: 0.5,
            multi_source_bonus: 0.2,
            no_contradiction_bonus: 0.1,
            accept_threshold: 0.7,
            pending_threshold: 0.4,
        }
    }
}

/// NLI verdict thresholds consumed by the domain layer (§5.5, §9).
///
/// The domain only needs the threshold pair that drives
/// [`NliResult::is_contradiction`] / [`NliResult::is_entailment`] and the
/// merge decision in [`NliResult::decide_merge`]. The adapter-boundary
/// strings (`model`, `cache_dir`) live in
/// `smos_adapters::config::NliBackendConfig` so this crate stays free of
/// the "domain type carries data only an adapter can interpret" smell.
///
/// `deny_unknown_fields` turns the most common operator mistake (putting
/// `model` / `cache_dir` under `[nli]` instead of `[nli_backend]`) into a
/// loud startup error rather than a silent drop — the layering invariant
/// is enforced at parse time, not just by code review.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NliConfig {
    /// Minimum `contradiction` softmax score for the contradiction verdict.
    pub contradiction_threshold: f32,
    /// Minimum `entailment` softmax score for the entailment verdict.
    pub entailment_threshold: f32,
}

impl Default for NliConfig {
    fn default() -> Self {
        Self {
            contradiction_threshold: 0.5,
            entailment_threshold: 0.6,
        }
    }
}

/// Merge candidate selection threshold (§5.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConfig {
    /// Cosine similarity at/above which a fact pair is handed to NLI.
    pub cosine_threshold: f32,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            cosine_threshold: 0.85,
        }
    }
}

/// Heat decay curve parameters (§7).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatConfig {
    /// Exponential decay rate per hour.
    pub decay_rate: f32,
    /// Live heat values at or below this threshold are filtered out of retrieval.
    pub min_threshold: f32,
}

impl Default for HeatConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.03,
            min_threshold: 0.2,
        }
    }
}

/// Vector retrieval + post-filter tunables (§3 step 4-5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Candidates requested from the vector store before post-filtering.
    pub top_k_initial: usize,
    /// Surviving candidates after reranking.
    pub top_k_final: usize,
    /// Hard floor below which a fact is not retrieved.
    pub min_confidence: f32,
    /// Minimum non-whitespace chars in the topic; below this we skip enrichment.
    pub min_topic_chars: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k_initial: 50,
            top_k_final: 5,
            min_confidence: 0.7,
            min_topic_chars: 3,
        }
    }
}

/// Safety-net dedup thresholds for the extraction pipeline.
///
/// Even with `temperature = 0.0` + a pinned seed, an upstream model swap or a
/// prompt tweak can shift the phrasing of a fact just enough to break the
/// `FactId = SHA1(content)` exact match the cross-session confirmation path
/// relies on. The semantic layer lets `persist_facts` recognise "the same
/// fact said slightly differently" via cosine similarity and route it through
/// confirmation instead of leaving the fact stuck at single-source confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Cosine similarity at/above which a freshly-extracted fact is treated
    /// as a re-observation of an existing fact (`persist_facts` step 2).
    /// Conservative default 0.95 — only near-identical semantic duplicates
    /// cross the bar, so two genuinely different facts never collapse.
    pub dedup_cosine_threshold: f32,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            dedup_cosine_threshold: 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_defaults_match_spec() {
        let c = ConfidenceConfig::default();
        assert_eq!(c.base, 0.5);
        assert_eq!(c.multi_source_bonus, 0.2);
        assert_eq!(c.no_contradiction_bonus, 0.1);
        assert_eq!(c.accept_threshold, 0.7);
        assert_eq!(c.pending_threshold, 0.4);
    }

    #[test]
    fn nli_defaults_match_spec() {
        let n = NliConfig::default();
        assert_eq!(n.contradiction_threshold, 0.5);
        assert_eq!(n.entailment_threshold, 0.6);
    }

    #[test]
    fn merge_defaults_match_spec() {
        assert_eq!(MergeConfig::default().cosine_threshold, 0.85);
    }

    #[test]
    fn heat_defaults_match_spec() {
        let h = HeatConfig::default();
        assert_eq!(h.decay_rate, 0.03);
        assert_eq!(h.min_threshold, 0.2);
    }

    #[test]
    fn retrieval_defaults_match_spec() {
        let r = RetrievalConfig::default();
        assert_eq!(r.top_k_initial, 50);
        assert_eq!(r.top_k_final, 5);
        assert_eq!(r.min_confidence, 0.7);
        assert_eq!(r.min_topic_chars, 3);
    }

    #[test]
    fn extraction_defaults_match_spec() {
        let e = ExtractionConfig::default();
        assert_eq!(e.dedup_cosine_threshold, 0.95);
    }

    #[test]
    fn configs_roundtrip_serde() {
        let configs: Vec<String> = vec![
            serde_json::to_string(&ConfidenceConfig::default()).unwrap(),
            serde_json::to_string(&NliConfig::default()).unwrap(),
            serde_json::to_string(&MergeConfig::default()).unwrap(),
            serde_json::to_string(&HeatConfig::default()).unwrap(),
            serde_json::to_string(&RetrievalConfig::default()).unwrap(),
            serde_json::to_string(&ExtractionConfig::default()).unwrap(),
        ];
        for c in configs {
            assert!(c.len() > 2);
        }
    }
}
