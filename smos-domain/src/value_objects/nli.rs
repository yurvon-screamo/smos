//! `NliResult` / `NliScores` — natural-language-inference verdict value objects.
//!
//! The actual DeBERTa classifier lives in an adapter; the domain layer owns the
//! policy: the canonical exact-match result and the threshold-based predicates
//! that downstream merge and confidence logic consume.

use crate::config::NliConfig;
use crate::enums::{MergeReason, NliLabel};
use serde::{Deserialize, Serialize};

/// Per-label softmax scores produced by an NLI classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NliScores {
    pub entailment: f32,
    pub neutral: f32,
    pub contradiction: f32,
}

/// Full NLI verdict for a fact pair.
///
/// `available = false` marks graceful-degradation placeholders emitted when the
/// classifier is unreachable. Downstream code must NOT treat those as "no
/// contradiction detected" — they mean "not checked".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NliResult {
    pub label: NliLabel,
    pub scores: NliScores,
    pub available: bool,
}

impl NliResult {
    /// The canonical NLI result returned for an exact text match.
    ///
    /// Identical text is entailment by definition; this bypasses the model and
    /// avoids DeBERTa's known quirk of returning `neutral` on identical pairs.
    pub fn exact_match_result() -> Self {
        Self {
            label: NliLabel::Entailment,
            scores: NliScores {
                entailment: 1.0,
                neutral: 0.0,
                contradiction: 0.0,
            },
            available: true,
        }
    }

    /// `true` iff the contradiction label dominates above the configured threshold.
    pub fn is_contradiction(&self, cfg: &NliConfig) -> bool {
        self.label == NliLabel::Contradiction
            && self.scores.contradiction >= cfg.contradiction_threshold
    }

    /// `true` iff the entailment label dominates above the configured threshold.
    pub fn is_entailment(&self, cfg: &NliConfig) -> bool {
        self.label == NliLabel::Entailment && self.scores.entailment >= cfg.entailment_threshold
    }

    /// Classify the NLI verdict into a [`MergeReason`].
    ///
    /// - `available = false`  → `NeutralSkipped` (refuse to guess, §5.3).
    /// - contradiction         → `Drift`.
    /// - entailment            → `Merged`.
    /// - neutral               → `NeutralSkipped`.
    pub fn decide_merge(&self, cfg: &NliConfig) -> MergeReason {
        if !self.available {
            return MergeReason::NeutralSkipped;
        }
        if self.is_contradiction(cfg) {
            return MergeReason::Drift;
        }
        if self.is_entailment(cfg) {
            return MergeReason::Merged;
        }
        MergeReason::NeutralSkipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NliConfig {
        NliConfig::default()
    }

    fn nli(
        label: NliLabel,
        available: bool,
        entailment: f32,
        neutral: f32,
        contradiction: f32,
    ) -> NliResult {
        NliResult {
            label,
            scores: NliScores {
                entailment,
                neutral,
                contradiction,
            },
            available,
        }
    }

    #[test]
    fn exact_match_result_is_entailment_with_full_scores() {
        let r = NliResult::exact_match_result();
        assert_eq!(r.label, NliLabel::Entailment);
        assert_eq!(r.scores.entailment, 1.0);
        assert_eq!(r.scores.neutral, 0.0);
        assert_eq!(r.scores.contradiction, 0.0);
        assert!(r.available);
    }

    #[test]
    fn is_contradiction_true_when_label_and_score_dominate() {
        let r = nli(NliLabel::Contradiction, true, 0.1, 0.2, 0.7);
        assert!(r.is_contradiction(&cfg()));
    }

    #[test]
    fn is_contradiction_false_when_label_is_entailment() {
        let r = nli(NliLabel::Entailment, true, 0.9, 0.05, 0.05);
        assert!(!r.is_contradiction(&cfg()));
    }

    #[test]
    fn is_contradiction_false_when_score_below_threshold() {
        let r = nli(NliLabel::Contradiction, true, 0.4, 0.2, 0.4);
        assert!(!r.is_contradiction(&cfg()));
    }

    #[test]
    fn is_entailment_true_when_label_and_score_dominate() {
        let r = nli(NliLabel::Entailment, true, 0.8, 0.1, 0.1);
        assert!(r.is_entailment(&cfg()));
    }

    #[test]
    fn is_entailment_false_when_label_is_contradiction() {
        let r = nli(NliLabel::Contradiction, true, 0.3, 0.2, 0.5);
        assert!(!r.is_entailment(&cfg()));
    }

    #[test]
    fn is_entailment_false_when_score_below_threshold() {
        let r = nli(NliLabel::Entailment, true, 0.5, 0.3, 0.2);
        assert!(!r.is_entailment(&cfg()));
    }

    #[test]
    fn decide_merge_entailment_yields_merged() {
        let r = nli(NliLabel::Entailment, true, 0.8, 0.1, 0.1);
        assert_eq!(r.decide_merge(&cfg()), MergeReason::Merged);
    }

    #[test]
    fn decide_merge_contradiction_yields_drift() {
        let r = nli(NliLabel::Contradiction, true, 0.1, 0.1, 0.8);
        assert_eq!(r.decide_merge(&cfg()), MergeReason::Drift);
    }

    #[test]
    fn decide_merge_neutral_yields_neutral_skipped() {
        let r = nli(NliLabel::Neutral, true, 0.1, 0.8, 0.1);
        assert_eq!(r.decide_merge(&cfg()), MergeReason::NeutralSkipped);
    }

    #[test]
    fn decide_merge_unavailable_yields_neutral_skipped() {
        let r = nli(NliLabel::Entailment, false, 1.0, 0.0, 0.0);
        assert_eq!(r.decide_merge(&cfg()), MergeReason::NeutralSkipped);
    }

    #[test]
    fn decide_merge_entailment_below_threshold_yields_neutral_skipped() {
        let r = nli(NliLabel::Entailment, true, 0.5, 0.3, 0.2);
        assert_eq!(r.decide_merge(&cfg()), MergeReason::NeutralSkipped);
    }

    #[test]
    fn serde_roundtrip_preserves_nli_result() {
        let r = nli(NliLabel::Contradiction, true, 0.1, 0.2, 0.7);
        let json = serde_json::to_string(&r).unwrap();
        let back: NliResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
