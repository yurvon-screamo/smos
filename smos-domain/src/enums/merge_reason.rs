//! Merge outcome reason codes (§5 merge logic).

use serde::{Deserialize, Serialize};

/// Outcome category of attempting to merge a new fact into an existing one.
///
/// `snake_case` matches the Python POC's wire representation so persisted
/// markdown frontmatter and JSON events stay compatible across the rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeReason {
    /// Entailment detected — facts merged (DeBERTa-only path).
    Merged,
    /// Contradiction detected — conflicts_with flagged, no merge.
    Drift,
    /// Nothing similar enough to consider.
    NoCandidate,
    /// Semantically close but neither entailed nor contradictory.
    NeutralSkipped,
}

impl MergeReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Merged => "merged",
            Self::Drift => "drift",
            Self::NoCandidate => "no_candidate",
            Self::NeutralSkipped => "neutral_skipped",
        }
    }
}

impl std::fmt::Display for MergeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_returns_snake_case_token() {
        assert_eq!(MergeReason::Merged.as_str(), "merged");
        assert_eq!(MergeReason::Drift.as_str(), "drift");
        assert_eq!(MergeReason::NoCandidate.as_str(), "no_candidate");
        assert_eq!(MergeReason::NeutralSkipped.as_str(), "neutral_skipped");
    }

    #[test]
    fn serde_serializes_to_snake_case_string() {
        assert_eq!(
            serde_json::to_string(&MergeReason::NeutralSkipped).unwrap(),
            "\"neutral_skipped\""
        );
        assert_eq!(
            serde_json::to_string(&MergeReason::NoCandidate).unwrap(),
            "\"no_candidate\""
        );
    }

    #[test]
    fn serde_roundtrip_preserves_reason() {
        for reason in [
            MergeReason::Merged,
            MergeReason::Drift,
            MergeReason::NoCandidate,
            MergeReason::NeutralSkipped,
        ] {
            let json = serde_json::to_string(&reason).unwrap();
            let back: MergeReason = serde_json::from_str(&json).unwrap();
            assert_eq!(reason, back);
        }
    }
}
