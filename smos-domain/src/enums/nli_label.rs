//! NLI label space produced by the cross-encoder contradiction classifier.

use serde::{Deserialize, Serialize};

/// Three-way label emitted by the DeBERTa NLI cross-encoder (§9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NliLabel {
    Entailment,
    Neutral,
    Contradiction,
}

impl NliLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entailment => "entailment",
            Self::Neutral => "neutral",
            Self::Contradiction => "contradiction",
        }
    }
}

impl std::fmt::Display for NliLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_returns_lowercase_token() {
        assert_eq!(NliLabel::Entailment.as_str(), "entailment");
        assert_eq!(NliLabel::Neutral.as_str(), "neutral");
        assert_eq!(NliLabel::Contradiction.as_str(), "contradiction");
    }

    #[test]
    fn serde_roundtrip_preserves_label() {
        for label in [
            NliLabel::Entailment,
            NliLabel::Neutral,
            NliLabel::Contradiction,
        ] {
            let json = serde_json::to_string(&label).unwrap();
            let back: NliLabel = serde_json::from_str(&json).unwrap();
            assert_eq!(label, back);
        }
    }

    #[test]
    fn serde_serializes_to_lowercase_string() {
        assert_eq!(
            serde_json::to_string(&NliLabel::Contradiction).unwrap(),
            "\"contradiction\""
        );
    }
}
