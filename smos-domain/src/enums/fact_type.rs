//! Fact taxonomy used across the pipeline.

use serde::{Deserialize, Serialize};

/// Categorical label of a fact's nature (mirrors the POC `FactType` literal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FactType {
    Decision,
    Preference,
    /// Default for freshly-extracted facts before classification runs.
    #[default]
    Entity,
    Event,
    Technical,
}

impl FactType {
    /// Stable lowercase wire representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Decision => "decision",
            Self::Preference => "preference",
            Self::Entity => "entity",
            Self::Event => "event",
            Self::Technical => "technical",
        }
    }
}

impl std::fmt::Display for FactType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_str_returns_lowercase_token() {
        assert_eq!(FactType::Decision.as_str(), "decision");
        assert_eq!(FactType::Preference.as_str(), "preference");
        assert_eq!(FactType::Entity.as_str(), "entity");
        assert_eq!(FactType::Event.as_str(), "event");
        assert_eq!(FactType::Technical.as_str(), "technical");
    }

    #[test]
    fn serde_roundtrip_preserves_type() {
        for fact_type in [
            FactType::Decision,
            FactType::Preference,
            FactType::Entity,
            FactType::Event,
            FactType::Technical,
        ] {
            let json = serde_json::to_string(&fact_type).unwrap();
            let back: FactType = serde_json::from_str(&json).unwrap();
            assert_eq!(fact_type, back);
        }
    }

    #[test]
    fn default_is_entity() {
        assert_eq!(FactType::default(), FactType::Entity);
    }
}
