//! Canonical fact validation status (ТРЕБОВАНИЯ.md §6).

use serde::{Deserialize, Serialize};

/// Lifecycle stage of a [`crate::entities::Fact`].
///
/// Facts start as [`Pending`](Self::Pending) at extraction time and transition
/// to a terminal state (`Accepted` or `Rejected`) after the session-end NLI
/// finalize pass (§5). Terminal states cannot transition out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FactStatus {
    Pending,
    Accepted,
    Rejected,
}

impl FactStatus {
    /// `true` for terminal states (no outgoing transitions allowed).
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Accepted | Self::Rejected)
    }

    /// Stable lowercase wire representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }
}

impl std::fmt::Display for FactStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_is_not_terminal() {
        assert!(!FactStatus::Pending.is_terminal());
    }

    #[test]
    fn accepted_is_terminal() {
        assert!(FactStatus::Accepted.is_terminal());
    }

    #[test]
    fn rejected_is_terminal() {
        assert!(FactStatus::Rejected.is_terminal());
    }

    #[test]
    fn as_str_matches_lowercase_serde_token() {
        assert_eq!(FactStatus::Pending.as_str(), "pending");
        assert_eq!(FactStatus::Accepted.as_str(), "accepted");
        assert_eq!(FactStatus::Rejected.as_str(), "rejected");
    }

    #[test]
    fn serde_roundtrip_preserves_status() {
        for status in [
            FactStatus::Pending,
            FactStatus::Accepted,
            FactStatus::Rejected,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: FactStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, back);
        }
    }

    #[test]
    fn serde_serializes_to_lowercase_string() {
        let json = serde_json::to_string(&FactStatus::Accepted).unwrap();
        assert_eq!(json, "\"accepted\"");
    }

    #[test]
    fn display_matches_as_str() {
        assert_eq!(FactStatus::Pending.to_string(), "pending");
        assert_eq!(FactStatus::Accepted.to_string(), "accepted");
        assert_eq!(FactStatus::Rejected.to_string(), "rejected");
    }
}
