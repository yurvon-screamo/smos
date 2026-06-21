//! `FactId` — deterministic, content-derived identifier.
//!
//! The id is `fact_<sha1(content)[:16]>`, so the same English canonical content
//! always produces the same id. This is what makes cross-session confirmation
//! (§4: `_confirm_existing_fact`) and dedup work without a separate lookup key.

use crate::error::DomainError;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

/// Canonical identifier of a stored fact.
///
/// Invariants enforced at construction:
/// - Starts with `fact_` followed by exactly 16 lowercase hex characters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FactId(String);

impl FactId {
    /// Build a `FactId` from raw content. Deterministic across calls.
    ///
    /// Same content always yields the same id — the foundation of cross-session
    /// confirmation and retrieval dedup.
    pub fn from_content(content: &str) -> Self {
        let mut hasher = Sha1::new();
        hasher.update(content.as_bytes());
        let digest = hasher.finalize();
        let hex = format!("{:x}", digest);
        Self(format!("fact_{}", &hex[..16]))
    }

    /// Parse from a pre-built string. Must match `^fact_[0-9a-f]{16}$`.
    pub fn from_raw(s: &str) -> Result<Self, DomainError> {
        if is_valid_fact_id(s) {
            Ok(Self(s.to_string()))
        } else {
            Err(DomainError::InvalidFactId(s.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn is_valid_fact_id(s: &str) -> bool {
    let Some(hex) = s.strip_prefix("fact_") else {
        return false;
    };
    hex.len() == 16
        && hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

impl std::fmt::Display for FactId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_content_is_deterministic_for_same_input() {
        let a = FactId::from_content("Rust is the systems language");
        let b = FactId::from_content("Rust is the systems language");
        assert_eq!(a, b);
    }

    #[test]
    fn from_content_differs_for_different_input() {
        let a = FactId::from_content("alpha");
        let b = FactId::from_content("beta");
        assert_ne!(a, b);
    }

    #[test]
    fn from_content_produces_fact_prefix_with_sixteen_hex() {
        let id = FactId::from_content("hello");
        let s = id.as_str();
        assert!(s.starts_with("fact_"));
        let hex = &s["fact_".len()..];
        assert_eq!(hex.len(), 16);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn from_raw_accepts_well_formed_id() {
        let content_id = FactId::from_content("example");
        let parsed = FactId::from_raw(content_id.as_str()).unwrap();
        assert_eq!(content_id, parsed);
    }

    #[test]
    fn from_raw_rejects_missing_prefix() {
        assert!(FactId::from_raw("abc123def456abc0").is_err());
    }

    #[test]
    fn from_raw_rejects_short_hex() {
        assert!(FactId::from_raw("fact_abc").is_err());
    }

    #[test]
    fn from_raw_rejects_long_hex() {
        assert!(FactId::from_raw("fact_abcdef0123456789abcd").is_err());
    }

    #[test]
    fn from_raw_rejects_uppercase_hex() {
        assert!(FactId::from_raw("fact_ABCDEF0123456789").is_err());
    }

    #[test]
    fn from_raw_rejects_non_hex() {
        assert!(FactId::from_raw("fact_zzzzzzzzzzzzzzzz").is_err());
    }

    #[test]
    fn display_returns_raw_string() {
        let id = FactId::from_content("x");
        assert_eq!(id.to_string(), id.as_str());
    }
}
