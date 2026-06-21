//! `FactContent` — non-empty canonical English fact text.

use crate::error::DomainError;
use serde::{Deserialize, Serialize};

/// The text body of a fact.
///
/// Trim-empty content is rejected because it would produce a degenerate id and
/// carries no extractable signal. Leading/trailing whitespace is preserved
/// verbatim — the extractor's job is to emit canonical English; the value
/// object's job is to refuse empty payloads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FactContent(String);

impl FactContent {
    pub fn new(s: String) -> Result<Self, DomainError> {
        if s.trim().is_empty() {
            return Err(DomainError::EmptyFactContent);
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Normalized text for comparison: lowercase + collapse whitespace runs.
    ///
    /// Used by the exact-match NLI short-circuit so trivially identical pairs
    /// (modulo casing or stray whitespace) never reach the model: they are
    /// entailment by definition.
    pub fn normalized(&self) -> String {
        Self::normalize_text(&self.0)
    }

    /// Static text normaliser: lowercase + collapse any whitespace run.
    ///
    /// Exposed so callers that only hold a `&str` (e.g. an existing fact's
    /// content) can normalise without first wrapping into a `FactContent`.
    pub fn normalize_text(s: &str) -> String {
        s.trim()
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// `true` if `self` and `other` are equal after normalisation.
    pub fn equals_normalized(&self, other: &FactContent) -> bool {
        self.normalized() == other.normalized()
    }

    /// `true` if `a` and `b` (raw `&str`) are equal after normalisation.
    pub fn text_equals_normalized(a: &str, b: &str) -> bool {
        Self::normalize_text(a) == Self::normalize_text(b)
    }
}

impl std::fmt::Display for FactContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DomainError;

    #[test]
    fn new_accepts_non_empty_text() {
        let c = FactContent::new("hello".to_string()).unwrap();
        assert_eq!(c.as_str(), "hello");
    }

    #[test]
    fn new_rejects_empty() {
        assert!(matches!(
            FactContent::new(String::new()),
            Err(DomainError::EmptyFactContent)
        ));
    }

    #[test]
    fn new_rejects_whitespace_only() {
        assert!(matches!(
            FactContent::new("   \n\t ".to_string()),
            Err(DomainError::EmptyFactContent)
        ));
    }

    #[test]
    fn display_returns_inner() {
        let c = FactContent::new("fact text".to_string()).unwrap();
        assert_eq!(c.to_string(), "fact text");
    }

    #[test]
    fn normalized_lowercases_and_collapses_whitespace() {
        let c = FactContent::new("  HeLLo   World\n\t!  ".to_string()).unwrap();
        assert_eq!(c.normalized(), "hello world !");
    }

    #[test]
    fn normalized_trims_leading_and_trailing() {
        let c = FactContent::new("   spaced   ".to_string()).unwrap();
        assert_eq!(c.normalized(), "spaced");
    }

    #[test]
    fn equals_normalized_true_for_same_text_different_case() {
        let a = FactContent::new("Hello World".to_string()).unwrap();
        let b = FactContent::new("hello world".to_string()).unwrap();
        assert!(a.equals_normalized(&b));
    }

    #[test]
    fn equals_normalized_true_for_same_text_different_whitespace() {
        let a = FactContent::new("Hello   World".to_string()).unwrap();
        let b = FactContent::new("hello world".to_string()).unwrap();
        assert!(a.equals_normalized(&b));
    }

    #[test]
    fn equals_normalized_false_for_different_text() {
        let a = FactContent::new("TTL=60".to_string()).unwrap();
        let b = FactContent::new("TTL=10".to_string()).unwrap();
        assert!(!a.equals_normalized(&b));
    }
}
