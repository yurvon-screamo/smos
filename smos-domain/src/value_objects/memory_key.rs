//! `MemoryKey` — path-traversal-safe namespace identifier.
//!
//! The key is also a directory name on disk (§6 storage layout) and a ChromaDB
//! collection name, so it must reject anything that could escape its directory:
//! no `/`, no `\`, no `..`, no leading dots. The validation rules below mirror
//! what the adapter layer uses when reading/writing markdown files.

use crate::error::DomainError;
use serde::{Deserialize, Serialize};

/// A safe namespace for memories, parsed from the model prefix via the
/// application-layer `parse_model` helper (`"origa:gpt-4o" → MemoryKey("origa")`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryKey(String);

impl MemoryKey {
    /// Validate and wrap a raw string.
    ///
    /// Rules: non-empty, first char alphanumeric, remaining chars in
    /// `[A-Za-z0-9_.-]`, no path separators, no `..`, no leading dot.
    pub fn from_raw(s: &str) -> Result<Self, DomainError> {
        if is_safe_memory_key(s) {
            Ok(Self(s.to_string()))
        } else {
            Err(DomainError::UnsafeMemoryKey(s.to_string()))
        }
    }

    /// Default namespace used when the model name carries no prefix.
    pub fn shared() -> Self {
        Self("shared".to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

fn is_safe_memory_key(s: &str) -> bool {
    if s.is_empty() || s.len() > 64 {
        return false;
    }
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_alphanumeric() {
        return false;
    }
    // Reject path-traversal sequences and path separators. `..` (as substring
    // or whole value) covers the parent-directory attack; explicit `== ".."`
    // is unnecessary because it is subsumed by `contains("..")`.
    if s.contains("..") || s.contains('/') || s.contains('\\') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

impl std::fmt::Display for MemoryKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DomainError;

    #[test]
    fn accepts_simple_alphanumeric() {
        assert!(MemoryKey::from_raw("origa").is_ok());
    }

    #[test]
    fn accepts_shared_keyword() {
        assert_eq!(MemoryKey::from_raw("shared").unwrap().as_str(), "shared");
    }

    #[test]
    fn accepts_dotted_and_dashed_names() {
        assert!(MemoryKey::from_raw("my-project_v2").is_ok());
        assert!(MemoryKey::from_raw("analog.finder").is_ok());
        assert!(MemoryKey::from_raw("a1-b2.c3").is_ok());
    }

    #[test]
    fn rejects_empty() {
        assert!(matches!(
            MemoryKey::from_raw(""),
            Err(DomainError::UnsafeMemoryKey(_))
        ));
    }

    #[test]
    fn rejects_dot_dot() {
        assert!(MemoryKey::from_raw("..").is_err());
    }

    #[test]
    fn rejects_path_traversal_with_slash() {
        assert!(MemoryKey::from_raw("a/b").is_err());
        assert!(MemoryKey::from_raw("/etc/passwd").is_err());
    }

    #[test]
    fn rejects_path_traversal_with_backslash() {
        assert!(MemoryKey::from_raw("a\\b").is_err());
    }

    #[test]
    fn rejects_embedded_dot_dot() {
        assert!(MemoryKey::from_raw("a..b").is_err());
    }

    #[test]
    fn rejects_leading_dot() {
        assert!(MemoryKey::from_raw(".hidden").is_err());
    }

    #[test]
    fn rejects_leading_dash() {
        assert!(MemoryKey::from_raw("-dash").is_err());
    }

    #[test]
    fn rejects_spaces_and_special() {
        assert!(MemoryKey::from_raw("hello world").is_err());
        assert!(MemoryKey::from_raw("origa!").is_err());
    }

    #[test]
    fn shared_default_is_canonical() {
        assert_eq!(MemoryKey::shared().as_str(), "shared");
    }

    #[test]
    fn display_matches_as_str() {
        let key = MemoryKey::from_raw("origa").unwrap();
        assert_eq!(key.to_string(), "origa");
    }
}
