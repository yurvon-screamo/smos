//! `parse_model` — split `"memory_key:real_model"` into a safe key and model id.
//!
//! No prefix → `MemoryKey::shared()`. Whitespace around the prefix and the
//! model id is trimmed. Only the first `:` splits, so model ids that contain a
//! colon keep the rest intact. Unsafe prefixes are rejected up-front.

use smos_domain::{DomainError, MemoryKey};

/// Split a model string into `(memory_key, upstream_model)`.
pub fn parse_model(model: &str) -> Result<(MemoryKey, String), DomainError> {
    let Some((key, real)) = model.split_once(':') else {
        return Ok((MemoryKey::shared(), model.trim().to_string()));
    };
    let key = key.trim();
    let real = real.trim();
    if real.is_empty() {
        return Ok((MemoryKey::shared(), String::new()));
    }
    let memory_key = MemoryKey::from_raw(key)?;
    Ok((memory_key, real.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use smos_domain::DomainError;

    #[test]
    fn parses_prefix_and_model() {
        let (key, model) = parse_model("origa:gpt-4o").unwrap();
        assert_eq!(key.as_str(), "origa");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn no_colon_returns_shared_key() {
        let (key, model) = parse_model("gpt-4o").unwrap();
        assert_eq!(key.as_str(), "shared");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn trims_whitespace_around_prefix_and_model() {
        let (key, model) = parse_model("  origa : gpt-4o  ").unwrap();
        assert_eq!(key.as_str(), "origa");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn only_first_colon_splits() {
        let (key, model) = parse_model("origa:openrouter:gpt-4o").unwrap();
        assert_eq!(key.as_str(), "origa");
        assert_eq!(model, "openrouter:gpt-4o");
    }

    #[test]
    fn empty_string_falls_back_to_shared() {
        let (key, model) = parse_model("").unwrap();
        assert_eq!(key.as_str(), "shared");
        assert_eq!(model, "");
    }

    #[test]
    fn colon_only_falls_back_to_shared() {
        let (key, model) = parse_model(":").unwrap();
        assert_eq!(key.as_str(), "shared");
        assert_eq!(model, "");
    }

    #[test]
    fn unsafe_prefix_is_rejected() {
        assert!(matches!(
            parse_model("../etc:gpt-4o"),
            Err(DomainError::UnsafeMemoryKey(_))
        ));
    }

    #[test]
    fn prefix_with_slash_is_rejected() {
        assert!(parse_model("a/b:gpt-4o").is_err());
    }
}
