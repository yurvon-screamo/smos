//! OpenAI message content flattener — shared wire-format helper.
//!
//! OpenAI chat messages can carry `content` either as a plain string or as a
//! multipart list of `{"type": "text"|"image_url"|..., ...}` parts. Several
//! helpers (session-marker detection, request enrichment, topic extraction)
//! need to reduce that shape to a single searchable / joinable string. The
//! knowledge of "what counts as text" lives here once instead of being
//! duplicated across the three call sites.

use serde_json::Value;

/// Reduce an OpenAI `content` field to a single space-joined string of all
/// `{"type": "text", "text": ...}` parts.
///
/// - String content → returned verbatim.
/// - Array of `{type: "text", text: "..."}` parts → joined with `" "`.
/// - Anything else (null, number, object, missing) → empty string.
///
/// Non-text parts (image_url, audio, …) contribute nothing.
pub fn flatten_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p {
                Value::Object(map) => {
                    let is_text = map
                        .get("type")
                        .and_then(Value::as_str)
                        .map(|t| t == "text")
                        .unwrap_or(false);
                    if is_text {
                        map.get("text").and_then(Value::as_str).map(str::to_string)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn string_content_returned_verbatim() {
        assert_eq!(flatten_text(&json!("hello world")), "hello world");
    }

    #[test]
    fn text_parts_joined_with_space() {
        let content = json!([
            { "type": "text", "text": "hello" },
            { "type": "text", "text": "world" }
        ]);
        assert_eq!(flatten_text(&content), "hello world");
    }

    #[test]
    fn non_text_parts_skipped() {
        let content = json!([
            { "type": "image_url", "image_url": "..." },
            { "type": "text", "text": "keep me" }
        ]);
        assert_eq!(flatten_text(&content), "keep me");
    }

    #[test]
    fn null_content_yields_empty_string() {
        assert_eq!(flatten_text(&Value::Null), "");
    }

    #[test]
    fn number_content_yields_empty_string() {
        assert_eq!(flatten_text(&json!(42)), "");
    }

    #[test]
    fn empty_array_yields_empty_string() {
        assert_eq!(flatten_text(&json!([])), "");
    }
}
