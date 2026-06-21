//! Topic extractor — flatten the last chat message's content (§3 step 3).
//!
//! Two entry points share the same "what counts as text" knowledge:
//! - [`extract_from_messages`] — the canonical path: takes the typed
//!   [`EnrichmentMessages`] array the request pipeline operates on and
//!   flattens the trailing message's content into a single topic string.
//! - [`extract_from_content`] — kept for backward compatibility with the
//!   JSON-shape fakes in tests that build a single `Value`; new code
//!   should reach for the typed entry point.
//!
//! Null / missing content degrades to an empty string — the caller filters
//! on length.

use serde_json::Value;

use crate::helpers::openai_content::flatten_text;
use crate::types::EnrichmentMessages;

/// Flatten the trailing message's content into a single topic string.
///
/// Mirrors the POC `extract_topic(messages[-1])`: returns the empty string
/// when `messages` is empty so the caller's `min_topic_chars` gate filters
/// it out before embedding.
pub fn extract_from_messages(messages: &EnrichmentMessages) -> String {
    messages
        .last()
        .map(|m| m.content.as_text())
        .unwrap_or_default()
}

/// Flatten a JSON-encoded message content into a single topic string.
///
/// Kept as a thin adapter over [`flatten_text`] so existing tests that
/// build a `serde_json::Value` (instead of the typed DTO) keep working.
/// New call sites should use [`extract_from_messages`].
pub fn extract_from_content(content: &Value) -> String {
    flatten_text(content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessageDto, MessageContent};
    use serde_json::json;

    fn user_msg(content: &str) -> ChatMessageDto {
        ChatMessageDto {
            role: "user".into(),
            content: MessageContent::Text(content.into()),
            tool_calls: None,
        }
    }

    #[test]
    fn extract_from_messages_returns_last_message_text() {
        let msgs: EnrichmentMessages = vec![user_msg("first"), user_msg("hello world")];
        assert_eq!(extract_from_messages(&msgs), "hello world");
    }

    #[test]
    fn extract_from_messages_returns_empty_when_no_messages() {
        let msgs: EnrichmentMessages = Vec::new();
        assert_eq!(extract_from_messages(&msgs), "");
    }

    #[test]
    fn extract_from_messages_flattens_multipart() {
        let msg = ChatMessageDto {
            role: "user".into(),
            content: MessageContent::Multipart(vec![
                crate::types::ContentPart {
                    kind: "text".into(),
                    text: "alpha".into(),
                },
                crate::types::ContentPart {
                    kind: "image_url".into(),
                    text: String::new(),
                },
                crate::types::ContentPart {
                    kind: "text".into(),
                    text: "beta".into(),
                },
            ]),
            tool_calls: None,
        };
        let msgs: EnrichmentMessages = vec![msg];
        assert_eq!(extract_from_messages(&msgs), "alpha beta");
    }

    // ---- legacy JSON entry point (back-compat) ------------------------

    #[test]
    fn plain_string_content_is_returned_verbatim() {
        assert_eq!(extract_from_content(&json!("hello world")), "hello world");
    }

    #[test]
    fn array_of_text_parts_is_joined_with_space() {
        let content = json!([
            { "type": "text", "text": "hello" },
            { "type": "text", "text": "world" }
        ]);
        assert_eq!(extract_from_content(&content), "hello world");
    }

    #[test]
    fn non_text_parts_are_skipped() {
        let content = json!([
            { "type": "image_url", "image_url": "..." },
            { "type": "text", "text": "keep me" }
        ]);
        assert_eq!(extract_from_content(&content), "keep me");
    }

    #[test]
    fn null_content_yields_empty_string() {
        assert_eq!(extract_from_content(&Value::Null), "");
    }

    #[test]
    fn number_content_yields_empty_string() {
        assert_eq!(extract_from_content(&json!(42)), "");
    }
}
