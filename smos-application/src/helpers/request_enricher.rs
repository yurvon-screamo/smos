//! Request enricher — prepend the memory block to `messages[0].content` (§3
//! step 8).
//!
//! Two entry points share the "prepend block to first message" logic:
//! - [`inject_value`] — the production path: operates on the raw
//!   `serde_json::Value` so EVERY per-message field (`name`,
//!   `tool_call_id`, `refusal`, image_url parts, audio parts, future
//!   OpenAI extensions) survives the enrichment mutation. The request
//!   pipeline (`EnrichRequest::execute`) uses this entry point.
//! - [`inject`] — typed counterpart for tests that build an
//!   `EnrichmentMessages` directly. Mutates in-place on the typed model,
//!   which is convenient for unit tests but loses per-message extras the
//!   typed DTO does not model. NOT used in the production path.
//!
//! Behaviour mirrors the POC `inject_into_messages`:
//! - Empty `messages` → inject a synthetic `{role: "system", content: block}`.
//! - String `messages[0].content` → prepend `block\n\n<old content>`.
//! - Array `messages[0].content` → flatten to text then prepend (the upstream
//!   adapter rebuilds multipart if needed).
//! - Empty block → returns the input unchanged (fail-open).

use serde_json::{Map, Value};

use crate::helpers::openai_content::flatten_text;
use crate::types::{ChatMessageDto, EnrichmentMessages, MessageContent};

/// Prepend `block` to the first typed message's content (in-place).
///
/// - Empty `messages` → push a fresh synthetic `system` message.
/// - Empty `block` → no-op (fail-open).
/// - Otherwise: take the first message's content as text, prepend
///   `block\n\n<old>` (or just `block` when the original was empty),
///   and replace the content with the combined string.
pub fn inject(messages: &mut EnrichmentMessages, block: &str) {
    if block.is_empty() {
        return;
    }
    if messages.is_empty() {
        messages.push(ChatMessageDto {
            role: "system".into(),
            content: MessageContent::Text(block.to_string()),
            tool_calls: None,
        });
        return;
    }
    let first = &mut messages[0];
    let original = first.content.as_text();
    let combined = if original.is_empty() {
        block.to_string()
    } else {
        format!("{block}\n\n{original}")
    };
    first.content = MessageContent::Text(combined);
}

/// Legacy JSON entry point — kept for adapter callers that hold a raw
/// `serde_json::Value` (e.g. non-streaming response injection paths that
/// never go through the typed pipeline).
pub fn inject_value(messages: &Value, block: &str) -> Value {
    if block.is_empty() {
        return messages.clone();
    }
    let Some(arr) = messages.as_array() else {
        return messages.clone();
    };
    if arr.is_empty() {
        return Value::Array(vec![new_system_message(block)]);
    }
    let mut enriched: Vec<Value> = arr.clone();
    let mut first = enriched[0].clone();
    let original_text = first.get("content").map(flatten_text).unwrap_or_default();
    let combined = if original_text.is_empty() {
        block.to_string()
    } else {
        format!("{block}\n\n{original_text}")
    };
    set_content(&mut first, combined);
    enriched[0] = first;
    Value::Array(enriched)
}

fn new_system_message(block: &str) -> Value {
    let mut map = Map::new();
    map.insert("role".to_string(), Value::String("system".to_string()));
    map.insert("content".to_string(), Value::String(block.to_string()));
    Value::Object(map)
}

fn set_content(message: &mut Value, value: String) {
    if let Value::Object(map) = message {
        map.insert("content".to_string(), Value::String(value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn user_msg(content: &str) -> ChatMessageDto {
        ChatMessageDto {
            role: "user".into(),
            content: MessageContent::Text(content.into()),
            tool_calls: None,
        }
    }

    #[test]
    fn inject_empty_messages_creates_synthetic_system_message() {
        let mut msgs: EnrichmentMessages = Vec::new();
        inject(&mut msgs, "block");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content.as_text(), "block");
    }

    #[test]
    fn inject_prepends_block_to_existing_string_content() {
        let mut msgs: EnrichmentMessages = vec![user_msg("hello")];
        inject(&mut msgs, "BLOCK");
        assert_eq!(msgs[0].content.as_text(), "BLOCK\n\nhello");
    }

    #[test]
    fn inject_prepends_block_when_existing_content_is_empty() {
        let mut msgs: EnrichmentMessages = vec![user_msg("")];
        inject(&mut msgs, "BLOCK");
        assert_eq!(msgs[0].content.as_text(), "BLOCK");
    }

    #[test]
    fn inject_flattens_multipart_before_prepend() {
        let msg = ChatMessageDto {
            role: "user".into(),
            content: MessageContent::Multipart(vec![
                crate::types::ContentPart {
                    kind: "text".into(),
                    text: "alpha".into(),
                },
                crate::types::ContentPart {
                    kind: "text".into(),
                    text: "beta".into(),
                },
            ]),
            tool_calls: None,
        };
        let mut msgs: EnrichmentMessages = vec![msg];
        inject(&mut msgs, "BLOCK");
        assert_eq!(msgs[0].content.as_text(), "BLOCK\n\nalpha beta");
    }

    #[test]
    fn inject_empty_block_is_noop() {
        let mut msgs: EnrichmentMessages = vec![user_msg("hi")];
        inject(&mut msgs, "");
        assert_eq!(msgs[0].content.as_text(), "hi");
    }

    #[test]
    fn inject_preserves_remaining_messages() {
        let mut msgs: EnrichmentMessages = vec![user_msg("first"), user_msg("second")];
        inject(&mut msgs, "B");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].content.as_text(), "second");
    }

    // ---- legacy JSON entry point --------------------------------------

    #[test]
    fn inject_value_empty_messages_creates_synthetic_system_message() {
        let out = inject_value(&json!([]), "block");
        let first = &out.as_array().unwrap()[0];
        assert_eq!(first.get("role").and_then(Value::as_str), Some("system"));
        assert_eq!(first.get("content").and_then(Value::as_str), Some("block"));
    }

    #[test]
    fn inject_value_prepends_block_to_existing_string_content() {
        let messages = json!([{ "role": "user", "content": "hello" }]);
        let out = inject_value(&messages, "BLOCK");
        assert_eq!(
            out.as_array().unwrap()[0]
                .get("content")
                .and_then(Value::as_str),
            Some("BLOCK\n\nhello")
        );
    }

    #[test]
    fn inject_value_empty_block_returns_input_unchanged() {
        let messages = json!([{ "role": "user", "content": "hi" }]);
        let out = inject_value(&messages, "");
        assert_eq!(out, messages);
    }
}
