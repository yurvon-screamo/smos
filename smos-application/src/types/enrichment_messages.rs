//! Typed OpenAI-compatible chat-message DTOs used by the enrichment
//! pipeline (§3 + §4 + H-5 refactor).
//!
//! Previously the helpers (`topic_extractor`, `session_marker`,
//! `request_enricher`) operated on raw `serde_json::Value`s — every read of
//! `messages[i].content` re-parsed the JSON, every mutation re-serialised
//! it, and the wire contract lived only in test fixtures. The typed DTOs
//! here give those helpers a single source of truth for the *read-only*
//! shape they care about (role, content, tool_calls).
//!
//! # H-5 — read-only projection contract
//!
//! The typed [`EnrichmentMessages`] array is a **read-only projection** of
//! the wire-shape `Vec<serde_json::Value>` that lives at the HTTP boundary:
//!
//! - `enrichment_messages_from_json` is called once at the top of the
//!   enrichment pipeline to build the projection.
//! - `topic_extractor::extract_from_messages` and
//!   `session_marker::detect_from_typed_messages` consume the projection
//!   for read-only operations (topic extraction, marker scan).
//! - The mutation step (`request_enricher::inject_value`) operates on the
//!   raw `Value` directly so it preserves EVERY per-message field
//!   (`name`, `tool_call_id`, `refusal`, `image_url` parts, audio parts,
//!   future OpenAI extensions). Round-tripping the typed DTO would
//!   silently drop those fields and break the fail-open contract for
//!   tool-calling and vision workflows.
//! - `enrichment_messages_to_json` exists only as a regression guard
//!   (its unit test pins the round-trip for string-content messages,
//!   which the e2e `enrichment_failure_preserves_original_user_message_
//!   verbatim` test exercises end-to-end). It is NOT used in the
//!   production path.
//!
//! The model intentionally covers only the fields the read-side helpers
//! need; per-message extras are dropped at the projection boundary.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One OpenAI-compatible chat message (role + content + optional tool calls).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessageDto {
    /// `system` / `user` / `assistant` / `tool`.
    pub role: String,
    /// Either a plain string or a multipart list of content parts.
    pub content: MessageContent,
    /// Optional OpenAI-style `tool_calls` block (assistant turns only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDto>>,
}

/// `content` field — either a plain string or a multipart array.
///
/// `#[serde(untagged)]` lets the value round-trip either shape without a
/// wrapper key, matching the OpenAI wire format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain string content (`"content": "hello"`).
    Text(String),
    /// Multipart content (`"content": [{"type":"text","text":"..."}, ...]`).
    Multipart(Vec<ContentPart>),
}

impl MessageContent {
    /// Reduce the content to a single space-joined string of every text
    /// part. Used wherever the pipeline needs the searchable / joinable
    /// text representation of a message (topic extraction, marker scan).
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Multipart(parts) => parts
                .iter()
                .filter(|p| p.is_text_kind())
                .map(|p| p.text.clone())
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    /// `true` when the content carries no searchable text (empty string or
    /// a multipart list with no text parts).
    pub fn is_empty(&self) -> bool {
        self.as_text().is_empty()
    }
}

/// One part of a multipart `content` array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentPart {
    /// OpenAI wire key is `type`; renamed here so the field reads as a
    /// normal Rust identifier at the call site.
    #[serde(rename = "type")]
    pub kind: String,
    /// Text payload; present (and meaningful) only when `kind == "text"`.
    /// Other parts (image_url, audio, …) carry their own payload shapes
    /// which the typed model does not need to introspect.
    #[serde(default)]
    pub text: String,
}

impl ContentPart {
    /// `true` when this part carries searchable text.
    pub fn is_text_kind(&self) -> bool {
        self.kind == "text"
    }
}

/// OpenAI-style `tool_calls[]` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallDto {
    /// OpenAI-assigned tool call id (forwarded verbatim).
    #[serde(default)]
    pub id: String,
    /// Always `"function"` in the current OpenAI spec; kept as a string
    /// for forward compatibility.
    #[serde(default, rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

/// `function` block of a `tool_calls[]` entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments string (OpenAI ships this as a string, not
    /// an object). The DTO keeps it verbatim — parsing is the consumer's
    /// concern.
    #[serde(default)]
    pub arguments: String,
}

/// Owned vector of [`ChatMessageDto`] — the typed counterpart of the
/// `Vec<serde_json::Value>` array OpenAI clients send on the wire.
pub type EnrichmentMessages = Vec<ChatMessageDto>;

/// Parse an OpenAI-shaped JSON message array into the typed
/// [`EnrichmentMessages`] form.
///
/// Lenient by design: a message missing `role` defaults to `user`,
/// missing or non-string `content` becomes an empty string, unknown
/// fields are dropped. The point is to never fail — the HTTP layer
/// already validated the request body shape, and a deeper failure
/// here would block enrichment (which is fail-open anyway).
pub fn enrichment_messages_from_json(values: &[Value]) -> EnrichmentMessages {
    values
        .iter()
        .map(|v| {
            let role = v
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .to_string();
            let content = parse_content(v.get("content"));
            let tool_calls = v
                .get("tool_calls")
                .and_then(Value::as_array)
                .map(|arr| arr.iter().filter_map(parse_tool_call).collect());
            ChatMessageDto {
                role,
                content,
                tool_calls,
            }
        })
        .collect()
}

/// Materialise typed messages back into the OpenAI wire shape (one JSON
/// object per message, `role` + `content` keys always present,
/// `tool_calls` only when set).
pub fn enrichment_messages_to_json(messages: &EnrichmentMessages) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let content = match &m.content {
                MessageContent::Text(s) => Value::String(s.clone()),
                MessageContent::Multipart(parts) => Value::Array(
                    parts
                        .iter()
                        .map(|p| serde_json::json!({"type": p.kind, "text": p.text}))
                        .collect(),
                ),
            };
            let mut obj = serde_json::Map::new();
            obj.insert("role".to_string(), Value::String(m.role.clone()));
            obj.insert("content".to_string(), content);
            if let Some(tool_calls) = &m.tool_calls {
                let arr: Vec<Value> = tool_calls
                    .iter()
                    .map(|tc| serde_json::to_value(tc).unwrap_or(Value::Null))
                    .collect();
                obj.insert("tool_calls".to_string(), Value::Array(arr));
            }
            Value::Object(obj)
        })
        .collect()
}

fn parse_content(value: Option<&Value>) -> MessageContent {
    let Some(value) = value else {
        return MessageContent::Text(String::new());
    };
    match value {
        Value::String(s) => MessageContent::Text(s.clone()),
        Value::Array(parts) => MessageContent::Multipart(
            parts
                .iter()
                .filter_map(|p| {
                    let kind = p
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    // Skip non-text parts to keep the typed model focused
                    // on what the enrichment pipeline reads.
                    if kind != "text" {
                        return None;
                    }
                    let text = p
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    Some(ContentPart { kind, text })
                })
                .collect(),
        ),
        _ => MessageContent::Text(String::new()),
    }
}

fn parse_tool_call(v: &Value) -> Option<ToolCallDto> {
    let function = v.get("function")?;
    let id = v
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let kind = v
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("function")
        .to_string();
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let arguments = match function.get("arguments") {
        Some(Value::String(raw)) => raw.clone(),
        Some(other) => serde_json::to_string(other).unwrap_or_else(|_| "null".into()),
        None => String::new(),
    };
    Some(ToolCallDto {
        id,
        kind,
        function: ToolCallFunction { name, arguments },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn from_json_values_parses_string_content() {
        let raw = vec![json!({"role": "user", "content": "hello"})];
        let msgs = enrichment_messages_from_json(&raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content.as_text(), "hello");
    }

    #[test]
    fn from_json_values_parses_multipart_content() {
        let raw = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "alpha"},
                {"type": "image_url", "image_url": "..."},
                {"type": "text", "text": "beta"},
            ]
        })];
        let msgs = enrichment_messages_from_json(&raw);
        assert_eq!(msgs[0].content.as_text(), "alpha beta");
    }

    #[test]
    fn from_json_values_defaults_missing_role_to_user() {
        let raw = vec![json!({"content": "hi"})];
        let msgs = enrichment_messages_from_json(&raw);
        assert_eq!(msgs[0].role, "user");
    }

    #[test]
    fn from_json_values_treats_missing_content_as_empty() {
        let raw = vec![json!({"role": "system"})];
        let msgs = enrichment_messages_from_json(&raw);
        assert!(msgs[0].content.is_empty());
    }

    #[test]
    fn from_json_values_parses_tool_calls() {
        let raw = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}
            }]
        })];
        let msgs = enrichment_messages_from_json(&raw);
        let tc = msgs[0].tool_calls.as_ref().expect("tool_calls parsed");
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "call_1");
        assert_eq!(tc[0].function.name, "read_file");
        assert_eq!(tc[0].function.arguments, r#"{"path":"a"}"#);
    }

    #[test]
    fn to_json_values_round_trips_string_content() {
        let msgs: EnrichmentMessages = vec![ChatMessageDto {
            role: "user".into(),
            content: MessageContent::Text("hi".into()),
            tool_calls: None,
        }];
        let out = enrichment_messages_to_json(&msgs);
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[0]["content"], "hi");
        assert!(out[0].get("tool_calls").is_none());
    }

    #[test]
    fn to_json_values_emits_tool_calls_when_present() {
        let msgs: EnrichmentMessages = vec![ChatMessageDto {
            role: "assistant".into(),
            content: MessageContent::Text(String::new()),
            tool_calls: Some(vec![ToolCallDto {
                id: "call_1".into(),
                kind: "function".into(),
                function: ToolCallFunction {
                    name: "search".into(),
                    arguments: "{\"q\":\"rust\"}".into(),
                },
            }]),
        }];
        let out = enrichment_messages_to_json(&msgs);
        assert_eq!(out[0]["tool_calls"][0]["function"]["name"], "search");
    }

    #[test]
    fn roundtrip_preserves_user_message_verbatim() {
        // Regression guard for the fail-open contract: a string-content
        // user message must survive `enrichment_messages_from_json` →
        // `enrichment_messages_to_json` unchanged so the upstream receives
        // the original bytes when enrichment fails (see `enrichment_
        // failure_preserves_original_user_message_verbatim` e2e test).
        let raw = vec![json!({"role": "user", "content": "regression-sentinel-7c4a8d1e"})];
        let msgs = enrichment_messages_from_json(&raw);
        let back = enrichment_messages_to_json(&msgs);
        assert_eq!(back[0]["content"], "regression-sentinel-7c4a8d1e");
        assert_eq!(back[0]["role"], "user");
    }

    #[test]
    fn message_content_as_text_handles_empty_multipart() {
        let content = MessageContent::Multipart(Vec::new());
        assert!(content.as_text().is_empty());
        assert!(content.is_empty());
    }

    #[test]
    fn message_content_as_text_skips_non_text_parts() {
        let content = MessageContent::Multipart(vec![
            ContentPart {
                kind: "image_url".into(),
                text: String::new(),
            },
            ContentPart {
                kind: "text".into(),
                text: "kept".into(),
            },
        ]);
        assert_eq!(content.as_text(), "kept");
    }
}
