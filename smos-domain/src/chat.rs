//! Pure chat message types.
//!
//! `ChatRequest` lives in the application layer (slice 3): it carries the full
//! OpenAI-compatible envelope. The domain layer only needs the message and
//! tool-call shapes so it can reason about content manipulation (enrichment,
//! marker detection, topic extraction).

use serde::{Deserialize, Serialize};

/// Tool invocation arguments — opaque at the domain layer.
///
/// The domain stores the raw payload without interpreting it: it has no
/// `serde_json` dependency (the layering invariant pinned by
/// `smos-domain/Cargo.toml`), and the only operations it performs on
/// arguments are pass-through (forwarding the call to the upstream) or
/// display (rendering into the extraction prompt). Adapters are the
/// boundary that converts between this opaque string and a parsed
/// `serde_json::Value`.
///
/// The wrapped string is expected to be JSON on the wire (OpenAI's
/// `function.arguments` arrives as a JSON-encoded string), but the domain
/// never asserts that — it is content-agnostic. Construction is via
/// [`ToolArguments::from_json`]; access is via [`ToolArguments::as_str`]
/// (borrow) or [`ToolArguments::into_string`] (owned).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolArguments(String);

impl ToolArguments {
    /// Wrap a raw JSON-shaped payload string. The caller decides whether
    /// the input has already been parsed (adapters passing structured
    /// data through) or is still the wire string (adapters passing the
    /// OpenAI `function.arguments` value through verbatim).
    pub fn from_json(json: impl Into<String>) -> Self {
        Self(json.into())
    }

    /// Borrow the underlying payload string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume into the underlying payload string.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl std::fmt::Display for ToolArguments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A single tool invocation reported by the assistant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    /// Tool-specific payload — kept opaque at the domain layer so the
    /// crate stays free of a `serde_json` dependency. Adapters parse the
    /// string into a `serde_json::Value` (or any other shape) at the
    /// boundary.
    pub arguments: ToolArguments,
}

/// One turn in a chat conversation.
///
/// `tool_calls` is omitted from the wire format when empty so round-tripped
/// messages stay close to the OpenAI shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_message_has_user_role() {
        let m = ChatMessage::user("hello");
        assert_eq!(m.role, "user");
        assert_eq!(m.content, "hello");
        assert!(m.tool_calls.is_empty());
    }

    #[test]
    fn serde_roundtrip_preserves_simple_message() {
        let m = ChatMessage::system("be helpful");
        let json = serde_json::to_string(&m).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn serde_omits_empty_tool_calls() {
        let m = ChatMessage::assistant("ok");
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("tool_calls"));
    }

    #[test]
    fn serde_preserves_tool_calls_when_present() {
        let m = ChatMessage {
            role: "assistant".to_string(),
            content: "running it".to_string(),
            tool_calls: vec![ToolCall {
                name: "bash".to_string(),
                arguments: ToolArguments::from_json(r#"{"cmd":"ls"}"#),
            }],
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
        assert_eq!(back.tool_calls.len(), 1);
    }

    #[test]
    fn serde_default_fills_missing_tool_calls() {
        let json = r#"{"role":"user","content":"hi"}"#;
        let back: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(back.role, "user");
        assert!(back.tool_calls.is_empty());
    }

    #[test]
    fn tool_arguments_round_trip_through_serde() {
        let args = ToolArguments::from_json(r#"{"k":"v","n":3}"#);
        let json = serde_json::to_string(&args).unwrap();
        let back: ToolArguments = serde_json::from_str(&json).unwrap();
        assert_eq!(args, back);
        assert_eq!(args.as_str(), r#"{"k":"v","n":3}"#);
        assert_eq!(args.into_string(), r#"{"k":"v","n":3}"#);
    }

    #[test]
    fn tool_arguments_display_renders_raw_payload() {
        let args = ToolArguments::from_json(r#"{"x":1}"#);
        assert_eq!(args.to_string(), r#"{"x":1}"#);
    }
}
