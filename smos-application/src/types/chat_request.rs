//! OpenAI-compatible chat-completion request envelope.
//!
//! The shape preserves *all* upstream fields (extra): the proxy must forward
//! unknown fields verbatim so future OpenAI parameters (e.g. `reasoning_effort`,
//! `response_format`) pass through without a release of this crate. `model`
//! and `messages` are typed for the use cases that need to read or modify
//! them; everything else falls through `extra`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Wire-shape chat-completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,

    /// Raw message bodies — kept as `serde_json::Value` so multipart content
    /// (image_url, audio, tool calls with arbitrary payloads) round-trips
    /// without bespoke enums.
    pub messages: Vec<Value>,

    /// Catch-all for every other OpenAI parameter (`temperature`, `stream`,
    /// `tools`, …). `#[serde(flatten)]` makes them inline peers of `model`
    /// and `messages` on the wire.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatRequest {
    /// Build a request with `model`, `messages`, and no extras.
    pub fn new(model: impl Into<String>, messages: Vec<Value>) -> Self {
        Self {
            model: model.into(),
            messages,
            extra: Map::new(),
        }
    }

    /// Insert (or replace) one extra parameter. Builder-style.
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Read one extra parameter by key.
    pub fn extra(&self, key: &str) -> Option<&Value> {
        self.extra.get(key)
    }

    /// `true` iff the request asks for streaming (`stream: true` in extras).
    pub fn is_streaming(&self) -> bool {
        self.extra
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serialises_known_fields_at_top_level() {
        let req = ChatRequest::new("gpt-4o", vec![json!({"role": "user", "content": "hi"})]);
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "gpt-4o");
        assert_eq!(v["messages"][0]["role"], "user");
    }

    #[test]
    fn extra_fields_flatten_alongside_known_fields() {
        let req = ChatRequest::new("m", vec![]).with_extra("temperature", json!(0.7));
        let v: Value = serde_json::to_value(&req).unwrap();
        assert_eq!(v["temperature"], 0.7);
        assert!(v.get("extra").is_none());
    }

    #[test]
    fn deserialises_unknown_fields_into_extra() {
        let raw = serde_json::json!({
            "model": "m",
            "messages": [],
            "temperature": 0.3,
            "tools": [{"type": "function"}],
        });
        let req: ChatRequest = serde_json::from_value(raw).unwrap();
        assert_eq!(req.model, "m");
        assert_eq!(req.extra("temperature"), Some(&json!(0.3)));
        assert!(req.extra("tools").is_some());
    }

    #[test]
    fn roundtrip_preserves_all_fields() {
        let req = ChatRequest::new("m", vec![json!({"role": "system"})])
            .with_extra("stream", json!(true))
            .with_extra("max_tokens", json!(128));
        let json = serde_json::to_string(&req).unwrap();
        let back: ChatRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "m");
        assert_eq!(back.extra("stream"), Some(&json!(true)));
        assert_eq!(back.extra("max_tokens"), Some(&json!(128)));
    }

    #[test]
    fn is_streaming_reads_extra_bool() {
        let streaming = ChatRequest::new("m", vec![]).with_extra("stream", json!(true));
        let non_streaming = ChatRequest::new("m", vec![]).with_extra("stream", json!(false));
        let unset = ChatRequest::new("m", vec![]);
        assert!(streaming.is_streaming());
        assert!(!non_streaming.is_streaming());
        assert!(!unset.is_streaming());
    }
}
