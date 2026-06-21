//! OpenAI-compatible chat-completion response envelope.
//!
//! Two flavours coexist on the same upstream:
//! - `NonStreaming` is the full buffered JSON response (callers parse it).
//! - `Streaming` is an opaque byte stream (callers pass it through as SSE).
//!
//! We deliberately do *not* model the JSON shape: the proxy forwards it
//! verbatim, and OpenAI may evolve it independently of this crate.

use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::UpstreamError;

/// Chat-completion response — buffered JSON or byte stream.
///
/// A manual `Debug` impl is required because `dyn Stream` has no `Debug`;
/// the streaming arm is rendered with a placeholder that still identifies
/// the variant.
pub enum ChatResponse {
    /// Full buffered body for non-streaming calls.
    NonStreaming(Value),

    /// Raw byte stream for streaming calls. The boxed trait object keeps the
    /// enum cheap to move and decouples callers from the concrete HTTP body
    /// type produced by the adapter.
    Streaming(Box<dyn Stream<Item = Result<Bytes, UpstreamError>> + Send + Unpin>),
}

impl std::fmt::Debug for ChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatResponse::NonStreaming(v) => f.debug_tuple("NonStreaming").field(v).finish(),
            ChatResponse::Streaming(_) => {
                f.debug_tuple("Streaming").field(&"<byte stream>").finish()
            }
        }
    }
}

// `NonStreaming` arm implements `Serialize` so callers that need to forward
// the response can do so without re-matching. The streaming arm is not
// serialisable by definition; we expose a helper instead.
impl Serialize for ChatResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ChatResponse::NonStreaming(v) => v.serialize(serializer),
            ChatResponse::Streaming(_) => Err(serde::ser::Error::custom(
                "cannot serialize a streaming ChatResponse; drain the stream first",
            )),
        }
    }
}

// `NonStreaming` arm deserialises directly; there is no wire representation
// for the streaming arm, so we only support the buffered shape here.
impl<'de> Deserialize<'de> for ChatResponse {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Value::deserialize(deserializer).map(ChatResponse::NonStreaming)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use serde_json::json;

    #[test]
    fn non_streaming_serialises_inner_value() {
        let resp = ChatResponse::NonStreaming(json!({"choices": []}));
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("choices"));
    }

    #[test]
    fn streaming_arm_is_not_serialisable() {
        let stream: Box<dyn Stream<Item = Result<Bytes, UpstreamError>> + Send + Unpin> =
            Box::new(stream::iter(vec![Ok(Bytes::from_static(b"chunk"))]));
        let resp = ChatResponse::Streaming(stream);
        let result = serde_json::to_string(&resp);
        assert!(result.is_err());
    }

    #[test]
    fn non_streaming_roundtrips_through_serde() {
        let resp = ChatResponse::NonStreaming(json!({"a": 1}));
        let json_str = serde_json::to_string(&resp).unwrap();
        let back: ChatResponse = serde_json::from_str(&json_str).unwrap();
        match back {
            ChatResponse::NonStreaming(v) => assert_eq!(v["a"], 1),
            ChatResponse::Streaming(_) => panic!("expected NonStreaming"),
        }
    }
}
