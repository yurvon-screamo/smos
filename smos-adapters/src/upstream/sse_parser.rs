//! SSE framing + session-marker injection helpers (§4).
//!
//! The streaming passthrough path receives raw bytes from the upstream
//! (`reqwest::bytes_stream`). Those bytes do NOT align with SSE event
//! boundaries — a single chunk may contain half an event, two events, or a
//! partial `[DONE]` — AND they do NOT align with UTF-8 character boundaries
//! (a Cyrillic/CJK/emoji codepoint split across two TCP chunks would be
//! corrupted by a stateless `from_utf8_lossy`). [`SseParser`] buffers raw
//! bytes, splits on the `\n\n` terminator (which can never appear inside a
//! UTF-8 multibyte sequence, so the split is charset-safe), and decodes only
//! fully-framed events — exactly like the Python POC's `_iter_sse_events`.
//!
//! The marker helpers (`inject_marker`, `inject_marker_non_streaming`)
//! implement §4 step 2: append `\n<!-- smos:sess_xxx -->` to the terminal
//! chunk's `delta.content` (streaming) or `message.content` (non-streaming).

use serde_json::Value;

/// One parsed SSE event: the payload after `data: ` and whether it is the
/// terminal `[DONE]` sentinel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub data: String,
    pub is_done: bool,
}

impl SseEvent {
    /// Try to parse the payload as JSON. Returns `None` for `[DONE]` or
    /// non-JSON frames (comments, keep-alives).
    pub fn parsed_json(&self) -> Option<Value> {
        if self.is_done {
            return None;
        }
        serde_json::from_str(&self.data).ok()
    }

    /// `true` when this event is a terminal chunk: `choices[0].finish_reason`
    /// is present and non-null. OpenAI emits `"stop"`, `"tool_calls"`,
    /// `"length"`, or `"content_filter"` on the last chunk of a completion —
    /// any of them means the session marker should land here.
    pub fn is_terminal(&self) -> bool {
        let Some(json) = self.parsed_json() else {
            return false;
        };
        json.pointer("/choices/0/finish_reason")
            .and_then(Value::as_str)
            .is_some()
    }
}

/// Defense-in-depth cap: a single SSE frame larger than this is treated as a
/// broken/malicious upstream and the buffer is reset so memory cannot grow
/// without bound. 16 MiB is far above any realistic OpenAI delta.
const MAX_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// Stateful SSE parser that buffers raw bytes across chunk boundaries and
/// decodes only fully-framed events, avoiding UTF-8 corruption on split
/// multibyte codepoints.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a raw byte chunk and return every fully-framed SSE event it
    /// completes. Incomplete trailing bytes stay buffered for the next feed.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() > MAX_BUFFER_BYTES {
            tracing::warn!(
                size = self.buffer.len(),
                "SSE buffer exceeded {MAX_BUFFER_BYTES} bytes; resetting (malformed upstream?)"
            );
            self.buffer.clear();
            return Vec::new();
        }
        self.drain_complete_events()
    }

    /// Flush any remaining buffered bytes as a single final event. Used when
    /// the upstream closes the stream without a trailing `\n\n` (rare but
    /// possible). Returns `None` when the buffer holds only whitespace.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if self.buffer.is_empty() {
            return None;
        }
        let remaining = std::mem::take(&mut self.buffer);
        let lossy = String::from_utf8_lossy(&remaining);
        let trimmed = lossy.trim();
        if trimmed.is_empty() {
            return None;
        }
        parse_event_block(trimmed)
    }

    fn drain_complete_events(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        while let Some(idx) = find_frame_terminator(&self.buffer) {
            // Consume the frame including its trailing `\n\n` terminator.
            let end = idx + 2;
            let body_bytes: Vec<u8> = self.buffer.drain(..end).collect();
            // The frame body is the bytes before `\n\n`; it is a complete
            // UTF-8 sequence (we only split on `\n\n`, which never sits inside
            // a multibyte codepoint), so lossy decoding is exact for valid
            // upstreams and degrades gracefully for malformed ones.
            let body = String::from_utf8_lossy(&body_bytes[..body_bytes.len() - 2]);
            if let Some(event) = parse_event_block(&body) {
                out.push(event);
            }
        }
        out
    }
}

/// Find the byte index where the first `\n\n` (two consecutive 0x0A bytes)
/// starts in `buf`. Returns `None` when no complete frame terminator is
/// present yet. `\n` never appears inside a UTF-8 multibyte sequence, so this
/// byte-level search is charset-safe.
fn find_frame_terminator(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

/// Parse one SSE frame body (the text between two `\n\n` separators) into an
/// event. Multiple `data:` lines are joined with `\n` per the SSE spec; the
/// `[DONE]` sentinel sets `is_done`. Non-`data:` fields (`event:`, `id:`,
/// `retry:`) are intentionally not propagated — the OpenAI `/v1/chat/
/// completions` SSE shape uses only `data:` frames, so this matches every
/// upstream SMOS targets.
fn parse_event_block(body: &str) -> Option<SseEvent> {
    let mut data_parts: Vec<&str> = Vec::new();
    for raw_line in body.lines() {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if let Some(rest) = line.strip_prefix("data:") {
            data_parts.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data_parts.is_empty() {
        return None;
    }
    let data = data_parts.join("\n");
    Some(SseEvent {
        is_done: data.trim() == "[DONE]",
        data,
    })
}

/// Append the session marker to the `content` field of the JSON container at
/// `pointer`. The container is `delta` for streaming and `message` for the
/// buffered JSON response. String content is extended in place; missing or
/// non-string `content` is set to the marker verbatim.
fn append_marker_to_content(mut json: Value, pointer: &str, marker: &str) -> Value {
    if let Some(container) = json.pointer_mut(pointer)
        && let Some(obj) = container.as_object_mut()
    {
        match obj.get_mut("content") {
            Some(Value::String(s)) => s.push_str(marker),
            Some(slot) => *slot = Value::String(marker.to_string()),
            None => {
                obj.insert("content".to_string(), Value::String(marker.to_string()));
            }
        }
    }
    json
}

/// Inject the marker into a streaming event's `choices[0].delta.content` and
/// return the re-serialised JSON payload. Falls back to the `Value` Display
/// form (infallible) if `serde_json::to_string` fails — this never happens for
/// JSON-sourced `Value`s, but the guard keeps the streaming hot path
/// panic-free.
pub fn inject_marker(event_json: Value, marker: &str) -> String {
    let injected = append_marker_to_content(event_json, "/choices/0/delta", marker);
    serde_json::to_string(&injected).unwrap_or_else(|_| injected.to_string())
}

/// Inject the marker into a non-streaming JSON response's
/// `choices[0].message.content` and return the mutated value.
pub fn inject_marker_non_streaming(json: Value, marker: &str) -> Value {
    append_marker_to_content(json, "/choices/0/message", marker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stop_event() -> Value {
        json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        })
    }

    #[test]
    fn feed_emits_one_event_per_complete_frame() {
        let mut p = SseParser::new();
        let events = p.feed(b"data: {\"a\":1}\n\ndata: {\"b\":2}\n\n");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "{\"a\":1}");
        assert_eq!(events[1].data, "{\"b\":2}");
    }

    #[test]
    fn feed_buffers_partial_frame_until_completed() {
        let mut p = SseParser::new();
        assert!(p.feed(b"data: {\"par").is_empty());
        let events = p.feed(b"t\":true}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"part\":true}");
    }

    #[test]
    fn feed_preserves_multibyte_chars_split_across_chunks() {
        // "Привет" = 0xD0 0x9F ... ; split the first 2-byte codepoint between
        // two chunks. A naive from_utf8_lossy would corrupt it; byte buffering
        // must reconstruct it cleanly.
        let cyrillic = "data: {\"c\":\"Привет\"}\n\n";
        let bytes = cyrillic.as_bytes();
        let split_at = bytes.len() - 5;
        let mut p = SseParser::new();
        assert!(p.feed(&bytes[..split_at]).is_empty());
        let events = p.feed(&bytes[split_at..]);
        assert_eq!(events.len(), 1);
        let v: Value = serde_json::from_str(&events[0].data).unwrap();
        assert_eq!(v["c"], "Привет");
    }

    #[test]
    fn feed_preserves_emoji_split_across_chunks() {
        let payload = "data: {\"c\":\"hi 🚀🌍\"}\n\n";
        let bytes = payload.as_bytes();
        let mut p = SseParser::new();
        // Feed one byte at a time; the parser must never corrupt the emoji.
        let mut all_events = Vec::new();
        for i in 0..bytes.len() {
            all_events.extend(p.feed(&bytes[i..i + 1]));
        }
        assert_eq!(all_events.len(), 1);
        let v: Value = serde_json::from_str(&all_events[0].data).unwrap();
        assert_eq!(v["c"], "hi 🚀🌍");
    }

    #[test]
    fn feed_flags_done_sentinel() {
        let mut p = SseParser::new();
        let events = p.feed(b"data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert!(events[0].is_done);
    }

    #[test]
    fn finish_flushes_unterminated_tail() {
        let mut p = SseParser::new();
        let _ = p.feed(b"data: {\"tail\":1}\n\n");
        let tail = p.feed(b"data: {\"unterminated\":true}");
        assert!(tail.is_empty());
        let last = p.finish().expect("flushed tail");
        assert_eq!(last.data, "{\"unterminated\":true}");
    }

    #[test]
    fn finish_returns_none_for_whitespace_only_buffer() {
        let mut p = SseParser::new();
        let _ = p.feed(b"   \n  ");
        assert!(p.finish().is_none());
    }

    #[test]
    fn feed_resets_buffer_when_cap_exceeded() {
        let mut p = SseParser::new();
        // One huge un-terminated frame: no events, buffer grows.
        p.feed(&vec![b'a'; MAX_BUFFER_BYTES]);
        // Push past the cap: buffer resets, still no events, no panic.
        let events = p.feed(b"X");
        assert!(events.is_empty());
    }

    #[test]
    fn is_terminal_detects_stop_chunk() {
        let event = SseEvent {
            data: serde_json::to_string(&stop_event()).unwrap(),
            is_done: false,
        };
        assert!(event.is_terminal());
    }

    #[test]
    fn is_terminal_detects_tool_calls_finish_reason() {
        let event = SseEvent {
            data: serde_json::to_string(&json!({
                "choices": [{"index":0,"delta":{"tool_calls":[]},"finish_reason":"tool_calls"}]
            }))
            .unwrap(),
            is_done: false,
        };
        // Function-calling terminal chunks must also count as terminal so the
        // session marker is not lost on tool-calling conversations.
        assert!(event.is_terminal());
    }

    #[test]
    fn is_terminal_false_for_content_chunk_with_null_finish() {
        let event = SseEvent {
            data: serde_json::to_string(&json!({
                "choices": [{"delta": {"content": "hi"}, "finish_reason": null}]
            }))
            .unwrap(),
            is_done: false,
        };
        assert!(!event.is_terminal());
    }

    #[test]
    fn inject_marker_sets_content_when_delta_empty() {
        let injected = inject_marker(stop_event(), "\n<!-- smos:sess_abc -->");
        let v: Value = serde_json::from_str(&injected).unwrap();
        assert_eq!(
            v["choices"][0]["delta"]["content"],
            "\n<!-- smos:sess_abc -->"
        );
    }

    #[test]
    fn inject_marker_appends_to_existing_content() {
        let event = json!({
            "choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]
        });
        let injected = inject_marker(event, "\n<!-- smos:sess_1 -->");
        let v: Value = serde_json::from_str(&injected).unwrap();
        assert_eq!(
            v["choices"][0]["delta"]["content"],
            "hi\n<!-- smos:sess_1 -->"
        );
    }

    #[test]
    fn inject_marker_non_streaming_appends_to_message_content() {
        let resp = json!({
            "choices": [{"message": {"role": "assistant", "content": "hello"}}]
        });
        let v = inject_marker_non_streaming(resp, "\n<!-- smos:sess_2 -->");
        assert_eq!(
            v["choices"][0]["message"]["content"],
            "hello\n<!-- smos:sess_2 -->"
        );
    }

    #[test]
    fn inject_marker_non_streaming_creates_content_when_missing() {
        let resp = json!({"choices": [{"message": {"role": "assistant"}}]});
        let v = inject_marker_non_streaming(resp, "\n<!-- smos:sess_3 -->");
        assert_eq!(
            v["choices"][0]["message"]["content"],
            "\n<!-- smos:sess_3 -->"
        );
    }

    #[test]
    fn inject_marker_leaves_response_intact_when_choices_missing() {
        let resp = json!({"object": "chat.completion"});
        let v = inject_marker_non_streaming(resp.clone(), "\n<!-- smos:sess_4 -->");
        assert_eq!(v, resp);
    }
}
