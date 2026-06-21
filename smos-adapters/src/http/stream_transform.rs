//! Transform an upstream byte stream into the SSE response with the session
//! marker appended (§4) and, in Slice-5, an optional parallel
//! [`StreamingBuffer`] that feeds the post-`[DONE]` fact-extraction task.
//!
//! Two entry points share the marker-emission logic but have genuinely
//! different per-chunk cost:
//!
//! - [`inject_marker`] — lightweight: forwards chunks 1:1, injects the marker
//!   on the terminal chunk. Used when `enable_response_extraction = false`
//!   (no per-chunk buffering, no `StreamingBuffer`, no JSON parsing).
//! - [`inject_marker_with_extraction`] — additionally feeds a
//!   [`StreamingBuffer`] every chunk and hands the finalised payload to an
//!   [`ExtractionSpawner`] once the stream ends. Used when extraction is on.
//!
//! Both append the marker to `/choices/0` on the terminal chunk (any non-null
//! `finish_reason`), forward the `[DONE]` sentinel, and emit a synthetic
//! marker chunk on an abnormal close (no terminal chunk AND no `[DONE]`) so
//! the client still receives the session trailer.

use std::convert::Infallible;

use async_stream::stream;
use axum::response::sse::Event;
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use serde_json::{Value, json};
use smos_application::errors::UpstreamError;
use smos_domain::chat::ToolCall;

use crate::upstream::sse_parser::{self, SseEvent, SseParser};
use crate::upstream::streaming_buffer::{MAX_TOOL_CALLS, StreamingBuffer};

/// Inject the marker into a terminal event and return the re-serialised data
/// payload; on a non-terminal/`[DONE]` event the original data is returned
/// unchanged.
fn inject_if_terminal(event: &SseEvent, marker: &str) -> String {
    if event.is_done {
        return event.data.clone();
    }
    if event.is_terminal() {
        sse_parser::inject_marker(event.parsed_json().unwrap_or_else(|| json!({})), marker)
    } else {
        event.data.clone()
    }
}

/// Emission decision for one SSE event, shared by both stream wrappers so the
/// marker logic lives in exactly one place.
enum EmitOutcome {
    /// The `[DONE]` sentinel.
    Done,
    /// A data event to forward (marker already injected when terminal).
    Event(String),
}

/// Classify one parsed event into the data the client should receive.
fn classify_event(event: &SseEvent, marker: &str, marker_emitted: &mut bool) -> EmitOutcome {
    if event.is_done {
        return EmitOutcome::Done;
    }
    if event.is_terminal() && !*marker_emitted {
        *marker_emitted = true;
        return EmitOutcome::Event(inject_if_terminal(event, marker));
    }
    EmitOutcome::Event(event.data.clone())
}

/// Hook the post-`[DONE]` extraction task into a streaming response.
///
/// `spawn_extraction` is called exactly once, after the stream has been fully
/// consumed (the client has received `[DONE]`). It owns every port the
/// extraction pipeline needs and runs the task detached so it never blocks the
/// response. Implementations must be `Send` (the stream is forwarded across
/// tasks by axum).
pub trait ExtractionSpawner: Send {
    /// Consume the spawner and launch the background extraction task.
    fn spawn_extraction(self, content: String, tool_calls: Vec<ToolCall>);
}

/// Lightweight marker-only wrapper: forwards chunks 1:1 and injects the
/// session marker on the terminal chunk. No `StreamingBuffer`, no per-chunk
/// JSON parsing — the hot path when extraction is disabled.
pub fn inject_marker<S>(
    upstream: S,
    marker: String,
) -> impl Stream<Item = Result<Event, Infallible>> + Send
where
    S: Stream<Item = Result<Bytes, UpstreamError>> + Send + Unpin + 'static,
{
    stream! {
        let mut upstream = upstream;
        let mut parser = SseParser::new();
        let mut marker_emitted = false;
        let mut done_seen = false;

        while let Some(result) = upstream.next().await {
            match result {
                Ok(bytes) => {
                    for event in parser.feed(&bytes) {
                        match classify_event(&event, &marker, &mut marker_emitted) {
                            EmitOutcome::Done => {
                                done_seen = true;
                                yield Ok(Event::default().data("[DONE]"));
                            }
                            EmitOutcome::Event(data) => yield Ok(Event::default().data(data)),
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "upstream stream error; closing SSE response");
                    break;
                }
            }
        }

        if let Some(tail) = parser.finish() {
            match classify_event(&tail, &marker, &mut marker_emitted) {
                EmitOutcome::Done => {
                    done_seen = true;
                    yield Ok(Event::default().data("[DONE]"));
                }
                EmitOutcome::Event(data) => yield Ok(Event::default().data(data)),
            }
        }

        if !marker_emitted && !done_seen {
            let synthetic = json!({
                "choices": [{"index": 0, "delta": {"content": marker}, "finish_reason": null}]
            });
            yield Ok(Event::default().data(synthetic.to_string()));
        }
    }
}

/// Wrap an upstream byte stream so that:
/// 1. Every chunk is forwarded to the client (marker appended to the terminal
///    chunk — same behaviour as [`inject_marker`]).
/// 2. Every chunk ALSO feeds `buffer` (content + tool-call deltas).
/// 3. Once the stream ends, `buffer` is finalised and `spawner.spawn_extraction`
///    launches the background extraction task.
pub fn inject_marker_with_extraction<S, E>(
    upstream: S,
    marker: String,
    buffer: StreamingBuffer,
    spawner: E,
) -> impl Stream<Item = Result<Event, Infallible>> + Send
where
    S: Stream<Item = Result<Bytes, UpstreamError>> + Send + Unpin + 'static,
    E: ExtractionSpawner + 'static,
{
    stream! {
        let mut upstream = upstream;
        let mut parser = SseParser::new();
        let mut marker_emitted = false;
        let mut done_seen = false;

        while let Some(result) = upstream.next().await {
            match result {
                Ok(bytes) => {
                    for event in parser.feed(&bytes) {
                        feed_buffer(&buffer, &event).await;
                        match classify_event(&event, &marker, &mut marker_emitted) {
                            EmitOutcome::Done => {
                                done_seen = true;
                                yield Ok(Event::default().data("[DONE]"));
                            }
                            EmitOutcome::Event(data) => yield Ok(Event::default().data(data)),
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "upstream stream error; closing SSE response");
                    break;
                }
            }
        }

        if let Some(tail) = parser.finish() {
            feed_buffer(&buffer, &tail).await;
            match classify_event(&tail, &marker, &mut marker_emitted) {
                EmitOutcome::Done => {
                    done_seen = true;
                    yield Ok(Event::default().data("[DONE]"));
                }
                EmitOutcome::Event(data) => yield Ok(Event::default().data(data)),
            }
        }

        if !marker_emitted && !done_seen {
            let synthetic = json!({
                "choices": [{"index": 0, "delta": {"content": marker}, "finish_reason": null}]
            });
            yield Ok(Event::default().data(synthetic.to_string()));
        }

        // Stream fully consumed: hand the buffered payload to the extraction
        // task. Non-blocking — `spawn_extraction` detaches the work.
        let (content, tool_calls) = buffer.finalize().await;
        spawner.spawn_extraction(content, tool_calls);
    }
}

/// Feed one SSE event's `content` + `tool_calls` deltas into `buffer`.
/// `[DONE]` and non-JSON frames contribute nothing.
async fn feed_buffer(buffer: &StreamingBuffer, event: &SseEvent) {
    if event.is_done {
        return;
    }
    let Some(json) = event.parsed_json() else {
        return;
    };
    if let Some(content) = json
        .pointer("/choices/0/delta/content")
        .and_then(Value::as_str)
    {
        buffer.append_content(content).await;
    }
    if let Some(tool_calls) = json
        .pointer("/choices/0/delta/tool_calls")
        .and_then(Value::as_array)
    {
        for entry in tool_calls {
            feed_tool_call_delta(buffer, entry).await;
        }
    }
}

/// Feed a single `tool_calls[]` delta entry into `buffer`, keyed by `index`.
///
/// `index >= MAX_TOOL_CALLS` is dropped at this layer too (defence-in-
/// depth: the buffer enforces the same cap, but filtering at the parser
/// avoids even constructing the request). A WARN log surfaces the
/// malformed chunk so a misbehaving upstream is visible.
async fn feed_tool_call_delta(buffer: &StreamingBuffer, entry: &Value) {
    let index = entry
        .get("index")
        .and_then(Value::as_u64)
        .map(|i| i as usize)
        .unwrap_or(0);
    if index >= MAX_TOOL_CALLS {
        tracing::warn!(
            index,
            max = MAX_TOOL_CALLS,
            "stream chunk carried a tool_call index above the sanity cap; dropping"
        );
        return;
    }
    let name = entry.pointer("/function/name").and_then(Value::as_str);
    let args = entry.pointer("/function/arguments").and_then(Value::as_str);
    buffer.append_tool_call_delta(index, name, args).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use std::sync::{Arc, Mutex};

    fn bytes_stream(
        parts: Vec<String>,
    ) -> Box<dyn Stream<Item = Result<Bytes, UpstreamError>> + Send + Unpin> {
        let items: Vec<Result<Bytes, UpstreamError>> =
            parts.into_iter().map(|p| Ok(Bytes::from(p))).collect();
        Box::new(stream::iter(items))
    }

    /// Drain a stream to completion, discarding emitted events. Used by tests
    /// that only assert on side effects (spawner invocation) — the emitted
    /// marker/content payloads are covered end-to-end via the HTTP suites
    /// (`axum::response::sse::Event` exposes no public data reader).
    async fn drain<S>(s: S)
    where
        S: Stream<Item = Result<Event, Infallible>>,
    {
        futures::pin_mut!(s);
        while s.next().await.is_some() {}
    }

    #[tokio::test]
    async fn inject_marker_lightweight_path_completes_without_extraction() {
        // The disabled path: inject_marker must NOT allocate a buffer or call
        // any feed logic. Completing without panic is the contract.
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";
        drain(inject_marker(
            bytes_stream(vec![body.into()]),
            "<!-- smos:sess_1 -->".into(),
        ))
        .await;
    }

    #[derive(Default)]
    struct Recording {
        content: Arc<Mutex<Option<String>>>,
        calls: Arc<Mutex<Vec<ToolCall>>>,
        invoked: Arc<Mutex<bool>>,
    }
    impl ExtractionSpawner for Recording {
        fn spawn_extraction(self, content: String, tool_calls: Vec<ToolCall>) {
            *self.content.lock().unwrap() = Some(content);
            *self.calls.lock().unwrap() = tool_calls;
            *self.invoked.lock().unwrap() = true;
        }
    }

    #[tokio::test]
    async fn extraction_feeds_buffer_and_invokes_spawner_at_stream_end() {
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello \"},\"finish_reason\":null}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":null}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
                    data: [DONE]\n\n";

        let content_slot = Arc::new(Mutex::new(None));
        let calls_slot = Arc::new(Mutex::new(Vec::new()));
        let invoked_slot = Arc::new(Mutex::new(false));
        let rec = Recording {
            content: content_slot.clone(),
            calls: calls_slot.clone(),
            invoked: invoked_slot.clone(),
        };

        drain(inject_marker_with_extraction(
            bytes_stream(vec![body.into()]),
            "\n<!-- smos:sess_x -->".into(),
            StreamingBuffer::new(),
            rec,
        ))
        .await;

        assert!(
            *invoked_slot.lock().unwrap(),
            "spawner invoked at stream end"
        );
        assert_eq!(
            content_slot.lock().unwrap().as_deref(),
            Some("Hello world"),
            "buffered content is the concatenation of deltas"
        );
        assert!(calls_slot.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn extraction_assembles_tool_call_deltas_into_spawner_payload() {
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"name\":\"read_file\"}}]},\"finish_reason\":null}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\"}}]},\"finish_reason\":null}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"auth.rs\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
                    data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                    data: [DONE]\n\n";

        let captured = Arc::new(Mutex::new(None));
        type Captured = Option<(String, Vec<ToolCall>)>;
        #[derive(Clone)]
        struct Capture(Arc<Mutex<Captured>>);
        impl ExtractionSpawner for Capture {
            fn spawn_extraction(self, content: String, tool_calls: Vec<ToolCall>) {
                *self.0.lock().unwrap() = Some((content, tool_calls));
            }
        }
        let cap = Capture(captured.clone());

        drain(inject_marker_with_extraction(
            bytes_stream(vec![body.into()]),
            "\n<!-- smos:sess_y -->".into(),
            StreamingBuffer::new(),
            cap,
        ))
        .await;

        let guard = captured.lock().unwrap();
        let (content, calls) = guard.as_ref().expect("spawner invoked");
        assert!(content.is_empty(), "no content deltas in this stream");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments.as_str(), r#"{"path":"auth.rs"}"#);
    }

    #[tokio::test]
    async fn extraction_invoked_even_on_abnormal_close() {
        let body = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n";
        let invoked = Arc::new(Mutex::new(false));
        #[derive(Clone)]
        struct Flag(Arc<Mutex<bool>>);
        impl ExtractionSpawner for Flag {
            fn spawn_extraction(self, _c: String, _t: Vec<ToolCall>) {
                *self.0.lock().unwrap() = true;
            }
        }
        let flag = Flag(invoked.clone());
        drain(inject_marker_with_extraction(
            bytes_stream(vec![body.into()]),
            "<!-- smos:sess_z -->".into(),
            StreamingBuffer::new(),
            flag,
        ))
        .await;
        assert!(
            *invoked.lock().unwrap(),
            "extraction runs even without [DONE]"
        );
    }
}
