//! `StreamingBuffer` — concurrent accumulator for the streaming extraction
//! path (§4 + §5).
//!
//! The streaming passthrough forwards every upstream chunk to the client 1:1
//! (with the session marker appended to the terminal chunk). Slice-5 ALSO
//! copies `content` + reassembled `tool_calls` into this buffer so the
//! post-`[DONE]` extraction prompt sees the full reply.
//!
//! OpenAI streams `tool_calls` incrementally by `index`: the first delta
//! carries `function.name`, subsequent deltas append to
//! `function.arguments` (a JSON **string** split across chunks). The buffer
//! reassembles them positionally and wraps the arguments string into a
//! [`ToolCall`] on [`StreamingBuffer::finalize`]. The arguments stay opaque
//! at the domain layer (raw string); adapters parse the JSON when needed.

use std::sync::Arc;

use smos_domain::chat::{ToolArguments, ToolCall};
use tokio::sync::Mutex;

/// Concurrent buffer that accumulates `content` + `tool_calls` from a
/// streaming response. Cheap to clone (one `Arc`); the clone shares the same
/// underlying state.
#[derive(Clone, Default)]
pub struct StreamingBuffer {
    inner: Arc<Mutex<BufferState>>,
}

#[derive(Default)]
struct BufferState {
    content: String,
    tool_calls: Vec<PartialToolCall>,
}

/// Defense-in-depth cap on the total accumulated `content` length.
///
/// The streaming passthrough forwards content to the client 1:1 as it
/// arrives, so the buffer is not the production memory bound — the HTTP
/// layer is. This cap exists so a runaway upstream (or a bug that disables
/// client flow-control) cannot grow `content` unbounded and OOM the proxy
/// process. 16 MB is well above any legitimate assistant reply; a hit is
/// logged at WARN and the offending delta is dropped, preserving whatever
/// was accumulated so far.
///
/// Trade-off: dropped deltas affect the **post-stream extraction prompt**
/// (see `spawn_extraction`), not the client SSE stream — the client already
/// received every byte 1:1. The extraction task sees a truncated buffer and
/// its NLI verdict degrades accordingly. For content past the 16 MB cap
/// the extraction is questionable anyway, but callers must not assume the
/// buffer ever holds the *complete* upstream reply.
pub const MAX_CONTENT_BYTES: usize = 16 * 1024 * 1024;

/// Sanity cap on the `index` of an `append_tool_call_delta` call.
///
/// OpenAI's streaming protocol reuses `index` to identify each tool call
/// positionally; a healthy stream never has more than a handful. A wildly
/// out-of-range `index` (e.g. `u64::MAX` from a malformed chunk) would let
/// one bad chunk allocate `index + 1` slots — `Vec<PartialToolCall>` — and
/// OOM the proxy. The cap is generous (1024 tool calls in a single
/// assistant turn is well above any reasonable usage) and a hit is logged
/// at WARN so a misbehaving upstream is visible without crashing the
/// stream.
///
/// Semantics: indices `0..MAX_TOOL_CALLS` are accepted (exactly
/// `MAX_TOOL_CALLS` slots); `index >= MAX_TOOL_CALLS` is dropped. The
/// name pins the slot count, not the maximum accepted index.
pub const MAX_TOOL_CALLS: usize = 1024;

/// In-progress tool call before its arguments JSON is complete.
#[derive(Default)]
struct PartialToolCall {
    name: String,
    args: String,
}

impl StreamingBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a content delta (SSE `choices[0].delta.content`).
    ///
    /// Defense-in-depth: if `content.len() + delta.len()` would exceed
    /// [`MAX_CONTENT_BYTES`], the delta is dropped with a WARN log rather
    /// than letting a runaway upstream grow the buffer unbounded. The
    /// check uses `saturating_add` so a pathologically huge `delta.len()`
    /// cannot wrap past the cap on a 32-bit target.
    pub async fn append_content(&self, delta: &str) {
        let mut state = self.inner.lock().await;
        // On 64-bit targets the saturating branch is unreachable in practice
        // (memory OOM happens first); it stays as defense-in-depth for 32-bit.
        let new_len = state.content.len().saturating_add(delta.len());
        if new_len > MAX_CONTENT_BYTES {
            tracing::warn!(
                current = state.content.len(),
                delta = delta.len(),
                max = MAX_CONTENT_BYTES,
                "content buffer exceeds cap, dropping delta"
            );
            return;
        }
        state.content.push_str(delta);
    }

    /// Append a tool-call delta for the tool call at `index`. SSE chunks arrive
    /// incrementally: the first delta usually carries `function.name`, later
    /// deltas append to `function.arguments`. Missing deltas (`None`) leave
    /// the field untouched.
    ///
    /// `index >= MAX_TOOL_CALLS` is dropped (with a WARN log) so a malformed
    /// upstream chunk cannot OOM the proxy by forcing a huge `Vec`
    /// allocation. The cap is generous: 1024 tool calls in a single
    /// assistant turn is well above any healthy usage.
    pub async fn append_tool_call_delta(
        &self,
        index: usize,
        name_delta: Option<&str>,
        args_delta: Option<&str>,
    ) {
        if index >= MAX_TOOL_CALLS {
            tracing::warn!(
                index,
                max = MAX_TOOL_CALLS,
                "tool_call index exceeds sanity cap; dropping delta to avoid OOM"
            );
            return;
        }
        let mut state = self.inner.lock().await;
        while state.tool_calls.len() <= index {
            state.tool_calls.push(PartialToolCall::default());
        }
        let slot = &mut state.tool_calls[index];
        if let Some(name) = name_delta {
            slot.name.push_str(name);
        }
        if let Some(args) = args_delta {
            slot.args.push_str(args);
        }
    }

    /// Drain the accumulated content + reassembled tool calls. Each tool
    /// call's accumulated arguments string is wrapped into the opaque
    /// [`ToolArguments`] verbatim — no JSON parsing happens at this layer
    /// (an unparseable string stays unparseable, the next layer that
    /// cares can decide what to do with it).
    pub async fn finalize(self) -> (String, Vec<ToolCall>) {
        let mut state = self.inner.lock().await;
        let content = std::mem::take(&mut state.content);
        let partials = std::mem::take(&mut state.tool_calls);
        let tool_calls = partials
            .into_iter()
            .map(|p| ToolCall {
                name: p.name,
                arguments: ToolArguments::from_json(p.args),
            })
            .collect();
        (content, tool_calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn append_content_concatenates_deltas() {
        let buf = StreamingBuffer::new();
        buf.append_content("Hello ").await;
        buf.append_content("world").await;
        let (content, calls) = buf.finalize().await;
        assert_eq!(content, "Hello world");
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn append_content_drops_delta_above_cap() {
        // A delta that would push the buffer past MAX_CONTENT_BYTES must be
        // dropped; the buffer stays at its prior size (empty here) instead of
        // allocating the runaway delta.
        let buf = StreamingBuffer::new();
        let big = "x".repeat(MAX_CONTENT_BYTES + 1);
        buf.append_content(&big).await;
        let state = buf.inner.lock().await;
        assert!(state.content.is_empty(), "content should be dropped");
    }

    #[tokio::test]
    async fn append_content_accepts_delta_at_exact_cap_boundary() {
        // Boundary: the check is strict `> MAX_CONTENT_BYTES`, so a delta
        // that lands the buffer EXACTLY on `MAX_CONTENT_BYTES` is accepted.
        // This guards against an off-by-one (`>=` regression) that would
        // reject the last legal byte.
        let buf = StreamingBuffer::new();
        let exact = "x".repeat(MAX_CONTENT_BYTES);
        buf.append_content(&exact).await;
        let state = buf.inner.lock().await;
        assert_eq!(
            state.content.len(),
            MAX_CONTENT_BYTES,
            "delta landing exactly on the cap must be accepted"
        );
    }

    #[tokio::test]
    async fn append_content_drops_delta_one_byte_above_cap() {
        // Boundary mirror: delta of `MAX_CONTENT_BYTES + 1` from an empty
        // buffer must be rejected. Pairs with the exact-cap test to pin the
        // `>` vs `>=` boundary.
        let buf = StreamingBuffer::new();
        let over = "x".repeat(MAX_CONTENT_BYTES + 1);
        buf.append_content(&over).await;
        let state = buf.inner.lock().await;
        assert_eq!(
            state.content.len(),
            0,
            "delta one byte above the cap must be dropped"
        );
    }

    #[tokio::test]
    async fn append_content_accumulation_then_drop_streaming_scenario() {
        // Realistic streaming scenario: many small deltas fill the buffer
        // close to the cap, then a final delta that would cross it is
        // dropped while the previously accumulated content is preserved.
        let buf = StreamingBuffer::new();
        // Fill to within 100 bytes of the cap.
        let fill = "x".repeat(MAX_CONTENT_BYTES - 100);
        buf.append_content(&fill).await;
        // Crossing delta — must be dropped, accumulated content stays.
        buf.append_content(&"y".repeat(200)).await;
        let state = buf.inner.lock().await;
        assert_eq!(
            state.content.len(),
            MAX_CONTENT_BYTES - 100,
            "accumulated content preserved; crossing delta dropped"
        );
        // Last 100 bytes of headroom still fit.
        drop(state);
        buf.append_content(&"z".repeat(100)).await;
        let state = buf.inner.lock().await;
        assert_eq!(
            state.content.len(),
            MAX_CONTENT_BYTES,
            "delta within remaining headroom is accepted"
        );
    }

    #[tokio::test]
    async fn append_content_small_delta_after_giant_drop_still_accepted() {
        // After a giant delta is dropped (buffer frozen at its prior size),
        // a subsequent small delta that fits under the cap must still be
        // accepted — the cap is on `content.len() + delta.len()`, not a
        // "permanently poisoned" flag.
        let buf = StreamingBuffer::new();
        let seed = "x".repeat(1_000);
        buf.append_content(&seed).await;
        // Giant drop.
        buf.append_content(&"y".repeat(MAX_CONTENT_BYTES)).await;
        // Small delta must still append to the seed.
        buf.append_content("tail").await;
        let state = buf.inner.lock().await;
        assert_eq!(
            state.content.len(),
            1_000 + "tail".len(),
            "small delta after a dropped giant delta must still be accepted"
        );
    }

    #[tokio::test]
    async fn append_tool_call_assembles_name_then_arguments() {
        let buf = StreamingBuffer::new();
        // First delta: name only.
        buf.append_tool_call_delta(0, Some("read_file"), None).await;
        // Second delta: arguments fragment.
        buf.append_tool_call_delta(0, None, Some("{\"path\":\"a"))
            .await;
        // Third delta: arguments tail.
        buf.append_tool_call_delta(0, None, Some("uth.rs\"}")).await;

        let (_content, calls) = buf.finalize().await;
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        // The opaque string carries the assembled JSON verbatim; the test
        // asserts the raw payload rather than indexing into a Value because
        // the domain layer no longer parses the arguments.
        assert_eq!(calls[0].arguments.as_str(), r#"{"path":"auth.rs"}"#);
    }

    #[tokio::test]
    async fn append_tool_call_handles_multiple_indices_in_order() {
        let buf = StreamingBuffer::new();
        buf.append_tool_call_delta(0, Some("first"), None).await;
        buf.append_tool_call_delta(1, Some("second"), Some("{}"))
            .await;
        buf.append_tool_call_delta(0, None, Some("{\"x\":1}")).await;

        let (_content, calls) = buf.finalize().await;
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "first");
        assert_eq!(calls[0].arguments.as_str(), r#"{"x":1}"#);
        assert_eq!(calls[1].name, "second");
        assert_eq!(calls[1].arguments.as_str(), r#"{}"#);
    }

    #[tokio::test]
    async fn finalize_keeps_raw_arguments_when_unparseable() {
        let buf = StreamingBuffer::new();
        buf.append_tool_call_delta(0, Some("x"), Some("{not json"))
            .await;
        let (_content, calls) = buf.finalize().await;
        assert_eq!(calls.len(), 1);
        // Opaque payload preserves the malformed JSON verbatim; downstream
        // parsers decide what to do with it.
        assert_eq!(calls[0].arguments.as_str(), "{not json");
    }

    #[tokio::test]
    async fn empty_buffer_finalizes_to_empty() {
        let buf = StreamingBuffer::new();
        let (content, calls) = buf.finalize().await;
        assert!(content.is_empty());
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn clone_shares_state_between_handles() {
        let buf = StreamingBuffer::new();
        let clone = buf.clone();
        clone.append_content("shared").await;
        let (content, _calls) = buf.finalize().await;
        assert_eq!(content, "shared", "clone shares the underlying state");
    }

    #[tokio::test]
    async fn append_tool_call_delta_drops_index_above_cap() {
        // An out-of-range `index` must NOT trigger a huge Vec allocation.
        // The buffer surfaces a WARN and discards the delta; the existing
        // tool calls (if any) are preserved unchanged.
        let buf = StreamingBuffer::new();
        buf.append_tool_call_delta(0, Some("keep"), None).await;
        buf.append_tool_call_delta(MAX_TOOL_CALLS, Some("drop"), None)
            .await;
        let (_content, calls) = buf.finalize().await;
        assert_eq!(
            calls.len(),
            1,
            "out-of-range index must not allocate a slot"
        );
        assert_eq!(calls[0].name, "keep");
    }

    #[tokio::test]
    async fn append_tool_call_delta_accepts_index_just_below_cap() {
        // Boundary: index == MAX_TOOL_CALLS - 1 is the last allowed slot.
        // The cap is on slot count, so exactly `MAX_TOOL_CALLS` slots
        // (indices 0..MAX_TOOL_CALLS) are accepted.
        let buf = StreamingBuffer::new();
        let last_allowed = MAX_TOOL_CALLS - 1;
        buf.append_tool_call_delta(last_allowed, Some("edge"), None)
            .await;
        let (_content, calls) = buf.finalize().await;
        assert_eq!(
            calls.len(),
            MAX_TOOL_CALLS,
            "index == MAX_TOOL_CALLS - 1 must allocate"
        );
        assert_eq!(calls[last_allowed].name, "edge");
    }
}
