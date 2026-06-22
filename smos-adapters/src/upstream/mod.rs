//! HTTP LLM upstream adapter (Slice-3): OpenAI-compatible passthrough.
//!
//! `ReqwestUpstream` implements `smos_application::ports::LlmUpstream` against
//! an OpenAI-compatible `/v1/chat/completions` endpoint. It forwards the
//! request verbatim (with the upstream model already rewritten by the
//! application layer's `route_request`) and returns either a buffered JSON
//! body or a raw byte stream the HTTP layer tunnels back to the client as
//! SSE.
//!
//! `ReqwestUpstreamRouter` wraps N single-provider instances keyed by name.
//! Each request is routed to the provider named by the caller — the name is
//! resolved upstream by `route_request` against the `[persons.*]` map.
//!
//! `sse_parser` holds the framing + session-marker injection helpers. The
//! extraction stream wrapper in `http/stream_transform` uses both the parser
//! and `streaming_buffer` to feed the post-`[DONE]` extraction task.

pub mod reqwest_upstream;
pub mod sse_parser;
pub mod streaming_buffer;

pub use reqwest_upstream::{ReqwestUpstream, ReqwestUpstreamRouter};
pub use streaming_buffer::StreamingBuffer;
