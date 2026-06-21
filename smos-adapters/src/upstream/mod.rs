//! HTTP LLM upstream adapter (Slice-3): OpenAI-compatible passthrough.
//!
//! `ReqwestUpstream` implements `smos_application::ports::LlmUpstream` against
//! an OpenAI-compatible `/v1/chat/completions` endpoint. It forwards the
//! request verbatim (with the memory-key prefix already stripped by the
//! handler) and returns either a buffered JSON body or a raw byte stream the
//! HTTP layer tunnels back to the client as SSE.
//!
//! `ReqwestUpstreamPool` wraps N single-provider instances and routes each
//! request by the configured strategy (`single` / `round_robin` /
//! `failover`). See `reqwest_upstream` for the strategy semantics.
//!
//! `sse_parser` holds the framing + session-marker injection helpers. The
//! extraction stream wrapper in `http/stream_transform` uses both the parser
//! and `streaming_buffer` to feed the post-`[DONE]` extraction task.

pub mod reqwest_upstream;
pub mod sse_parser;
pub mod streaming_buffer;

pub use reqwest_upstream::{ReqwestUpstream, ReqwestUpstreamPool};
pub use streaming_buffer::StreamingBuffer;
