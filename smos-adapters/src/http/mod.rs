//! HTTP server (axum) + OpenAI-compatible routes.
//!
//! `axum_server` wires the `AppState`, builds the router, and runs `axum::serve`
//! with graceful shutdown. `routes/` holds the handlers; `stream_transform`
//! turns an upstream byte stream into the SSE response with the session
//! marker appended; `error_mapper` maps `UpstreamError` to HTTP status codes.

pub mod axum_server;
pub mod error_mapper;
pub mod routes;
pub mod stream_transform;
