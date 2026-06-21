//! Application-layer helpers — protocol-shaped pure functions.
//!
//! These helpers operate on wire-shape data (OpenAI message envelopes, JSON
//! payloads, SMOS-internal text markers) and are NOT domain concerns. They take
//! domain types in and return domain types out, but they live here so the
//! domain layer stays free of `serde_json::Value` / `regex` / HTTP-protocol
//! parsing.
//!
//! Migrated out of `smos-domain::services` to enforce the strict DDD rule:
//! the domain layer contains only entities and value objects.

pub mod memory_block;
pub mod model_parser;
pub mod noise_filter;
pub mod openai_content;
pub mod request_enricher;
pub mod retrieval_planner;
pub mod session_marker;
pub mod topic_extractor;
