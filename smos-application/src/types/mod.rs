//! Transport / DTO types used at port boundaries.
//!
//! These types live in the application layer (not domain) because they model
//! the *wire* shape — OpenAI-compatible chat envelopes, vector-search hits,
//! rerank responses — rather than the invariants the domain layer enforces.

pub mod chat_request;
pub mod chat_response;
pub mod enrichment_messages;
pub mod merge_result;
pub mod rerank_result;
pub mod search_hit;

pub use chat_request::ChatRequest;
pub use chat_response::ChatResponse;
pub use enrichment_messages::{
    ChatMessageDto, ContentPart, EnrichmentMessages, MessageContent, ToolCallDto, ToolCallFunction,
    enrichment_messages_from_json, enrichment_messages_to_json,
};
pub use merge_result::MergeResult;
pub use rerank_result::RerankResult;
pub use search_hit::{SearchHit, SearchHitMetadata};

// NLI types are pure domain value objects, so `NliResult`/`NliScores` live in
// `smos-domain::value_objects::nli` and are re-exported here so callers have a
// single import path through the application layer.
pub use smos_domain::{NliResult, NliScores};

// `MergeCandidate` is similarly defined in the domain (`entities::fact`). It is
// re-exported here for ergonomic single-source imports; the domain remains its
// canonical home.
pub use smos_domain::entities::MergeCandidate;
