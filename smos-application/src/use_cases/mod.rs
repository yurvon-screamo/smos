//! Use cases — orchestrations over the port traits.
//!
//! A use case wires several ports together to deliver one externally-visible
//! capability (`EnrichRequest`, `HandleChatCompletion`, …). The use-case layer
//! is intentionally async and IO-aware — it is the *only* place where the
//! ordering of port calls and fail-open policies live. Pure domain logic
//! (entity methods, value objects) and protocol helpers (`crate::helpers`) are
//! called freely; their results are persisted / forwarded via the ports.

pub mod enrich_request;
pub mod extract_facts_from_response;
pub mod finalize_session;
pub mod handle_chat_completion;
pub mod import_opencode_session;

pub use enrich_request::EnrichRequest;
pub use extract_facts_from_response::{ExtractFactsFromResponse, format_tool_calls};
pub use finalize_session::{FinalizeSession, FinalizeStats};
pub use handle_chat_completion::{HandleChatCompletion, extract_response_payload};
pub use import_opencode_session::{AssistantTurn, ImportOpencodeSession, ImportStats};
