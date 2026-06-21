//! `smos-domain` — pure domain layer of the SMOS memory OS.
//!
//! This crate is intentionally IO-free: no tokio, no axum, no reqwest, no
//! SurrealDB. Entities, value objects, and pure domain logic live here.
//! Adapters and the application layer build on top of these types in later
//! slices.
//!
//! See `ТРЕБОВАНИЯ.md` for the canonical specification.

pub mod chat;
pub mod config;
pub mod entities;
pub mod enums;
pub mod error;
pub mod value_objects;

// Re-export commonly-used types at the crate root for ergonomic call sites.
pub use entities::{Fact, SessionState};
pub use enums::{FactStatus, FactType, MergeReason, NliLabel};
pub use error::DomainError;
pub use value_objects::{
    Confidence, Cosine, Embedding, FactContent, FactId, Heat, MemoryKey, NliResult, NliScores,
    SessionId, SourceSessions, Timestamp,
};
