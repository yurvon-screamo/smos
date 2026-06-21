//! Domain value objects.
//!
//! Each newtype here enforces a single invariant at construction time and
//! exposes only read-only accessors afterward. Storage/transit representations
//! are serde-derived so the domain layer round-trips JSON without help from
//! any adapter.

pub mod confidence;
pub mod cosine;
pub mod embedding;
pub mod fact_content;
pub mod fact_id;
pub mod heat;
pub mod memory_key;
pub mod nli;
pub mod session_id;
pub mod source_sessions;
pub mod timestamp;

pub use confidence::Confidence;
pub use cosine::Cosine;
pub use embedding::Embedding;
pub use fact_content::FactContent;
pub use fact_id::FactId;
pub use heat::Heat;
pub use memory_key::MemoryKey;
pub use nli::{NliResult, NliScores};
pub use session_id::SessionId;
pub use source_sessions::SourceSessions;
pub use timestamp::Timestamp;
