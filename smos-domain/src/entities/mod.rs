//! Domain entities (aggregate roots).

pub mod fact;
pub mod session;

pub use fact::{Fact, FactRecord, MergeCandidate, NewPendingRequest};
pub use session::{SessionRecord, SessionState};
