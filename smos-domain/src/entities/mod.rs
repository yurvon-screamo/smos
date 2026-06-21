//! Domain entities (aggregate roots).

pub mod fact;
pub mod session;

pub use fact::{Fact, MergeCandidate};
pub use session::SessionState;
