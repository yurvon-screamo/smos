//! Shared test doubles for `smos-application` unit tests (refactor slice R4).
//!
//! Classicist-style fakes: in-memory repositories plus scripted providers. The
//! three use-case test modules (`finalize_session`, `extract_facts_from_response`,
//! `import_opencode_session`) share these instead of maintaining three divergent
//! copies. Adapter-layer doubles against the real `SurrealStore` / live NLI
//! backend remain in `smos-adapters/tests/`.
//!
//! The module is unconditionally `pub` (not `#[cfg(test)]`) so that downstream
//! crates' unit tests can reference `smos_application::testkit`; a `cfg(test)`
//! gate would make the path vanish when the crate is compiled as a dependency.

pub mod clock;
pub mod facts;
pub mod providers;
pub mod sessions;

pub use clock::{FixedClock, NoOpDelay};
pub use facts::InMemoryFacts;
pub use providers::{
    ConstantEmbedder, RecordingEmbedder, ScriptedExtractor, ScriptedNliClassifier,
};
pub use sessions::InMemorySessions;

#[cfg(test)]
mod tests;
