//! Shared test fixtures for the [`crate::git_sync`] submodules. Kept in
//! its own `#[cfg(test)]` module so the four test sites (`export`,
//! `import`, `manager`, `format_tests`) construct `Fact` instances
//! through a single helper rather than copy-pasting ~20 lines of
//! `Fact::new_pending` + `set_status_and_confidence` scaffolding each
//! time. When the `Fact` constructor signature changes, only this file
//! needs to track it.

#![cfg(test)]

use smos_domain::Fact;
use smos_domain::config::ConfidenceConfig;
use smos_domain::{
    Confidence, Embedding, FactStatus, MemoryKey, NewPendingRequest, SessionId, Timestamp,
};

/// Build an `Accepted` fact with a fixed session + a 3-dim placeholder
/// embedding. The caller picks the body content and the memory key — the
/// two fields tests actually vary on. Everything else is pinned so the
/// helper stays a one-liner at the call site.
pub fn sample_fact(content: &str, memory_key: &str) -> Fact {
    let session = SessionId::from_raw("sess_abcdef012345").expect("valid session id");
    let mk = MemoryKey::from_raw(memory_key)
        .unwrap_or_else(|e| panic!("invalid memory key {memory_key:?}: {e}"));
    let emb = Embedding::new(vec![0.1, 0.2, 0.3]).expect("3-dim embedding");
    let mut fact = Fact::new_pending(NewPendingRequest {
        content,
        memory_key: mk,
        session,
        embedding: emb,
        extracted_at: Timestamp::from_unix_secs(1_700_000_000).expect("valid timestamp"),
        base_confidence: ConfidenceConfig::default().base,
    })
    .expect("pending fact construction");
    fact.set_status_and_confidence(
        FactStatus::Accepted,
        Confidence::new(0.9).expect("valid confidence"),
        &ConfidenceConfig::default(),
    )
    .expect("status transition");
    fact
}
