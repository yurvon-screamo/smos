//! Parity tests pinning the contract of each testkit double.
//!
//! These run as part of the workspace test suite and MUST stay green both
//! before and after the use-case test modules migrate onto the testkit. A
//! failure here means the shared double drifted from the contract a migrated
//! test relied on.
//!
//! Fixtures live here and are shared by the [`facts_sessions`] and
//! [`providers`] submodules via `use super::*`.

use super::*;
use crate::ports::{
    Clock, EmbeddingProvider, FactRepository, LlmExtractor, NliClassifier, SessionRepository,
};
use smos_domain::config::ConfidenceConfig;
use smos_domain::enums::NliLabel;
use smos_domain::{
    Confidence, Embedding, Fact, FactId, FactStatus, MemoryKey, NewPendingRequest, NliResult,
    NliScores, SessionId, Timestamp,
};

mod facts_sessions;
mod providers;

// ---- Fixtures (private; visible to child submodules) ----

fn ts() -> Timestamp {
    Timestamp::from_unix_secs(1_700_000_000).unwrap()
}

fn sid(n: u8) -> SessionId {
    SessionId::from_raw(&format!("sess_{:012x}", n as u64)).unwrap()
}

fn mk(name: &str) -> MemoryKey {
    MemoryKey::from_raw(name).unwrap()
}

fn pending(content: &str, embedding: Vec<f32>) -> Fact {
    Fact::new_pending(NewPendingRequest {
        content,
        memory_key: mk("testkit"),
        session: sid(1),
        embedding: Embedding::new(embedding).unwrap(),
        extracted_at: ts(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap()
}

fn accepted(content: &str, embedding: Vec<f32>) -> Fact {
    let mut f = pending(content, embedding);
    f.set_status_and_confidence(
        FactStatus::Accepted,
        Confidence::new(0.9).unwrap(),
        &ConfidenceConfig::default(),
    )
    .unwrap();
    f
}

fn fact_in(content: &str, memory_key: MemoryKey, session: SessionId) -> Fact {
    Fact::new_pending(NewPendingRequest {
        content,
        memory_key,
        session,
        embedding: Embedding::new(vec![1.0]).unwrap(),
        extracted_at: ts(),
        base_confidence: ConfidenceConfig::default().base,
    })
    .unwrap()
}

fn neutral_verdict() -> NliResult {
    NliResult {
        label: NliLabel::Neutral,
        scores: NliScores {
            entailment: 0.2,
            neutral: 0.7,
            contradiction: 0.1,
        },
        available: true,
    }
}
