use super::*;
use crate::testkit::{
    ConstantEmbedder, FixedClock, InMemoryFacts, NoOpDelay, RecordingEmbedder, ScriptedExtractor,
};
use crate::types::{SearchHit, SearchHitMetadata};
use smos_domain::{Fact, FactId, Timestamp};
use std::sync::Mutex;

mod format_tool_calls;
mod pipeline;
mod semantic_dedup;

// ---- Fakes live in `crate::testkit`; `RecordingSessions` is local
//      because extraction asserts on its `pending` drain order. ----

#[derive(Default, Clone)]
struct RecordingSessions {
    pending: std::sync::Arc<Mutex<Vec<FactId>>>,
}
impl SessionRepository for RecordingSessions {
    async fn add_pending(
        &self,
        _id: &SessionId,
        fact_ids: &[FactId],
    ) -> Result<(), crate::errors::RepoError> {
        self.pending
            .lock()
            .unwrap()
            .extend(fact_ids.iter().cloned());
        Ok(())
    }
    async fn get_or_create(
        &self,
        _i: &SessionId,
        _m: &MemoryKey,
    ) -> Result<smos_domain::SessionState, crate::errors::RepoError> {
        unreachable!("not used by extraction")
    }
    async fn collect_expired(
        &self,
        _t: Duration,
    ) -> Result<Vec<(SessionId, smos_domain::SessionState)>, crate::errors::RepoError> {
        Ok(Vec::new())
    }
    async fn snapshot_all(
        &self,
    ) -> Result<Vec<(SessionId, smos_domain::SessionState)>, crate::errors::RepoError> {
        Ok(Vec::new())
    }
    async fn remove_pending_owned(
        &self,
        _i: &SessionId,
        _o: &[FactId],
    ) -> Result<(), crate::errors::RepoError> {
        Ok(())
    }
    async fn clear_session(&self, _i: &SessionId) -> Result<(), crate::errors::RepoError> {
        Ok(())
    }
    async fn dedup_and_mark(
        &self,
        _i: &SessionId,
        _m: &MemoryKey,
        _c: &[FactId],
    ) -> Result<Vec<FactId>, crate::errors::RepoError> {
        Ok(Vec::new())
    }
    async fn save(
        &self,
        _i: &SessionId,
        _s: &smos_domain::SessionState,
    ) -> Result<(), crate::errors::RepoError> {
        Ok(())
    }
}

fn mk() -> MemoryKey {
    MemoryKey::from_raw("proj").unwrap()
}
fn sid(tag: u8) -> SessionId {
    SessionId::from_raw(&format!("sess_{:012x}", tag as u64)).unwrap()
}
fn cfg() -> ConfidenceConfig {
    ConfidenceConfig::default()
}
fn extraction_cfg() -> ExtractionConfig {
    ExtractionConfig::default()
}
fn clock() -> FixedClock {
    FixedClock(Timestamp::from_unix_secs(1_700_000_000).unwrap())
}

#[allow(clippy::too_many_arguments)]
fn build<'a>(
    facts: &'a InMemoryFacts,
    sessions: &'a RecordingSessions,
    extractor: &'a ScriptedExtractor,
    embedder: &'a ConstantEmbedder,
    clock: &'a FixedClock,
    cfg: &'a ConfidenceConfig,
    extraction_cfg: &'a ExtractionConfig,
) -> ExtractFactsFromResponse<
    'a,
    InMemoryFacts,
    RecordingSessions,
    ConstantEmbedder,
    ScriptedExtractor,
    FixedClock,
    NoOpDelay,
> {
    ExtractFactsFromResponse {
        facts,
        sessions,
        embedder,
        extractor,
        clock,
        delay: &NO_OP_DELAY,
        confidence_cfg: cfg,
        extraction_cfg,
        enable_response_extraction: true,
    }
}

/// Singleton no-op delay — every unit test reuses it so the retry loop
/// never actually sleeps.
static NO_OP_DELAY: NoOpDelay = NoOpDelay;

/// Shared fixture: embedder + clock + confidence config owned by the test
/// so the returned use case can borrow them for its whole lifetime.
struct Fix {
    embedder: ConstantEmbedder,
    clock: FixedClock,
    cfg: ConfidenceConfig,
    extraction_cfg: ExtractionConfig,
}
impl Fix {
    fn new() -> Self {
        Self {
            embedder: ConstantEmbedder(vec![0.1, 0.2, 0.3]),
            clock: clock(),
            cfg: cfg(),
            extraction_cfg: extraction_cfg(),
        }
    }
}

// ---- Layer 2 — semantic dedup safety net ----

/// Build a `SearchHit` whose `metadata.distance` corresponds to the given
/// cosine similarity. The store reports cosine distance, so
/// `distance = 1.0 - similarity` (Layer 2 inverts it back).
fn hit_for(fact: &Fact, similarity: f32, mk: MemoryKey) -> SearchHit {
    let metadata = SearchHitMetadata {
        status: "pending".into(),
        confidence: 0.5,
        valid_until: None,
        heat_base: 1.0,
        last_access_at: 1_700_000_000.0,
        distance: Some(1.0 - similarity),
        created_at: None,
        conflicts_with: Vec::new(),
    };
    SearchHit {
        id: fact.id().clone(),
        document: fact.content().to_string(),
        memory_key: mk,
        metadata,
    }
}

// -----------------------------------------------------------------------
// RecordingEmbedder — verify distinct facts get distinct vectors
// -----------------------------------------------------------------------

/// Build a use case backed by a `RecordingEmbedder`. The default
/// `embed_batch` loops `embed`, so every fact handed to the pipeline
/// produces one recorded call.
#[allow(clippy::too_many_arguments)]
fn build_with_recording_embedder<'a>(
    facts: &'a InMemoryFacts,
    sessions: &'a RecordingSessions,
    extractor: &'a ScriptedExtractor,
    embedder: &'a RecordingEmbedder,
    clock: &'a FixedClock,
    cfg: &'a ConfidenceConfig,
    extraction_cfg: &'a ExtractionConfig,
) -> ExtractFactsFromResponse<
    'a,
    InMemoryFacts,
    RecordingSessions,
    RecordingEmbedder,
    ScriptedExtractor,
    FixedClock,
    NoOpDelay,
> {
    ExtractFactsFromResponse {
        facts,
        sessions,
        embedder,
        extractor,
        clock,
        delay: &NO_OP_DELAY,
        confidence_cfg: cfg,
        extraction_cfg,
        enable_response_extraction: true,
    }
}
