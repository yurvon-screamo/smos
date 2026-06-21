//! Port-trait shape tests.
//!
//! These tests do NOT exercise any production adapter. They confirm that the
//! port traits are *implementable* (a `Mock*` exists for each), that generic
//! dispatch works (`fn use_it<T: Trait>(t: &T)`), and that the default
//! `embed_batch` is reachable through the `EmbeddingProvider` trait.

#![allow(async_fn_in_trait)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use smos_application::errors::{ProviderError, RepoError, UpstreamError};
use smos_application::ports::{
    Clock, EmbeddingProvider, FactRepository, IdGenerator, LlmExtractor, LlmUpstream,
    NliClassifier, RerankProvider, SessionRepository,
};
use smos_application::types::{ChatRequest, ChatResponse, MergeResult, RerankResult, SearchHit};
use smos_domain::{
    Fact, FactId, Heat, MemoryKey, NliResult, NliScores, SessionId, SessionState, Timestamp,
    chat::ToolCall, config::ConfidenceConfig,
};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Mocks — one per port trait.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MockClock {
    fixed: Timestamp,
}

impl Default for MockClock {
    fn default() -> Self {
        Self {
            fixed: Timestamp::from_unix_secs(0).expect("epoch"),
        }
    }
}

impl Clock for MockClock {
    fn now(&self) -> Timestamp {
        self.fixed
    }
}

/// Mock `IdGenerator` that hands out deterministic, monotonically-increasing
/// session ids. Mirrors the per-port `Mock*` pattern so `IdGenerator` has the
/// same shape coverage as every other port in this file.
struct CountingIdGenerator {
    counter: Arc<AtomicUsize>,
}

impl CountingIdGenerator {
    fn new() -> Self {
        Self {
            counter: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl IdGenerator for CountingIdGenerator {
    fn new_session_id(&self) -> SessionId {
        let n = self.counter.fetch_add(1, Ordering::SeqCst) as u64;
        // 12 hex chars, zero-padded — matches the canonical `sess_<12 hex>`
        // shape that `SessionId::from_raw` validates.
        SessionId::from_raw(&format!("sess_{:012x}", n)).expect("well-formed mock id")
    }
}

struct CountingEmbeddingProvider {
    call_count: Arc<AtomicUsize>,
}

impl CountingEmbeddingProvider {
    fn new() -> Self {
        Self {
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl EmbeddingProvider for CountingEmbeddingProvider {
    async fn embed(&self, _text: &str) -> Result<Option<Vec<f32>>, ProviderError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(Some(vec![0.0, 1.0]))
    }
}

struct StubRerankProvider;

impl RerankProvider for StubRerankProvider {
    async fn rerank(
        &self,
        _query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<RerankResult>, ProviderError> {
        Ok(documents
            .iter()
            .take(top_k)
            .enumerate()
            .map(|(i, d)| RerankResult {
                index: i,
                score: 1.0 - i as f32 * 0.1,
                document: d.clone(),
            })
            .collect())
    }
}

struct StubNliClassifier;

impl NliClassifier for StubNliClassifier {
    async fn classify(
        &self,
        _premise: &str,
        _hypothesis: &str,
    ) -> Result<NliResult, ProviderError> {
        Ok(NliResult {
            label: smos_domain::enums::NliLabel::Entailment,
            scores: NliScores {
                entailment: 0.9,
                neutral: 0.05,
                contradiction: 0.05,
            },
            available: true,
        })
    }
}

struct StubLlmExtractor;

impl LlmExtractor for StubLlmExtractor {
    async fn extract_facts(
        &self,
        content: &str,
        _tool_calls: &[ToolCall],
    ) -> Result<Vec<String>, ProviderError> {
        Ok(vec![content.to_string()])
    }
}

struct StubLlmUpstream;

impl LlmUpstream for StubLlmUpstream {
    async fn complete(&self, _request: ChatRequest) -> Result<ChatResponse, UpstreamError> {
        Ok(ChatResponse::NonStreaming(
            serde_json::json!({"choices": []}),
        ))
    }
}

struct StubFactRepository;

impl FactRepository for StubFactRepository {
    async fn save(&self, _fact: &Fact) -> Result<(), RepoError> {
        Ok(())
    }
    async fn get(&self, _id: &FactId, _memory_key: &MemoryKey) -> Result<Option<Fact>, RepoError> {
        Ok(None)
    }
    async fn list_accepted(&self, _memory_key: &MemoryKey) -> Result<Vec<Fact>, RepoError> {
        Ok(Vec::new())
    }
    async fn list_pending(&self, _memory_key: &MemoryKey) -> Result<Vec<Fact>, RepoError> {
        Ok(Vec::new())
    }
    async fn list_memory_keys_for_session(
        &self,
        _session_id: &SessionId,
    ) -> Result<Vec<MemoryKey>, RepoError> {
        Ok(Vec::new())
    }
    async fn search_similar(
        &self,
        _embedding: Vec<f32>,
        _memory_key: &MemoryKey,
        _limit: usize,
    ) -> Result<Vec<SearchHit>, RepoError> {
        Ok(Vec::new())
    }
    async fn update_heat_batch(
        &self,
        _ids: &[FactId],
        _memory_key: &MemoryKey,
        _heat_base: Heat,
        _last_access: Timestamp,
    ) -> Result<(), RepoError> {
        Ok(())
    }
}

struct StubSessionRepository;

impl SessionRepository for StubSessionRepository {
    async fn get_or_create(
        &self,
        id: &SessionId,
        memory_key: &MemoryKey,
    ) -> Result<SessionState, RepoError> {
        Ok(SessionState::new(
            id.clone(),
            memory_key.clone(),
            Timestamp::from_unix_secs(0).expect("ts"),
        ))
    }
    async fn collect_expired(
        &self,
        _timeout: Duration,
    ) -> Result<Vec<(SessionId, SessionState)>, RepoError> {
        Ok(Vec::new())
    }
    async fn snapshot_all(&self) -> Result<Vec<(SessionId, SessionState)>, RepoError> {
        Ok(Vec::new())
    }
    async fn add_pending(&self, _id: &SessionId, _fact_ids: &[FactId]) -> Result<(), RepoError> {
        Ok(())
    }
    async fn remove_pending_owned(
        &self,
        _id: &SessionId,
        _owned: &[FactId],
    ) -> Result<(), RepoError> {
        Ok(())
    }
    async fn clear_session(&self, _id: &SessionId) -> Result<(), RepoError> {
        Ok(())
    }
    async fn dedup_and_mark(
        &self,
        _id: &SessionId,
        _memory_key: &MemoryKey,
        candidate_ids: &[FactId],
    ) -> Result<Vec<FactId>, RepoError> {
        Ok(candidate_ids.to_vec())
    }
    async fn save(&self, _id: &SessionId, _state: &SessionState) -> Result<(), RepoError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Generic-dispatch tests — if the trait shape compiles, the test passes.
// ---------------------------------------------------------------------------

async fn use_embedding_provider<T: EmbeddingProvider>(p: &T) -> Vec<Option<Vec<f32>>> {
    p.embed_batch(&["a", "b", "c"]).await.expect("batch")
}

#[tokio::test]
async fn embedding_provider_default_embed_batch_loops_embed_per_call() {
    let provider = CountingEmbeddingProvider::new();
    let out = use_embedding_provider(&provider).await;
    assert_eq!(out.len(), 3);
    // The default `embed_batch` impl must call `embed` once per text.
    assert_eq!(provider.call_count.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn rerank_provider_returns_top_k_in_order() {
    let p = StubRerankProvider;
    let docs = vec!["d1".to_string(), "d2".to_string(), "d3".to_string()];
    let out = p.rerank("q", &docs, 2).await.expect("rerank");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].index, 0);
    assert_eq!(out[1].index, 1);
    assert!(out[0].score >= out[1].score);
}

#[tokio::test]
async fn nli_classifier_returns_verdict() {
    let c = StubNliClassifier;
    let r = c.classify("a", "b").await.expect("classify");
    assert!(r.available);
    assert_eq!(r.label, smos_domain::enums::NliLabel::Entailment);
}

#[tokio::test]
async fn llm_extractor_returns_extracted_strings() {
    let e = StubLlmExtractor;
    let out = e.extract_facts("hello", &[]).await.expect("extract");
    assert_eq!(out, vec!["hello".to_string()]);
}

#[tokio::test]
async fn llm_upstream_returns_non_streaming_value() {
    let u = StubLlmUpstream;
    let resp = u
        .complete(ChatRequest::new("m", Vec::new()))
        .await
        .expect("complete");
    match resp {
        ChatResponse::NonStreaming(v) => assert!(v.get("choices").is_some()),
        ChatResponse::Streaming(_) => panic!("expected NonStreaming"),
    }
}

#[tokio::test]
async fn clock_port_returns_fixed_time() {
    let clock = MockClock {
        fixed: Timestamp::from_unix_secs(1_234_567_890).expect("ts"),
    };
    assert_eq!(clock.now().as_unix_secs(), 1_234_567_890);
}

#[tokio::test]
async fn fact_repository_stub_implements_all_methods() {
    let repo = StubFactRepository;
    let mk = MemoryKey::shared();
    let id = FactId::from_content("a");
    // Every method must be callable without panicking.
    repo.save(
        &Fact::new_pending(
            "a",
            mk.clone(),
            SessionId::from_raw("sess_abcdef012345").unwrap(),
            smos_domain::Embedding::new(vec![1.0]).unwrap(),
            Timestamp::from_unix_secs(0).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap(),
    )
    .await
    .expect("save");
    assert!(repo.get(&id, &mk).await.unwrap().is_none());
    assert!(repo.list_accepted(&mk).await.unwrap().is_empty());
    assert!(repo.list_pending(&mk).await.unwrap().is_empty());
    assert!(
        repo.search_similar(vec![1.0], &mk, 10)
            .await
            .unwrap()
            .is_empty()
    );
    repo.update_heat_batch(
        &[],
        &mk,
        Heat::new(0.5).unwrap(),
        Timestamp::from_unix_secs(1).unwrap(),
    )
    .await
    .expect("heat");
}

#[tokio::test]
async fn session_repository_stub_implements_all_methods() {
    let repo = StubSessionRepository;
    let id = SessionId::from_raw("sess_abcdef012345").unwrap();
    let mk = MemoryKey::shared();
    let state = repo.get_or_create(&id, &mk).await.expect("goc");
    assert_eq!(state.id(), &id);

    assert!(
        repo.collect_expired(Duration::from_secs(60))
            .await
            .unwrap()
            .is_empty()
    );
    assert!(repo.snapshot_all().await.unwrap().is_empty());

    let candidates = vec![FactId::from_content("a"), FactId::from_content("b")];
    let new = repo
        .dedup_and_mark(&id, &mk, &candidates)
        .await
        .expect("dedup");
    assert_eq!(new.len(), 2);

    repo.add_pending(&id, &candidates).await.expect("add");
    repo.remove_pending_owned(&id, &candidates)
        .await
        .expect("remove");
    repo.save(&id, &state).await.expect("save");
    repo.clear_session(&id).await.expect("clear");
}

// ---------------------------------------------------------------------------
// Use-case-style composition test — proves ports compose into a higher-level
// function via generic dispatch (this is how slice 7 use cases will look).
// ---------------------------------------------------------------------------

async fn retrieve_and_dedup<R, S>(
    facts: &R,
    sessions: &S,
    embedding: Vec<f32>,
    memory_key: &MemoryKey,
    session_id: &SessionId,
) -> Result<Vec<FactId>, RepoError>
where
    R: FactRepository + ?Sized,
    S: SessionRepository + ?Sized,
{
    let hits = facts
        .search_similar(embedding, memory_key, 10)
        .await?
        .into_iter()
        .map(|h| h.id)
        .collect::<Vec<_>>();
    sessions.dedup_and_mark(session_id, memory_key, &hits).await
}

#[tokio::test]
async fn ports_compose_into_higher_level_use_case_via_generic_dispatch() {
    let facts = StubFactRepository;
    let sessions = StubSessionRepository;
    let mk = MemoryKey::shared();
    let sid = SessionId::from_raw("sess_abcdef012345").unwrap();
    let new = retrieve_and_dedup(&facts, &sessions, vec![1.0], &mk, &sid)
        .await
        .expect("compose");
    assert!(new.is_empty(), "stub facts return no hits → empty dedup");
}

#[tokio::test]
async fn merge_result_type_compiles_and_carries_data() {
    let r = MergeResult {
        merged: false,
        reason: smos_domain::enums::MergeReason::NoCandidate,
        merged_fact: None,
        nli_result: None,
    };
    assert!(!r.merged);
    assert!(r.merged_fact.is_none());
}

// ---------------------------------------------------------------------------
// IdGenerator — the only port whose primary method is synchronous. Verify it
// (a) implements against a mock, (b) composes via generic dispatch, and
// (c) yields the canonical `sess_<12 hex>` shape so the marker round-trip
// still works.
// ---------------------------------------------------------------------------

fn use_id_generator<G: IdGenerator>(generator: &G) -> SessionId {
    generator.new_session_id()
}

#[test]
fn id_generator_stub_implements_port_and_yields_canonical_shape() {
    let generator = CountingIdGenerator::new();
    let id = use_id_generator(&generator);
    let s = id.as_str();
    assert!(s.starts_with("sess_"));
    let hex = &s["sess_".len()..];
    assert_eq!(hex.len(), 12);
    assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    // The id round-trips through the marker pipeline (`to_marker` +
    // `detect_in_text` is the production path that re-discovers the id from
    // the trailing comment).
    let marker = id.to_marker();
    let recovered = smos_application::helpers::session_marker::detect_in_text(&marker);
    assert_eq!(recovered.as_ref(), Some(&id));
}
