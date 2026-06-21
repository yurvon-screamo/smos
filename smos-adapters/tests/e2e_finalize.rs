//! E2E tests for `FinalizeSession` against a real `SurrealStore`.
//!
//! Each test spins up an isolated in-process SurrealDB RocksDB instance,
//! seeds accepted + pending facts, runs the use case with a deterministic
//! mock NLI classifier, then asserts on the persisted state through the
//! public port traits. The mock lets us exercise every code path
//! (entailment / contradiction / neutral / C3 guard / exact match / drift
//! walk / graceful degradation) without depending on a Python runtime.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use smos_adapters::SurrealStore;
use smos_application::errors::ProviderError;
use smos_application::ports::{FactRepository, NliClassifier, SessionRepository};
use smos_application::types::NliResult;
use smos_application::use_cases::{FinalizeSession, FinalizeStats};
use smos_domain::config::{ConfidenceConfig, MergeConfig, NliConfig};
use smos_domain::enums::NliLabel;
use smos_domain::{
    Confidence, Embedding, Fact, FactId, FactStatus, MemoryKey, NliScores, SessionId, Timestamp,
};
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Mock NLI classifier
// ---------------------------------------------------------------------------

/// Pluggable NLI mock. The matcher closure keys verdicts on the hypothesis
/// text so tests are stable regardless of the order in which the use case
/// happens to scan candidates (the underlying SurrealQL order is not
/// guaranteed across versions).
type NliMatcher = Arc<dyn Fn(&str, &str) -> Result<NliResult, ProviderError> + Send + Sync>;

struct MockNliClassifier {
    matcher: NliMatcher,
    call_count: Arc<AtomicUsize>,
}

impl MockNliClassifier {
    fn matching<F>(matcher: F) -> Self
    where
        F: Fn(&str, &str) -> Result<NliResult, ProviderError> + Send + Sync + 'static,
    {
        Self {
            matcher: Arc::new(matcher),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Always succeed with the same verdict (no per-pair logic).
    fn constant(verdict: NliResult) -> Self {
        Self::matching(move |_premise, _hypothesis| Ok(verdict.clone()))
    }

    /// Always fail with `Unavailable` — exercises graceful degradation.
    fn always_unavailable() -> Self {
        Self::matching(|_p, _h| Err(ProviderError::Unavailable("mock offline".into())))
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl NliClassifier for MockNliClassifier {
    async fn classify(&self, premise: &str, hypothesis: &str) -> Result<NliResult, ProviderError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        (self.matcher)(premise, hypothesis)
    }
}

// ---------------------------------------------------------------------------
// Verdict builders
// ---------------------------------------------------------------------------

fn entailment() -> NliResult {
    NliResult {
        label: NliLabel::Entailment,
        scores: NliScores {
            entailment: 0.92,
            neutral: 0.06,
            contradiction: 0.02,
        },
        available: true,
    }
}

fn neutral() -> NliResult {
    NliResult {
        label: NliLabel::Neutral,
        scores: NliScores {
            entailment: 0.15,
            neutral: 0.75,
            contradiction: 0.10,
        },
        available: true,
    }
}

fn contradiction() -> NliResult {
    NliResult {
        label: NliLabel::Contradiction,
        scores: NliScores {
            entailment: 0.03,
            neutral: 0.12,
            contradiction: 0.85,
        },
        available: true,
    }
}

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/// Build a fresh isolated store against a tempdir-backed RocksDB instance.
async fn fresh_store(test_name: &str) -> (SurrealStore, TempDir) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join(test_name);
    let path_str = path.to_string_lossy().to_string();
    let db = Surreal::new::<RocksDb>(path_str)
        .await
        .expect("rocksdb connect");
    db.use_ns("test").use_db("test").await.expect("use ns/db");
    let store = SurrealStore::from_client(db);
    store.run_migrations().await.expect("migrations");
    (store, tmp)
}

fn memory_key() -> MemoryKey {
    MemoryKey::from_raw("origa").expect("memory key")
}

fn sid(n: u8) -> SessionId {
    SessionId::from_raw(&format!("sess_{:012x}", n as u64)).expect("session id")
}

fn ts() -> Timestamp {
    Timestamp::from_unix_secs(1_700_000_000).expect("timestamp")
}

const EMBED_DIM: usize = 1024;

/// Unit-norm embedding with `1.0` at `axis`, zero elsewhere.
fn unit_embedding(axis: usize) -> Embedding {
    let mut v = vec![0.0_f32; EMBED_DIM];
    v[axis] = 1.0;
    Embedding::new(v).expect("embedding")
}

/// Constant embedding (every dim set to `value`); cosine against another
/// constant embedding with the same value is 1.0 — useful for "above the
/// merge threshold" pairs without pinning a specific axis.
fn constant_embedding(value: f32) -> Embedding {
    Embedding::new(vec![value; EMBED_DIM]).expect("embedding")
}

/// Two-axis blend embedding — `1.0` at axis 0 plus `b` at axis 1, zero
/// elsewhere. Cosine against another blend with the same `b` is 1.0;
/// against a blend with a different `b` the cosine is `1 / sqrt(1 + b²)`
/// (when comparing against axis-0-only) or
/// `(1 + b1*b2) / (sqrt(1+b1²) * sqrt(1+b2²))` between two blends. Used by
/// drift-priority tests that need to control cosine ordering between
/// candidates (constant embeddings cannot do this — they are all parallel).
fn blend_embedding(b: f32) -> Embedding {
    let mut v = vec![0.0_f32; EMBED_DIM];
    v[0] = 1.0;
    v[1] = b;
    Embedding::new(v).expect("embedding")
}

/// Pending fact (status `Pending`, single-source provenance, base confidence).
fn pending_fact(content: &str, embedding: Embedding, session: SessionId) -> Fact {
    Fact::new_pending(
        content,
        memory_key(),
        session,
        embedding,
        ts(),
        ConfidenceConfig::default().base,
    )
    .expect("pending fact")
}

/// Accepted fact lifted above the accept threshold via `set_status_and_confidence`.
fn accepted_fact(content: &str, embedding: Embedding, session: SessionId) -> Fact {
    let mut f = Fact::new_pending(
        content,
        memory_key(),
        session,
        embedding,
        ts(),
        ConfidenceConfig::default().base,
    )
    .expect("pending");
    f.set_status_and_confidence(
        FactStatus::Accepted,
        Confidence::new(0.9).expect("confidence"),
        &ConfidenceConfig::default(),
    )
    .expect("accept");
    f
}

/// Accepted fact whose provenance already carries two sessions. Used by
/// multi-source confidence tests to assert that merge unions grow the
/// distinct session count past the bonus threshold.
fn accepted_fact_with_sessions(content: &str, embedding: Embedding, sessions: &[u8]) -> Fact {
    let mut f = accepted_fact(content, embedding, sid(sessions[0]));
    for &n in &sessions[1..] {
        f.confirm_cross_session(&sid(n), &ConfidenceConfig::default())
            .expect("confirm");
    }
    f
}

/// Seed a fact and register its id on the session's pending list.
async fn seed_pending(
    store: &SurrealStore,
    session: &SessionId,
    content: &str,
    embedding: Embedding,
    src_session: SessionId,
) -> FactId {
    let fact = pending_fact(content, embedding, src_session);
    let id = fact.id().clone();
    FactRepository::save(store, &fact)
        .await
        .expect("save pending");
    SessionRepository::add_pending(store, session, std::slice::from_ref(&id))
        .await
        .expect("add_pending");
    id
}

/// Seed an accepted fact (not on any pending list).
async fn seed_accepted(
    store: &SurrealStore,
    content: &str,
    embedding: Embedding,
    src_session: SessionId,
) -> FactId {
    let fact = accepted_fact(content, embedding, src_session);
    let id = fact.id().clone();
    FactRepository::save(store, &fact)
        .await
        .expect("save accepted");
    id
}

/// Ensure the session row exists before finalize runs.
async fn ensure_session(store: &SurrealStore, session: &SessionId) {
    let _ = SessionRepository::get_or_create(store, session, &memory_key())
        .await
        .expect("get_or_create");
}

fn build<'a>(
    facts: &'a SurrealStore,
    sessions: &'a SurrealStore,
    classifier: &'a MockNliClassifier,
    confidence_cfg: &'a ConfidenceConfig,
    nli_cfg: &'a NliConfig,
    merge_cfg: &'a MergeConfig,
) -> FinalizeSession<'a, SurrealStore, SurrealStore, MockNliClassifier> {
    FinalizeSession {
        facts,
        sessions,
        classifier,
        confidence_cfg,
        nli_cfg,
        merge_cfg,
    }
}

/// Bundles the three configs the use case borrows so tests do not litter
/// each function with three extra locals.
struct Cfgs {
    confidence: ConfidenceConfig,
    nli: NliConfig,
    merge: MergeConfig,
}
impl Cfgs {
    fn new() -> Self {
        Self {
            confidence: ConfidenceConfig::default(),
            nli: NliConfig::default(),
            merge: MergeConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Entailment merges a pending fact into an existing accepted one
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_entailment_merges_pair() {
    let (store, _tmp) = fresh_store("entailment_merges").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    let existing_id = seed_accepted(
        &store,
        "ttl=10 prevents the token refresh loop",
        constant_embedding(0.5),
        sid(2),
    )
    .await;
    // Pending twin: same constant embedding → cosine 1.0 (≥ 0.85 threshold).
    let pending_id = seed_pending(
        &store,
        &session,
        "ttl=10 stops the refresh loop",
        constant_embedding(0.5),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::constant(entailment());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.merged, 1);
    assert_eq!(stats.finalized, 0);
    assert_eq!(stats.conflicts, 0);
    assert_eq!(stats.rejected, 1);

    // Existing fact: provenance grew (two sessions), still Accepted.
    let merged = FactRepository::get(&store, &existing_id, &memory_key())
        .await
        .expect("get")
        .expect("existing present");
    assert!(merged.source_sessions().distinct_count() >= 2);
    assert_eq!(merged.status(), FactStatus::Accepted);
    // Pending twin: Rejected.
    let twin = FactRepository::get(&store, &pending_id, &memory_key())
        .await
        .expect("get")
        .expect("pending present");
    assert_eq!(twin.status(), FactStatus::Rejected);
}

// ---------------------------------------------------------------------------
// 2. Contradiction flags the pair bidirectionally, statuses unchanged
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_contradiction_flags_conflict() {
    let (store, _tmp) = fresh_store("contradiction_flags").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    let existing_id = seed_accepted(
        &store,
        "ttl is sixty seconds",
        constant_embedding(0.5),
        sid(2),
    )
    .await;
    let pending_id = seed_pending(
        &store,
        &session,
        "ttl is ten seconds",
        constant_embedding(0.5),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::constant(contradiction());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.conflicts, 1);
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.finalized, 0);

    let existing_after = FactRepository::get(&store, &existing_id, &memory_key())
        .await
        .expect("get")
        .expect("existing present");
    let pending_after = FactRepository::get(&store, &pending_id, &memory_key())
        .await
        .expect("get")
        .expect("pending present");
    assert!(existing_after.conflicts_with().contains(&pending_id));
    assert!(pending_after.conflicts_with().contains(&existing_id));
    // Status UNCHANGED on both sides.
    assert_eq!(existing_after.status(), FactStatus::Accepted);
    assert_eq!(pending_after.status(), FactStatus::Pending);
    // No tombstone on either side — drift is not a death.
    assert!(existing_after.valid_until().is_none());
    assert!(pending_after.valid_until().is_none());
}

// ---------------------------------------------------------------------------
// 3. Standalone promotion when there is no merge candidate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_standalone_promotes_when_no_candidate() {
    let (store, _tmp) = fresh_store("standalone_promotes").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // Pending fact with a unique orthogonal embedding → no cosine candidate.
    let pending_id = seed_pending(
        &store,
        &session,
        "the user prefers rust over go",
        unit_embedding(7),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::constant(neutral());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.finalized, 1);
    // Single-source standalone: base confidence 0.5 → validation gate keeps
    // it Pending (below the 0.7 accept threshold). The finalize counter
    // still records the resolution; status is the validation gate's call.
    let finalized = FactRepository::get(&store, &pending_id, &memory_key())
        .await
        .expect("get")
        .expect("present");
    assert_eq!(finalized.status(), FactStatus::Pending);
    assert!(
        classifier.call_count() == 0,
        "no NLI call without a candidate"
    );
}

// ---------------------------------------------------------------------------
// 4. Exact-match short-circuit does NOT call the sidecar
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_exact_match_skips_sidecar() {
    let (store, _tmp) = fresh_store("exact_match_skips").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    let existing_id = seed_accepted(
        &store,
        "identical anchor fact",
        constant_embedding(0.5),
        sid(2),
    )
    .await;
    // Pending twin: same embedding (cosine 1.0) AND case-different content
    // so the exact-match normalisation fires but the FactIds still differ.
    let _pending_id = seed_pending(
        &store,
        &session,
        "IDENTICAL ANCHOR FACT",
        constant_embedding(0.5),
        sid(1),
    )
    .await;

    // The matcher would have returned contradiction — proving the
    // exact-match short-circuit skipped it entirely.
    let classifier = MockNliClassifier::constant(contradiction());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.merged, 1, "exact-match yields entailment → merge");
    assert_eq!(stats.conflicts, 0);
    assert_eq!(
        classifier.call_count(),
        0,
        "exact-match must short-circuit before any sidecar call"
    );

    let merged = FactRepository::get(&store, &existing_id, &memory_key())
        .await
        .expect("get")
        .expect("existing present");
    assert!(
        merged.source_sessions().distinct_count() >= 2,
        "provenance grew despite the short-circuit"
    );
}

// ---------------------------------------------------------------------------
// 5. C3 guard — already-flagged pairs skip the sidecar
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_c3_guard_skips_flagged_pairs() {
    let (store, _tmp) = fresh_store("c3_guard").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // Pre-flag the pair so the C3 guard fires before any sidecar call.
    let mut existing = accepted_fact("ttl is sixty seconds", constant_embedding(0.5), sid(2));
    let mut pending_fact_initial =
        pending_fact("ttl is ten seconds", constant_embedding(0.5), sid(1));
    existing
        .flag_conflict(pending_fact_initial.id().clone())
        .expect("flag existing");
    pending_fact_initial
        .flag_conflict(existing.id().clone())
        .expect("flag pending");
    let existing_id = existing.id().clone();
    let pending_id = pending_fact_initial.id().clone();
    FactRepository::save(&store, &existing)
        .await
        .expect("save existing");
    FactRepository::save(&store, &pending_fact_initial)
        .await
        .expect("save pending");
    SessionRepository::add_pending(&store, &session, std::slice::from_ref(&pending_id))
        .await
        .expect("add_pending");

    let classifier = MockNliClassifier::constant(contradiction());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    // The C3 guard skipped the only candidate → standalone promotion (the
    // conflict was already recorded, no new drift to flag).
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.finalized, 1);
    assert_eq!(stats.conflicts, 0);
    assert_eq!(
        classifier.call_count(),
        0,
        "C3 guard must skip every sidecar call"
    );

    // Existing flags unchanged — no double-flag.
    let existing_after = FactRepository::get(&store, &existing_id, &memory_key())
        .await
        .expect("get")
        .expect("present");
    assert_eq!(existing_after.conflicts_with().len(), 1);
    assert!(existing_after.conflicts_with().contains(&pending_id));
}

// ---------------------------------------------------------------------------
// 6. Drift-priority walk — contradiction beats an earlier neutral hit
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_drift_priority_walk() {
    let (store, _tmp) = fresh_store("drift_priority").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // Two accepted candidates with DISTINCT cosine against the pending fact:
    //   - "similar" has b=0.05 → cosine(pending, similar) ≈ 0.999 (top of
    //     the scan). Matcher returns Neutral.
    //   - "drift"   has b=0.50 → cosine(pending, drift)   ≈ 0.935 (further
    //     down the scan). Matcher returns Contradiction.
    // Without drift-priority, the top neutral would mask the contradiction.
    let similar_id =
        seed_accepted(&store, "rust is memory safe", blend_embedding(0.05), sid(2)).await;
    let drift_id = seed_accepted(
        &store,
        "rust leaks memory everywhere",
        blend_embedding(0.50),
        sid(3),
    )
    .await;
    let pending_id = seed_pending(
        &store,
        &session,
        "rust is memory safe language",
        blend_embedding(0.05),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::matching(|premise, _hypothesis| match premise {
        "rust is memory safe" => Ok(neutral()),
        "rust leaks memory everywhere" => Ok(contradiction()),
        other => Err(ProviderError::InvalidResponse(format!(
            "unexpected: {other}"
        ))),
    });
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(
        stats.conflicts, 1,
        "drift must win over the earlier neutral"
    );
    assert_eq!(stats.merged, 0);

    let pending_after = FactRepository::get(&store, &pending_id, &memory_key())
        .await
        .expect("get")
        .expect("present");
    assert!(
        pending_after.conflicts_with().contains(&drift_id),
        "drift flag points to the contradicting candidate"
    );
    assert!(
        !pending_after.conflicts_with().contains(&similar_id),
        "no spurious flag on the neutral candidate"
    );
}

// ---------------------------------------------------------------------------
// 7. no_contradiction_bonus is applied on the standalone path
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_no_contradiction_bonus_applied_on_standalone_with_candidate() {
    let (store, _tmp) = fresh_store("no_contradiction_bonus").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // Existing fact whose only NLI verdict is Neutral — the pending fact
    // gets the no_contradiction_bonus (single source: 0.5 + 0.1 = 0.6, still
    // Pending because < 0.7 accept threshold, but the bonus is observable
    // as the confidence delta vs the no-candidate standalone case).
    let _existing_id = seed_accepted(
        &store,
        "rust is memory safe",
        constant_embedding(0.6),
        sid(2),
    )
    .await;
    let pending_id = seed_pending(
        &store,
        &session,
        "rust is memory safe language",
        constant_embedding(0.6),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::constant(neutral());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.finalized, 1);

    let finalized = FactRepository::get(&store, &pending_id, &memory_key())
        .await
        .expect("get")
        .expect("present");
    // 0.5 base + 0.1 no_contradiction_bonus = 0.6 — Pending but boosted.
    assert!(
        (finalized.confidence().value() - 0.6).abs() < 1e-5,
        "no_contradiction_bonus must lift the confidence from 0.5 to 0.6, got {}",
        finalized.confidence().value()
    );
}

// ---------------------------------------------------------------------------
// 8. Sidecar unavailable — graceful degradation, fact stays pending
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_sidecar_unavailable_graceful() {
    let (store, _tmp) = fresh_store("sidecar_unavailable").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // Three pending facts, each with a candidate (so the sidecar would have
    // been consulted). All three must stay Pending — graceful degradation.
    let contents = [
        "first pending fact with a candidate",
        "second pending fact with a candidate",
        "third pending fact with a candidate",
    ];
    let mut pending_ids = Vec::new();
    for content in &contents {
        pending_ids
            .push(seed_pending(&store, &session, content, constant_embedding(0.5), sid(1)).await);
    }
    seed_accepted(
        &store,
        "first pending fact with a candidate twin",
        constant_embedding(0.5),
        sid(2),
    )
    .await;

    let classifier = MockNliClassifier::always_unavailable();
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc
        .execute(&session, &memory_key())
        .await
        .expect("must not raise");
    // No outcome tallied for any of the three.
    assert_eq!(stats.finalized, 0);
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.conflicts, 0);

    for id in &pending_ids {
        let still_pending = FactRepository::get(&store, id, &memory_key())
            .await
            .expect("get")
            .expect("present");
        assert_eq!(still_pending.status(), FactStatus::Pending);
        assert!(still_pending.conflicts_with().is_empty());
    }
    assert_eq!(
        classifier.call_count(),
        3,
        "every candidate pair must still drive one sidecar attempt"
    );
}

// ---------------------------------------------------------------------------
// 9. Batch continues after a single pair failure
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_batch_continues_on_single_error() {
    let (store, _tmp) = fresh_store("batch_continues").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // Three pending facts:
    //   - p1: candidate against "anchor" → Unavailable (skip pair, stay pending)
    //   - p2: candidate against "anchor" → Entailment (merge)
    //   - p3: orthogonal → standalone finalize
    seed_accepted(
        &store,
        "shared anchor fact here",
        constant_embedding(0.5),
        sid(2),
    )
    .await;
    let p1_id = seed_pending(
        &store,
        &session,
        "shared anchor fact here too",
        constant_embedding(0.5),
        sid(1),
    )
    .await;
    let p2_id = seed_pending(
        &store,
        &session,
        "shared anchor fact but longer",
        constant_embedding(0.5),
        sid(1),
    )
    .await;
    let p3_id = seed_pending(
        &store,
        &session,
        "totally unrelated pending fact",
        unit_embedding(50),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::matching(|_premise, hypothesis| match hypothesis {
        "shared anchor fact here too" => Err(ProviderError::Unavailable("transient".into())),
        "shared anchor fact but longer" => Ok(entailment()),
        other => Err(ProviderError::InvalidResponse(format!(
            "unexpected: {other}"
        ))),
    });
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.processed, 3);
    assert_eq!(stats.merged, 1);
    assert_eq!(stats.finalized, 1);

    let p1_after = FactRepository::get(&store, &p1_id, &memory_key())
        .await
        .expect("get")
        .expect("p1 present");
    assert_eq!(p1_after.status(), FactStatus::Pending, "p1 stayed pending");

    let p2_after = FactRepository::get(&store, &p2_id, &memory_key())
        .await
        .expect("get")
        .expect("p2 present");
    assert_eq!(
        p2_after.status(),
        FactStatus::Rejected,
        "p2 merged → rejected"
    );

    // p3 standalone: single source, base confidence (no candidate) → Pending.
    let p3_after = FactRepository::get(&store, &p3_id, &memory_key())
        .await
        .expect("get")
        .expect("p3 present");
    assert_eq!(p3_after.status(), FactStatus::Pending);
}

// ---------------------------------------------------------------------------
// 10. Session bookkeeping — owned pending ids cleared after finalize
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_clears_session_pending() {
    let (store, _tmp) = fresh_store("clears_pending").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // Three standalone pending facts (orthogonal embeddings → no candidate).
    for axis in 0..3usize {
        let content = format!("standalone fact number {axis}");
        seed_pending(&store, &session, &content, unit_embedding(axis), sid(1)).await;
    }

    let classifier = MockNliClassifier::constant(neutral());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.processed, 3);
    assert_eq!(stats.finalized, 3);

    // After finalize, the session's pending list is empty (all owned ids
    // drained). The facts themselves stay in the store, just reclassified.
    let state = SessionRepository::get_or_create(&store, &session, &memory_key())
        .await
        .expect("session");
    assert!(
        state.pending_facts().is_empty(),
        "owned pending ids cleared; got {:?}",
        state.pending_facts()
    );
}

// ---------------------------------------------------------------------------
// 11. Multi-source confidence — merge unions source sessions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_multi_source_confidence_after_merge() {
    let (store, _tmp) = fresh_store("multi_source_confidence").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // Existing accepted fact with two sessions of provenance (already above
    // the multi-source bonus threshold). The merge will union the pending
    // fact's session into it, growing the distinct count from 2 to 3.
    let existing =
        accepted_fact_with_sessions("shared anchor fact", constant_embedding(0.5), &[2, 3]);
    let existing_id = existing.id().clone();
    FactRepository::save(&store, &existing)
        .await
        .expect("save existing");

    let _pending_id = seed_pending(
        &store,
        &session,
        "shared anchor fact paraphrase",
        constant_embedding(0.5),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::constant(entailment());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.merged, 1);

    let merged = FactRepository::get(&store, &existing_id, &memory_key())
        .await
        .expect("get")
        .expect("present");
    assert!(
        merged.source_sessions().distinct_count() >= 3,
        "merge must union the pending twin's session into the existing provenance"
    );
    assert!(
        merged.source_sessions().contains(&sid(1)),
        "the pending twin's session must land on the merged fact"
    );
}

// ---------------------------------------------------------------------------
// 12. Full pipeline smoke — three outcomes in one finalize run
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_full_pipeline_three_outcomes() {
    let (store, _tmp) = fresh_store("full_pipeline").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // 1. Entailment merge target.
    seed_accepted(
        &store,
        "shared anchor fact",
        constant_embedding(0.5),
        sid(2),
    )
    .await;
    let _merge_pending = seed_pending(
        &store,
        &session,
        "shared anchor fact paraphrase",
        constant_embedding(0.5),
        sid(1),
    )
    .await;

    // 2. Contradiction pair (blend embedding distinct from the entailment
    //    pair so the candidates do not collapse into one row).
    seed_accepted(&store, "rust leaks memory", blend_embedding(0.4), sid(3)).await;
    let _drift_pending = seed_pending(
        &store,
        &session,
        "rust guarantees memory safety forever",
        blend_embedding(0.4),
        sid(1),
    )
    .await;

    // 3. Standalone promotion (no candidate).
    let _standalone_pending = seed_pending(
        &store,
        &session,
        "the user prefers rust over go",
        unit_embedding(99),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::matching(|premise, _hypothesis| match premise {
        "shared anchor fact" => Ok(entailment()),
        "rust leaks memory" => Ok(contradiction()),
        other => Err(ProviderError::InvalidResponse(format!(
            "unexpected: {other}"
        ))),
    });
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.processed, 3);
    assert_eq!(stats.merged, 1);
    assert_eq!(stats.conflicts, 1);
    assert_eq!(stats.finalized, 1);
    assert_eq!(stats.rejected, 1);
}

// ---------------------------------------------------------------------------
// 13. Unknown session returns empty stats
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_unknown_session_returns_empty_stats() {
    let (store, _tmp) = fresh_store("unknown_session").await;
    // No session row created.
    let classifier = MockNliClassifier::constant(entailment());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&sid(99), &memory_key()).await.expect("Ok");
    assert_eq!(stats.processed, 0);
    assert_eq!(stats.finalized, 0);
    assert_eq!(stats.session_id, sid(99).as_str());
}

// ---------------------------------------------------------------------------
// 13b. Regression: missing SessionState must NOT mask pending facts.
// ---------------------------------------------------------------------------
//
// Mirrors the operator-facing bug: HTTP extraction persists only
// `fact.source_sessions` — it never writes the SessionState row. The previous
// implementation read `SessionState.pending_facts()` for ownership, so a
// session with 24 pending facts in the store but no SessionState row was
// reported as "nothing to do". FinalizeSession now derives ownership from
// `source_sessions`, so the missing row must not affect the drain.

#[tokio::test]
async fn finalize_drains_pending_facts_even_when_session_state_is_absent() {
    let (store, _tmp) = fresh_store("missing_session_state_drains").await;
    let session = sid(1);
    // Deliberately DO NOT call `ensure_session(&store, &session)` —
    // simulate the HTTP request path that never persists SessionState.

    // Seed two pending facts whose `source_sessions = [sid(1)]` (the
    // `seed_pending` helper wires `src_session` onto the fact's
    // provenance list). Skip the SessionRepository::add_pending call so
    // the SessionState row stays absent.
    let fact_a = pending_fact(
        "first fact extracted via http without session row",
        unit_embedding(11),
        sid(1),
    );
    let id_a = fact_a.id().clone();
    FactRepository::save(&store, &fact_a)
        .await
        .expect("save fact_a");

    let fact_b = pending_fact(
        "second fact extracted via http without session row",
        unit_embedding(12),
        sid(1),
    );
    let id_b = fact_b.id().clone();
    FactRepository::save(&store, &fact_b)
        .await
        .expect("save fact_b");

    let classifier = MockNliClassifier::constant(neutral());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(
        stats.processed, 2,
        "missing SessionState must not mask the two pending facts"
    );
    assert_eq!(stats.finalized, 2);

    // Both facts survived the drain (status is the validation gate's call;
    // the regression assertion is that they were processed at all).
    let a_after = FactRepository::get(&store, &id_a, &memory_key())
        .await
        .expect("get a")
        .expect("fact_a present");
    assert!(a_after.source_sessions().contains(&sid(1)));
    let b_after = FactRepository::get(&store, &id_b, &memory_key())
        .await
        .expect("get b")
        .expect("fact_b present");
    assert!(b_after.source_sessions().contains(&sid(1)));

    // Cross-check: SessionState is STILL absent — finalize did not
    // silently create one. The store snapshot must be empty.
    let snapshot = SessionRepository::snapshot_all(&store)
        .await
        .expect("snapshot_all");
    assert!(
        snapshot.iter().all(|(id, _)| id != &session),
        "finalize must not create a SessionState row as a side effect"
    );
}

// ---------------------------------------------------------------------------
// 13c. Cross-namespace discovery via list_memory_keys_for_session
// ---------------------------------------------------------------------------
//
// Exercises the CLI discovery fallback: a session whose facts are spread
// across multiple memory_keys is fully drainable when the operator does not
// pass `--memory-key`. The new FactRepository port method surfaces every
// distinct memory_key whose facts reference the session.

#[tokio::test]
async fn list_memory_keys_for_session_returns_distinct_keys() {
    let (store, _tmp) = fresh_store("memory_keys_discovery").await;
    let session = sid(1);

    let other_key = MemoryKey::from_raw("other-namespace").expect("memory key");

    // Two pending facts under the canonical memory_key, one under a second
    // namespace. All three carry the same session in source_sessions.
    let mk_a = pending_fact("canonical-ns fact one", unit_embedding(1), sid(1));
    let mk_b = pending_fact("canonical-ns fact two", unit_embedding(2), sid(1));
    let other_mk_fact = Fact::new_pending(
        "other-ns fact",
        other_key.clone(),
        sid(1),
        unit_embedding(3),
        ts(),
        ConfidenceConfig::default().base,
    )
    .expect("pending");
    FactRepository::save(&store, &mk_a).await.expect("save a");
    FactRepository::save(&store, &mk_b).await.expect("save b");
    FactRepository::save(&store, &other_mk_fact)
        .await
        .expect("save other");

    // Sanity: a fact under the same memory_key but a DIFFERENT session
    // must NOT contribute its memory_key to the discovery result for
    // `session`.
    let other_session_fact = Fact::new_pending(
        "unrelated-session fact in third namespace",
        MemoryKey::from_raw("third-namespace").expect("mk"),
        sid(2),
        unit_embedding(4),
        ts(),
        ConfidenceConfig::default().base,
    )
    .expect("pending");
    FactRepository::save(&store, &other_session_fact)
        .await
        .expect("save unrelated");

    let keys = FactRepository::list_memory_keys_for_session(&store, &session)
        .await
        .expect("list_memory_keys_for_session");

    // The canonical memory_key and the other-namespace must both surface;
    // the third namespace (which only references sid(2)) must NOT.
    assert_eq!(
        keys.len(),
        2,
        "expected exactly 2 distinct keys, got {keys:?}"
    );
    assert!(
        keys.iter().any(|k| k.as_str() == memory_key().as_str()),
        "canonical memory_key must be discovered, got {keys:?}"
    );
    assert!(
        keys.iter().any(|k| k.as_str() == "other-namespace"),
        "other-namespace must be discovered, got {keys:?}"
    );
    assert!(
        !keys.iter().any(|k| k.as_str() == "third-namespace"),
        "third-namespace references sid(2), must NOT be discovered for sid(1)"
    );
}

// ---------------------------------------------------------------------------
// 14. Pending-only session (no accepted pool) — every fact finalized standalone
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_with_empty_accepted_pool_finalizes_every_pending_standalone() {
    let (store, _tmp) = fresh_store("empty_accepted_pool").await;
    let session = sid(1);
    ensure_session(&store, &session).await;

    // No accepted facts. Two pending facts with distinct embeddings → no
    // candidate scan can produce a match → every fact finalized standalone.
    seed_pending(
        &store,
        &session,
        "first standalone fact",
        unit_embedding(1),
        sid(1),
    )
    .await;
    seed_pending(
        &store,
        &session,
        "second standalone fact",
        unit_embedding(2),
        sid(1),
    )
    .await;

    let classifier = MockNliClassifier::constant(entailment());
    let cfgs = Cfgs::new();
    let uc = build(
        &store,
        &store,
        &classifier,
        &cfgs.confidence,
        &cfgs.nli,
        &cfgs.merge,
    );

    let stats = uc.execute(&session, &memory_key()).await.expect("finalize");
    assert_eq!(stats.processed, 2);
    assert_eq!(stats.finalized, 2);
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.conflicts, 0);
    assert_eq!(classifier.call_count(), 0, "no NLI call without candidates");
}

// ---------------------------------------------------------------------------
// 15. FinalizeStats fields surface to operators
// ---------------------------------------------------------------------------

#[tokio::test]
async fn finalize_stats_default_is_zeroed_and_session_id_round_trips() {
    let stats = FinalizeStats {
        session_id: "sess_roundtrip".into(),
        ..FinalizeStats::default()
    };
    assert_eq!(stats.session_id, "sess_roundtrip");
    assert_eq!(stats.processed, 0);
    assert_eq!(stats.finalized, 0);
    assert_eq!(stats.merged, 0);
    assert_eq!(stats.conflicts, 0);
    assert_eq!(stats.rejected, 0);
}
