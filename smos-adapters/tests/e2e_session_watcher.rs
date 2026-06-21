//! E2E tests for `SessionWatcher`.
//!
//! Each test spins up an isolated in-process SurrealDB RocksDB instance,
//! seeds sessions with predatated `last_active` (so `collect_expired` picks
//! them up without `tokio::time::pause`), starts the watcher via
//! `tokio::spawn(watcher.into_loop(rx))`, and asserts on the persisted state
//! through the public port traits. The classifier is a mock so no external
//! NLI dependency is required.
//!
//! The watcher's `inflight` guard, graceful-loop continuation, and graceful
//! shutdown drain are all exercised here. See `e2e_finalize.rs` for the
//! classifier mock + SurrealStore fixture lineage — this file reuses the
//! same patterns (constant / unit / blend embeddings, scripted verdicts)
//! without re-testing the finalize pipeline itself.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use smos_adapters::SessionWatcher;
use smos_adapters::SurrealStore;
use smos_adapters::SystemClock;
use smos_adapters::config::{ServerConfig, SessionConfig};
use smos_application::errors::ProviderError;
use smos_application::ports::{Clock, FactRepository, NliClassifier, SessionRepository};
use smos_application::types::NliResult;
use smos_domain::config::{ConfidenceConfig, MergeConfig, NliConfig};
use smos_domain::enums::NliLabel;
use smos_domain::{
    Confidence, Embedding, Fact, FactId, FactStatus, MemoryKey, NliScores, SessionId, SessionState,
    Timestamp,
};
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Mock NLI classifier (mirrors `e2e_finalize.rs`, plus concurrency tracking)
// ---------------------------------------------------------------------------

type NliMatcher = Arc<dyn Fn(&str, &str) -> Result<NliResult, ProviderError> + Send + Sync>;

struct MockNliClassifier {
    matcher: NliMatcher,
    call_count: Arc<AtomicUsize>,
    /// High-water mark of concurrently in-flight `classify` calls. Stays at 1
    /// when the watcher's per-session serialisation holds; climbs above 1 if
    /// a future refactor parallelises the same session id by accident.
    high_water: Arc<AtomicUsize>,
    current: Arc<AtomicUsize>,
}

impl MockNliClassifier {
    fn matching<F>(matcher: F) -> Self
    where
        F: Fn(&str, &str) -> Result<NliResult, ProviderError> + Send + Sync + 'static,
    {
        Self {
            matcher: Arc::new(matcher),
            call_count: Arc::new(AtomicUsize::new(0)),
            high_water: Arc::new(AtomicUsize::new(0)),
            current: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn constant(verdict: NliResult) -> Self {
        Self::matching(move |_p, _h| Ok(verdict.clone()))
    }

    fn always_unavailable() -> Self {
        Self::matching(|_p, _h| Err(ProviderError::Unavailable("mock offline".into())))
    }
}

impl NliClassifier for MockNliClassifier {
    async fn classify(&self, premise: &str, hypothesis: &str) -> Result<NliResult, ProviderError> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let cur = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        // Update high-water mark; relaxed is fine — we only read it after the
        // watcher is fully drained.
        let mut prev = self.high_water.load(Ordering::Relaxed);
        while cur > prev {
            match self.high_water.compare_exchange_weak(
                prev,
                cur,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => prev = actual,
            }
        }
        let result = (self.matcher)(premise, hypothesis);
        self.current.fetch_sub(1, Ordering::SeqCst);
        result
    }
}

// ---------------------------------------------------------------------------
// Verdict builders
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const EMBED_DIM: usize = 1024;

/// Fresh isolated store against a tempdir-backed RocksDB instance.
async fn fresh_store(test_name: &str) -> (SurrealStore, TempDir) {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join(test_name);
    let db = Surreal::new::<RocksDb>(path.to_string_lossy().to_string())
        .await
        .expect("rocksdb");
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

fn unit_embedding(axis: usize) -> Embedding {
    let mut v = vec![0.0_f32; EMBED_DIM];
    v[axis] = 1.0;
    Embedding::new(v).expect("embedding")
}

fn constant_embedding(value: f32) -> Embedding {
    Embedding::new(vec![value; EMBED_DIM]).expect("embedding")
}

/// Pending fact (status `Pending`, single-source provenance, base confidence).
fn pending_fact(content: &str, embedding: Embedding, session: SessionId) -> Fact {
    Fact::new_pending(
        content,
        memory_key(),
        session,
        embedding,
        SystemClock.now(),
        smos_domain::config::ConfidenceConfig::default().base,
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
        SystemClock.now(),
        smos_domain::config::ConfidenceConfig::default().base,
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

/// Seed a pending fact and register its id on `session`'s pending list.
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

/// Seed a session row with `last_active` predatated by `age` so
/// `collect_expired` reports it without `tokio::time::pause`. The session
/// carries the supplied pending fact ids on its pending list.
async fn seed_session_aged(
    store: &SurrealStore,
    session: &SessionId,
    pending: Vec<FactId>,
    age: Duration,
) {
    let now_secs = SystemClock.now().as_unix_secs();
    let aged_secs = now_secs - age.as_secs() as i64;
    let aged_ts = Timestamp::from_unix_secs(aged_secs).expect("aged ts");
    let state = SessionState::rehydrate(
        session.clone(),
        memory_key(),
        std::iter::empty(),
        pending,
        aged_ts,
        aged_ts,
    )
    .expect("valid: created_at == last_active");
    SessionRepository::save(store, session, &state)
        .await
        .expect("save session");
}

/// Build a `SessionWatcher` against the supplied store + classifier using a
/// fast-scan test config (`scan_interval_seconds = 1`, real session timeout).
fn build_watcher(
    store: SurrealStore,
    classifier: MockNliClassifier,
) -> SessionWatcher<SurrealStore, SurrealStore, MockNliClassifier> {
    SessionWatcher::new(
        store.clone(),
        store,
        classifier,
        Arc::new(ConfidenceConfig::default()),
        Arc::new(NliConfig::default()),
        Arc::new(MergeConfig::default()),
        Arc::new(SessionConfig {
            scan_interval_seconds: 1,
            timeout_seconds: 1800,
            pending_overflow_threshold: 20,
        }),
        Arc::new(ServerConfig::default()),
    )
}

/// Read a fact from the store; panics if absent.
async fn get_fact(store: &SurrealStore, id: &FactId) -> Fact {
    FactRepository::get(store, id, &memory_key())
        .await
        .expect("get")
        .expect("fact present")
}

/// Read a session's current pending count (0 when the session row is gone).
async fn pending_count(store: &SurrealStore, session: &SessionId) -> usize {
    SessionRepository::snapshot_all(store)
        .await
        .expect("snapshot_all")
        .into_iter()
        .find(|(id, _)| id == session)
        .map(|(_, s)| s.pending_facts().len())
        .unwrap_or(0)
}

// ===========================================================================
// 1. Watcher triggers FinalizeSession on an expired session
// ===========================================================================

#[tokio::test]
async fn watcher_triggers_finalize_on_expired_session() {
    let (store, _tmp) = fresh_store("expired_triggers").await;
    let session = sid(1);

    // Pending fact with an orthogonal embedding → standalone promotion path
    // (no merge candidate). The classifier is configured but should NOT be
    // called — the absence of a candidate short-circuits the scan.
    let pending_id = seed_pending(
        &store,
        &session,
        "the user prefers rust over go",
        unit_embedding(7),
        sid(1),
    )
    .await;
    seed_session_aged(
        &store,
        &session,
        vec![pending_id.clone()],
        Duration::from_secs(31 * 60),
    )
    .await;

    let classifier = MockNliClassifier::constant(neutral());
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    // One scan interval (1 s) + slack for the finalize round-trip.
    tokio::time::sleep(Duration::from_millis(1500)).await;

    // Stop the watcher before assertions so it does not race the test exit.
    let _ = tx.send(()).await;
    let _ = handle.await;

    let finalized = get_fact(&store, &pending_id).await;
    assert_eq!(
        finalized.status(),
        FactStatus::Pending,
        "single-source standalone keeps Pending (validation gate below 0.7), \
         but the watcher must still have drained the pending list"
    );
    assert_eq!(
        pending_count(&store, &session).await,
        0,
        "watcher must have removed the pending fact from the session bookkeeping"
    );
}

// ===========================================================================
// 2. Watcher skips active sessions (last_active recent)
// ===========================================================================

#[tokio::test]
async fn watcher_skips_active_sessions() {
    let (store, _tmp) = fresh_store("active_skipped").await;
    let session = sid(1);

    let pending_id = seed_pending(
        &store,
        &session,
        "an active fact that must not be touched",
        unit_embedding(11),
        sid(1),
    )
    .await;
    // `last_active = now` (no aging) — `collect_expired` returns nothing.
    seed_session_aged(&store, &session, vec![pending_id.clone()], Duration::ZERO).await;

    let classifier = MockNliClassifier::constant(neutral());
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    tokio::time::sleep(Duration::from_millis(1500)).await;

    // Assert BEFORE the shutdown signal — `drain_all` deliberately
    // processes every still-tracked session regardless of expiry (§12), so
    // the post-shutdown state would show the pending list drained even
    // though the periodic scan correctly skipped the active session.
    let fact = get_fact(&store, &pending_id).await;
    assert_eq!(
        fact.status(),
        FactStatus::Pending,
        "active session pending facts must be left untouched by the scan"
    );
    assert_eq!(
        pending_count(&store, &session).await,
        1,
        "active session pending bookkeeping must be preserved by the scan"
    );

    let _ = tx.send(()).await;
    let _ = handle.await;
}

// ===========================================================================
// 3. Overflow trigger fires without waiting for the inactivity timeout
// ===========================================================================

#[tokio::test]
async fn watcher_triggers_on_overflow() {
    let (store, _tmp) = fresh_store("overflow_triggers").await;
    let session = sid(1);

    // Seed 25 pending facts (above the 20-fact overflow threshold). Each
    // gets a distinct orthogonal embedding so none of them find a merge
    // candidate — finalize promotes each one standalone.
    let mut pending_ids = Vec::with_capacity(25);
    for axis in 0..25usize {
        let content = format!("overflow pending fact number {axis}");
        let id = seed_pending(&store, &session, &content, unit_embedding(axis), sid(1)).await;
        pending_ids.push(id);
    }
    // `last_active = now` — overflow is the ONLY trigger here.
    seed_session_aged(&store, &session, pending_ids.clone(), Duration::ZERO).await;

    let classifier = MockNliClassifier::constant(neutral());
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    tokio::time::sleep(Duration::from_millis(1500)).await;
    let _ = tx.send(()).await;
    let _ = handle.await;

    assert_eq!(
        pending_count(&store, &session).await,
        0,
        "overflow must drain every pending fact from the session"
    );
}

// ===========================================================================
// 4. inflight guard — high-water mark of concurrent classify calls
// ===========================================================================
//
// NOTE on coverage: in the current single-task design the per-cycle scan
// awaits each `try_finalize` sequentially, so concurrent finalize calls on
// the same session id are structurally impossible. The `inflight` guard in
// `session_watcher.rs` is therefore defensive — it exists to keep the
// per-session serialisation invariant intact if a future refactor
// parallelises the scan (mirroring the POC's documented concurrency seam).
//
// This test pins the OBSERVABLE consequence of the invariant — the
// classifier never sees more than one in-flight call for the same session
// id — so a refactor that breaks the invariant surfaces as a high-water
// mark > 1. It is NOT a behaviour regression test of the guard itself; the
// guard's correctness cannot be exercised without concurrent finalize
// paths that the current design deliberately avoids.

#[tokio::test]
async fn watcher_inflight_guard_holds_high_water_mark_at_one() {
    let (store, _tmp) = fresh_store("inflight_guard").await;
    let session = sid(1);

    // One pending fact + one accepted candidate so the classifier is
    // consulted (otherwise the standalone path bypasses NLI). The session
    // is seeded as expired; the scan visits it exactly once per cycle.
    seed_accepted(
        &store,
        "shared anchor fact",
        constant_embedding(0.5),
        sid(2),
    )
    .await;
    let mut pending_ids = Vec::with_capacity(20);
    for i in 0..20usize {
        let content = format!("shared anchor fact twin number {i}");
        let id = seed_pending(&store, &session, &content, constant_embedding(0.5), sid(1)).await;
        pending_ids.push(id);
    }
    seed_session_aged(&store, &session, pending_ids, Duration::from_secs(31 * 60)).await;

    let classifier = MockNliClassifier::constant(neutral());
    let high_water = classifier.high_water.clone();
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    tokio::time::sleep(Duration::from_millis(1500)).await;
    let _ = tx.send(()).await;
    let _ = handle.await;

    assert_eq!(
        high_water.load(Ordering::SeqCst),
        1,
        "high-water mark of concurrent classifier calls must stay at 1 — \
         breaks if a future refactor parallelises the per-cycle scan without \
         preserving the per-session serialisation invariant"
    );
}

// ===========================================================================
// 5. Watcher loop continues after a cycle error (graceful-loop property)
// ===========================================================================

#[tokio::test]
async fn watcher_graceful_loop_continues_on_error() {
    let (store, _tmp) = fresh_store("graceful_loop").await;
    let session = sid(1);

    // Pending fact + accepted candidate → classifier IS consulted, so we can
    // make it fail and observe the watcher's reaction.
    seed_accepted(
        &store,
        "shared anchor fact",
        constant_embedding(0.5),
        sid(2),
    )
    .await;
    let pending_id = seed_pending(
        &store,
        &session,
        "shared anchor fact paraphrase",
        constant_embedding(0.5),
        sid(1),
    )
    .await;
    seed_session_aged(
        &store,
        &session,
        vec![pending_id.clone()],
        Duration::from_secs(31 * 60),
    )
    .await;

    // Always-unavailable classifier: every finalize attempt fails-open at
    // the use-case layer (Pending stays Pending) but the watcher itself must
    // keep cycling. A real sidecar outage is the production analogue.
    let classifier = MockNliClassifier::always_unavailable();
    let call_count = classifier.call_count.clone();
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    tokio::time::sleep(Duration::from_millis(2500)).await;

    assert!(
        !handle.is_finished(),
        "watcher must still be running after every cycle failed — graceful \
         loop property"
    );
    assert!(
        call_count.load(Ordering::SeqCst) > 0,
        "watcher must have attempted finalize at least once"
    );

    let _ = tx.send(()).await;
    let _ = handle.await;

    let fact = get_fact(&store, &pending_id).await;
    assert_eq!(
        fact.status(),
        FactStatus::Pending,
        "graceful degradation: an unavailable sidecar leaves the pending fact \
         Pending for the next cycle"
    );
}

// ===========================================================================
// 6. Graceful shutdown drains every still-tracked session
// ===========================================================================

#[tokio::test]
async fn watcher_graceful_shutdown_drains_all() {
    let (store, _tmp) = fresh_store("shutdown_drain").await;

    // Three sessions: one expired, two active (recent `last_active`). The
    // drain bypasses the expiry check so all three must be finalized.
    let mut pending_specs = Vec::new();
    for n in 1..=3u8 {
        let session = sid(n);
        let id = seed_pending(
            &store,
            &session,
            &format!("drain candidate fact for session {n}"),
            unit_embedding(100 + n as usize),
            sid(n),
        )
        .await;
        // Half are aged (expired), half are recent (active). The drain must
        // process both kinds.
        let age = if n == 1 {
            Duration::from_secs(31 * 60)
        } else {
            Duration::ZERO
        };
        seed_session_aged(&store, &session, vec![id.clone()], age).await;
        pending_specs.push((session, id));
    }

    let classifier = MockNliClassifier::constant(neutral());
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    // Send the shutdown immediately — drain-all runs on receipt.
    let _ = tx.send(()).await;
    let _ = handle.await;

    for (session, _id) in &pending_specs {
        assert_eq!(
            pending_count(&store, session).await,
            0,
            "drain_all must process session {} regardless of expiry state",
            session
        );
    }
}

// ===========================================================================
// 7. Watcher processes multiple expired sessions in one cycle
// ===========================================================================

#[tokio::test]
async fn watcher_processes_multiple_expired_sessions() {
    let (store, _tmp) = fresh_store("multi_expired").await;

    let mut sessions = Vec::with_capacity(5);
    for n in 1..=5u8 {
        let session = sid(n);
        let id = seed_pending(
            &store,
            &session,
            &format!("multi-expired fact for session {n}"),
            unit_embedding(200 + n as usize),
            sid(n),
        )
        .await;
        seed_session_aged(
            &store,
            &session,
            vec![id.clone()],
            Duration::from_secs(31 * 60),
        )
        .await;
        sessions.push((session, id));
    }

    let classifier = MockNliClassifier::constant(neutral());
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    tokio::time::sleep(Duration::from_millis(1500)).await;
    let _ = tx.send(()).await;
    let _ = handle.await;

    for (session, _id) in &sessions {
        assert_eq!(
            pending_count(&store, session).await,
            0,
            "every expired session must be drained in the same cycle"
        );
    }
}

// ===========================================================================
// 8. Watcher is a no-op when the session store is empty
// ===========================================================================

#[tokio::test]
async fn watcher_no_op_when_no_expired_sessions() {
    let (store, _tmp) = fresh_store("no_op_empty").await;

    let classifier = MockNliClassifier::constant(neutral());
    let call_count = classifier.call_count.clone();
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    tokio::time::sleep(Duration::from_millis(1500)).await;

    assert!(
        !handle.is_finished(),
        "watcher must still be running — an empty store is a normal state, \
         not a termination condition"
    );
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        0,
        "no sessions means no finalize attempts means no classifier calls"
    );

    let _ = tx.send(()).await;
    let _ = handle.await;
}

// ===========================================================================
// 9. Watcher stops cleanly when the shutdown sender is dropped
// ===========================================================================

#[tokio::test]
async fn watcher_stops_when_shutdown_sender_dropped() {
    let (store, _tmp) = fresh_store("drop_sender_stops").await;

    let classifier = MockNliClassifier::constant(neutral());
    let watcher = build_watcher(store, classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    // Dropping the last sender makes `Receiver::recv` return `None`, which
    // the `tokio::select!` arm treats as a shutdown signal. Mirrors an
    // abrupt component drop in the main binary.
    drop(tx);

    // The watcher must exit on its own (drain-all of an empty store is
    // fast). Bounded by `tokio::time::timeout` so a regression that hangs
    // the loop fails the test instead of the suite.
    tokio::time::timeout(Duration::from_secs(2), handle)
        .await
        .expect("watcher must terminate within 2 s of the sender dropping")
        .expect("watcher task must not panic");
}

// ===========================================================================
// 10. Cycle continues across sessions when one finalize fails mid-cycle
// ===========================================================================

#[tokio::test]
async fn watcher_continues_after_a_per_session_failure() {
    let (store, _tmp) = fresh_store("per_session_failure").await;

    // Three expired sessions, each with one pending fact + a shared accepted
    // anchor so the classifier IS consulted. The matcher fails for session
    // `sess_2` (the middle one) and succeeds for the other two — exercises
    // the watcher's "swallow per-session errors, keep scanning the rest of
    // the cycle" contract. The POC sweeper relies on the same property:
    // a single sidecar timeout must not block every other session's drain.
    let anchor_id = seed_accepted(
        &store,
        "shared anchor fact",
        constant_embedding(0.5),
        sid(99),
    )
    .await;

    let mut sessions_with_pending = Vec::new();
    for n in 1..=3u8 {
        let session = sid(n);
        let pending_id = seed_pending(
            &store,
            &session,
            &format!("shared anchor fact twin from session {n}"),
            constant_embedding(0.5),
            sid(n),
        )
        .await;
        seed_session_aged(
            &store,
            &session,
            vec![pending_id.clone()],
            Duration::from_secs(31 * 60),
        )
        .await;
        sessions_with_pending.push((session, pending_id));
    }

    let classifier = MockNliClassifier::matching(|_premise, hypothesis| {
        // The matcher keys on the hypothesis text (the pending fact's
        // content) so the order in which the watcher happens to scan
        // sessions does not change which one fails.
        if hypothesis.contains("from session 2") {
            Err(ProviderError::Unavailable("transient outage".into()))
        } else {
            Ok(neutral())
        }
    });
    let watcher = build_watcher(store.clone(), classifier);
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    tokio::time::sleep(Duration::from_millis(1500)).await;
    let _ = tx.send(()).await;
    let _ = handle.await;

    let (s1, p1) = &sessions_with_pending[0];
    let (s2, _p2) = &sessions_with_pending[1];
    let (s3, p3) = &sessions_with_pending[2];

    // Sessions 1 and 3: classifier succeeded → finalize completed → pending
    // bookkeeping cleared (standalone promotion; statuses are the validation
    // gate's call and are NOT asserted here).
    assert_eq!(
        pending_count(&store, s1).await,
        0,
        "session 1 must be drained — its classifier call succeeded"
    );
    assert_eq!(
        pending_count(&store, s3).await,
        0,
        "session 3 must be drained — its classifier call succeeded"
    );
    // Session 2: classifier failed → finalize graceful-degrades → owned
    // pending id is still removed from the bookkeeping (FinalizeSession
    // always runs the bookkeeping cleanup step), so the count is also 0.
    // The check below is the regression guard: a single failure inside a
    // cycle must not block the cycle from reaching sessions scanned later.
    assert_eq!(
        pending_count(&store, s2).await,
        0,
        "session 2 bookkeeping must be cleared even though its classifier \
         call failed — proves the cycle continued past the failure"
    );

    // The anchor fact must still be present and Accepted — finalize merges
    // absorb the twins only on entailment, and our matcher returned Neutral.
    let anchor = get_fact(&store, &anchor_id).await;
    assert_eq!(anchor.status(), FactStatus::Accepted);

    // Pending twins of sessions 1 and 3 went through standalone promotion
    // (Neutral verdict → no merge). Validation gate kept them `Pending` at
    // the default confidence — that is the use case's contract, not the
    // watcher's, so the check is a smoke check, not a behaviour assertion.
    let _p1_after = get_fact(&store, p1).await;
    let _p3_after = get_fact(&store, p3).await;
}

// ===========================================================================
// 11. Bounded drain — a wedged finalize breaks the drain within the budget
// ===========================================================================
//
// Exercises the COMMON-shutdown bound: `drain_all` runs against a TOTAL
// budget (== `shutdown_extraction_grace_seconds`, the same knob
// `ExtractionSupervisor::drain` uses). A wedged sidecar / pathological
// pending backlog cannot pin the process on Ctrl+C: the budget is
// exhausted, the loop breaks, and the remaining sessions' pending facts
// stay pending for the next process start.

/// Classifier whose every call hangs forever — models a wedged sidecar.
struct HangingClassifier;

impl NliClassifier for HangingClassifier {
    async fn classify(
        &self,
        _premise: &str,
        _hypothesis: &str,
    ) -> Result<NliResult, ProviderError> {
        std::future::pending::<()>().await;
        unreachable!("pending future never resolves")
    }
}

#[tokio::test]
async fn watcher_drain_stops_within_budget_when_finalize_hangs() {
    let (store, _tmp) = fresh_store("bounded_drain").await;
    let session = sid(1);

    // One expired session whose finalize would invoke the classifier (it
    // has an accepted merge candidate). The classifier hangs, so the
    // finalize never returns on its own.
    seed_accepted(
        &store,
        "shared anchor fact",
        constant_embedding(0.5),
        sid(2),
    )
    .await;
    let pending_id = seed_pending(
        &store,
        &session,
        "shared anchor fact paraphrase",
        constant_embedding(0.5),
        sid(1),
    )
    .await;
    seed_session_aged(
        &store,
        &session,
        vec![pending_id.clone()],
        Duration::from_secs(31 * 60),
    )
    .await;

    // Tight 1-second budget — well below the default 30 s, so the test
    // resolves in seconds instead of minutes. With N=1 session the TOTAL
    // and per-session semantics coincide; the assertion still proves the
    // drain terminates, but does NOT by itself distinguish the two
    // semantics. The semantic distinction (N×grace vs grace) is a
    // docstring + code-reading contract, not a behaviour test.
    let server_cfg = ServerConfig {
        shutdown_extraction_grace_seconds: 1,
        ..ServerConfig::default()
    };
    let watcher = SessionWatcher::new(
        store.clone(),
        store.clone(),
        HangingClassifier,
        Arc::new(ConfidenceConfig::default()),
        Arc::new(NliConfig::default()),
        Arc::new(MergeConfig::default()),
        Arc::new(SessionConfig {
            scan_interval_seconds: 60,
            timeout_seconds: 1800,
            pending_overflow_threshold: 20,
        }),
        Arc::new(server_cfg),
    );
    let (tx, rx) = tokio::sync::mpsc::channel::<()>(1);
    let handle = tokio::spawn(watcher.into_loop(rx));

    // Send shutdown immediately. drain_all runs, hits the hanging
    // classifier, the TOTAL budget fires after 1 s, the drain completes.
    // Bounded by 5 s overall so a regression that drops the bound fails
    // the test instead of hanging it.
    let _ = tx.send(()).await;
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("drain_all must terminate within 5 s even when a finalize hangs")
        .expect("watcher task must not panic");

    // The hanging finalize was cancelled mid-flight (the wedge point is
    // the classify call inside `resolve_one`, before any `facts.save`), so
    // no state was committed: the pending fact stays Pending AND the
    // session bookkeeping still references it (the remove_pending_owned
    // cleanup runs at the end of FinalizeSession::execute, which never
    // reached it). Pending-stays-pending is the durability invariant.
    let fact = get_fact(&store, &pending_id).await;
    assert_eq!(
        fact.status(),
        FactStatus::Pending,
        "the wedged finalize was cancelled before it could commit any state"
    );
    assert_eq!(
        pending_count(&store, &session).await,
        1,
        "the wedged finalize was cancelled before the bookkeeping cleanup"
    );
}
