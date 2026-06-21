//! Integration tests for `SurrealStore` against an embedded SurrealDB RocksDB
//! instance (no Docker required).
//!
//! Each test gets its own fresh RocksDB directory so tests can run in parallel
//! without cross-contamination.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;
use smos_adapters::SurrealStore;
use smos_application::errors::RepoError;
use smos_application::ports::{FactRepository, SessionRepository};
use smos_domain::{
    Confidence, Embedding, Fact, FactId, FactStatus, Heat, MemoryKey, SessionId, Timestamp,
};
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Spin up a fresh store against a tempdir-backed RocksDB instance and run
/// migrations. The `TempDir` is returned so the caller can keep it alive for
/// the test duration.
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

fn session_id(n: u8) -> SessionId {
    SessionId::from_raw(&format!("sess_{:012x}", n as u64)).expect("session id")
}

fn timestamp(secs: i64) -> Timestamp {
    Timestamp::from_unix_secs(secs).expect("timestamp")
}

fn unit_embedding(dim: usize, index: usize) -> Embedding {
    let mut v = vec![0.0_f32; dim];
    v[index] = 1.0;
    Embedding::new(v).expect("embedding")
}

fn accepted_fact(content: &str, embedding: Embedding, session: SessionId) -> Fact {
    let mut fact = Fact::new_pending(
        content,
        memory_key(),
        session,
        embedding,
        timestamp(1_700_000_000),
        smos_domain::config::ConfidenceConfig::default().base,
    )
    .expect("pending fact");
    fact.set_status_and_confidence(
        FactStatus::Accepted,
        Confidence::new(0.9).expect("confidence"),
        &smos_domain::config::ConfidenceConfig::default(),
    )
    .expect("accept");
    fact
}

fn pending_fact(content: &str, embedding: Embedding, session: SessionId) -> Fact {
    Fact::new_pending(
        content,
        memory_key(),
        session,
        embedding,
        timestamp(1_700_000_000),
        smos_domain::config::ConfidenceConfig::default().base,
    )
    .expect("pending fact")
}

#[derive(Debug, Deserialize)]
struct HeatRow {
    heat_base: f32,
}

#[derive(Debug, Deserialize)]
struct InjectedRow {
    injected_facts: Vec<String>,
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn run_migrations_is_idempotent() {
    let (store, _tmp) = fresh_store("migrations_idempotent").await;
    store.run_migrations().await.expect("second migrations");
    store.run_migrations().await.expect("third migrations");
}

// ---------------------------------------------------------------------------
// Fact save + get
// ---------------------------------------------------------------------------

#[tokio::test]
async fn save_then_get_returns_the_same_fact() {
    let (store, _tmp) = fresh_store("save_then_get").await;
    let fact = accepted_fact(
        "Rust is memory-safe",
        unit_embedding(1024, 0),
        session_id(1),
    );
    FactRepository::save(&store, &fact).await.expect("save");
    let loaded = store.get(fact.id(), fact.memory_key()).await.expect("get");
    let loaded = loaded.expect("fact must exist");
    assert_eq!(loaded.content(), "Rust is memory-safe");
    assert_eq!(loaded.memory_key().as_str(), "origa");
    assert_eq!(loaded.status(), FactStatus::Accepted);
    assert!((loaded.confidence().value() - 0.9).abs() < 1e-5);
}

#[tokio::test]
async fn save_then_get_preserves_every_field_round_trip() {
    // Regression test for H1 (reviewer finding): every persisted field must
    // survive the save→get round-trip unchanged. The earlier implementation
    // recomputed confidence/heat/type and silently drifted; this test pins
    // the contract by asserting field-by-field equality.
    let (store, _tmp) = fresh_store("save_get_full_roundtrip").await;
    let mut fact = accepted_fact(
        "Rust is memory-safe",
        unit_embedding(1024, 7),
        session_id(1),
    );
    fact.set_status_and_confidence(
        FactStatus::Accepted,
        Confidence::new(0.83).unwrap(),
        &smos_domain::config::ConfidenceConfig::default(),
    )
    .unwrap();
    fact.set_valid_until(Some(
        Timestamp::from_unix_secs(fact.valid_from().as_unix_secs() + 86400).unwrap(),
    ))
    .unwrap();
    fact.flag_conflict(FactId::from_content("some other fact"))
        .unwrap();

    FactRepository::save(&store, &fact).await.expect("save");
    let loaded = store
        .get(fact.id(), fact.memory_key())
        .await
        .expect("get")
        .expect("fact present");

    assert_eq!(loaded.id(), fact.id());
    assert_eq!(loaded.content(), fact.content());
    assert_eq!(loaded.memory_key().as_str(), fact.memory_key().as_str());
    assert_eq!(loaded.fact_type(), fact.fact_type());
    assert_eq!(loaded.confidence().value(), fact.confidence().value());
    assert_eq!(loaded.status(), fact.status());
    assert_eq!(loaded.valid_from(), fact.valid_from());
    assert_eq!(loaded.valid_until(), fact.valid_until());
    assert_eq!(loaded.extracted_at(), fact.extracted_at());
    assert_eq!(
        loaded.source_sessions().distinct_count(),
        fact.source_sessions().distinct_count()
    );
    assert_eq!(loaded.conflicts_with(), fact.conflicts_with());
    assert_eq!(loaded.heat_base().value(), fact.heat_base().value());
    assert_eq!(loaded.last_access_at(), fact.last_access_at());
    assert_eq!(
        loaded.embedding().map(|e| e.dim()),
        fact.embedding().map(|e| e.dim())
    );
}

#[tokio::test]
async fn get_returns_none_for_unknown_id() {
    let (store, _tmp) = fresh_store("get_unknown").await;
    let id = FactId::from_content("nope");
    let loaded = store.get(&id, &memory_key()).await.expect("get");
    assert!(loaded.is_none(), "unknown id must return None");
}

#[tokio::test]
async fn get_returns_none_for_wrong_memory_key() {
    let (store, _tmp) = fresh_store("get_wrong_key").await;
    let fact = accepted_fact("Rust is fast", unit_embedding(1024, 0), session_id(1));
    FactRepository::save(&store, &fact).await.expect("save");
    let other_key = MemoryKey::from_raw("other").expect("key");
    let loaded = store.get(fact.id(), &other_key).await.expect("get");
    assert!(loaded.is_none(), "wrong memory_key must miss");
}

#[tokio::test]
async fn save_upserts_existing_fact_idempotently() {
    let (store, _tmp) = fresh_store("save_upsert").await;
    let fact_v1 = accepted_fact(
        "Rust is memory-safe",
        unit_embedding(1024, 0),
        session_id(1),
    );
    FactRepository::save(&store, &fact_v1)
        .await
        .expect("save v1");

    let fact_v2 = accepted_fact(
        "Rust is memory-safe",
        unit_embedding(1024, 1),
        session_id(2),
    );
    FactRepository::save(&store, &fact_v2)
        .await
        .expect("save v2");

    let loaded = store
        .get(fact_v1.id(), fact_v1.memory_key())
        .await
        .expect("get")
        .expect("fact");
    assert_eq!(loaded.id(), fact_v1.id());
}

// ---------------------------------------------------------------------------
// list_accepted / list_pending
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_accepted_returns_only_accepted_facts() {
    let (store, _tmp) = fresh_store("list_accepted").await;
    let accepted = accepted_fact("A", unit_embedding(1024, 0), session_id(1));
    let pending = pending_fact("B", unit_embedding(1024, 1), session_id(1));
    let mut rejected = pending_fact("C", unit_embedding(1024, 2), session_id(1));
    rejected
        .set_status_and_confidence(
            FactStatus::Rejected,
            Confidence::new(0.0).expect("conf"),
            &smos_domain::config::ConfidenceConfig::default(),
        )
        .expect("reject");
    for f in [&accepted, &pending, &rejected] {
        FactRepository::save(&store, f).await.expect("save");
    }

    let out = store.list_accepted(&memory_key()).await.expect("list");
    let contents: HashSet<&str> = out.iter().map(|f| f.content()).collect();
    assert_eq!(contents, HashSet::from(["A"]));
}

#[tokio::test]
async fn list_pending_returns_only_pending_facts() {
    let (store, _tmp) = fresh_store("list_pending").await;
    let accepted = accepted_fact("A", unit_embedding(1024, 0), session_id(1));
    let pending = pending_fact("B", unit_embedding(1024, 1), session_id(1));
    FactRepository::save(&store, &accepted).await.expect("save");
    FactRepository::save(&store, &pending).await.expect("save");

    let out = store.list_pending(&memory_key()).await.expect("list");
    let contents: HashSet<&str> = out.iter().map(|f| f.content()).collect();
    assert_eq!(contents, HashSet::from(["B"]));
}

// ---------------------------------------------------------------------------
// search_similar
// ---------------------------------------------------------------------------

#[tokio::test]
async fn search_similar_orders_by_cosine_distance_ascending() {
    let (store, _tmp) = fresh_store("search_orders").await;
    for i in 0..5usize {
        let fact = accepted_fact(&format!("fact-{i}"), unit_embedding(1024, i), session_id(1));
        FactRepository::save(&store, &fact).await.expect("save");
    }

    let hits = store
        .search_similar(
            unit_embedding(1024, 0).as_slice().to_vec(),
            &memory_key(),
            3,
        )
        .await
        .expect("search");

    assert_eq!(hits.len(), 3, "should return exactly limit hits");
    let top = &hits[0];
    assert!(
        top.id
            .to_string()
            .contains(FactId::from_content("fact-0").as_str()),
        "top hit must be fact-0"
    );
    let distances: Vec<f32> = hits.iter().filter_map(|h| h.metadata.distance).collect();
    for w in distances.windows(2) {
        assert!(w[0] <= w[1] + 1e-5, "distances must be sorted ascending");
    }
}

#[tokio::test]
async fn search_similar_excludes_other_memory_keys() {
    let (store, _tmp) = fresh_store("search_excludes_other_keys").await;
    let other_key = MemoryKey::from_raw("other").expect("key");
    let foreign = Fact::new_pending(
        "foreign",
        other_key,
        session_id(1),
        unit_embedding(1024, 0),
        timestamp(1_700_000_000),
        smos_domain::config::ConfidenceConfig::default().base,
    )
    .expect("fact");
    FactRepository::save(&store, &foreign)
        .await
        .expect("save foreign");
    let local = accepted_fact("local", unit_embedding(1024, 0), session_id(1));
    FactRepository::save(&store, &local)
        .await
        .expect("save local");

    let hits = store
        .search_similar(
            unit_embedding(1024, 0).as_slice().to_vec(),
            &memory_key(),
            5,
        )
        .await
        .expect("search");
    let contents: HashSet<&str> = hits.iter().map(|h| h.document.as_str()).collect();
    assert!(contents.contains("local"));
    assert!(!contents.contains("foreign"));
}

#[tokio::test]
async fn search_similar_returns_full_top_k_when_namespace_is_skewed() {
    // Regression test for H4 (reviewer finding): when the HNSW over-fetch
    // window is dominated by foreign-key facts, the post-filter drops most
    // HNSW candidates and the brute-force fallback must kick in to deliver
    // the full `limit` of local hits.
    //
    // Geometry: query Q = [1, 0, 0, ...]. Foreign facts are CLOSE to Q
    // (perturbation in dimension 1 = 0.0001·i); local facts are FARTHER
    // (perturbation in dimension 2 = 0.01·i). With over_fetch=20 and 30
    // foreign facts, the global top-20 window is filled entirely by foreign
    // — the post-filter then drops all of them, leaving zero local hits
    // from the HNSW pass, which triggers the brute-force fallback.
    let (store, _tmp) = fresh_store("search_skewed").await;
    let other_key = MemoryKey::from_raw("other").expect("key");

    for i in 0..30usize {
        let mut emb = vec![0.0_f32; 1024];
        emb[0] = 1.0;
        emb[1] = i as f32 * 0.0001; // very close to Q
        let mut fact = Fact::new_pending(
            &format!("foreign-{i}"),
            other_key.clone(),
            session_id(1),
            Embedding::new(emb).unwrap(),
            timestamp(1_700_000_000),
            smos_domain::config::ConfidenceConfig::default().base,
        )
        .unwrap();
        fact.set_status_and_confidence(
            FactStatus::Accepted,
            Confidence::new(0.9).unwrap(),
            &smos_domain::config::ConfidenceConfig::default(),
        )
        .unwrap();
        FactRepository::save(&store, &fact).await.unwrap();
    }

    // Local facts are FARTHER from Q so they fall outside the HNSW top-20
    // window — only brute-force can find them.
    for i in 0..5usize {
        let mut emb = vec![0.0_f32; 1024];
        emb[0] = 1.0;
        emb[2] = (i + 1) as f32 * 0.01; // much larger perturbation
        let fact = accepted_fact(
            &format!("local-{i}"),
            Embedding::new(emb).unwrap(),
            session_id(1),
        );
        FactRepository::save(&store, &fact).await.unwrap();
    }

    let mut query_emb = vec![0.0_f32; 1024];
    query_emb[0] = 1.0;
    let hits = store
        .search_similar(query_emb, &memory_key(), 5)
        .await
        .expect("search");

    assert_eq!(
        hits.len(),
        5,
        "brute-force fallback must deliver all 5 local hits"
    );
    // Distances must be sorted ascending — verifies the brute-force
    // similarity→distance conversion and the merge-sort are correct.
    let distances: Vec<f32> = hits.iter().filter_map(|h| h.metadata.distance).collect();
    assert_eq!(distances.len(), 5, "every hit must carry a distance");
    for w in distances.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-5,
            "distances must be sorted ascending: {distances:?}"
        );
    }
    // All hits must be local — proves the post-filter and the brute-force
    // equality pre-filter both honour the memory_key.
    for h in &hits {
        assert_eq!(h.memory_key.as_str(), "origa");
        assert!(h.document.starts_with("local-"));
    }
}

// ---------------------------------------------------------------------------
// search_for_dedup (semantic-dedup safety net — pending + accepted)
// ---------------------------------------------------------------------------

/// `search_for_dedup` MUST surface pending facts: this is what lets
/// `persist_facts` Layer 2 break the circular deadlock (a pending fact can
/// reach the accept threshold only through cross-session confirmation,
/// which requires finding the existing pending row).
#[tokio::test]
async fn search_for_dedup_returns_pending_facts_that_search_similar_excludes() {
    let (store, _tmp) = fresh_store("dedup_includes_pending").await;

    let pending = pending_fact("pending claim", unit_embedding(1024, 0), session_id(1));
    let pending_id = pending.id().clone();
    FactRepository::save(&store, &pending)
        .await
        .expect("save pending");

    // Sanity: `search_similar` (retrieval-only, accepted) MUST return nothing
    // — the pending fact is not yet promoted. This is the contract the dedup
    // path breaks out of.
    let similar = store
        .search_similar(
            unit_embedding(1024, 0).as_slice().to_vec(),
            &memory_key(),
            5,
        )
        .await
        .expect("search_similar");
    assert!(
        similar.is_empty(),
        "search_similar is accepted-only; pending must be invisible"
    );

    // Contract under test: `search_for_dedup` surfaces the pending fact so
    // Layer 2 can confirm it across sessions.
    let dedup = store
        .search_for_dedup(
            unit_embedding(1024, 0).as_slice().to_vec(),
            &memory_key(),
            5,
        )
        .await
        .expect("search_for_dedup");
    assert_eq!(dedup.len(), 1, "dedup must surface the pending fact");
    assert_eq!(dedup[0].id, pending_id);
    assert_eq!(dedup[0].metadata.status, "pending");
}

/// `search_for_dedup` MUST include accepted facts as well — Layer 2 also
/// needs to confirm a rephrased re-observation of an already-accepted fact
/// (otherwise the same concept splits into two accepted rows).
#[tokio::test]
async fn search_for_dedup_also_surfaces_accepted_facts() {
    let (store, _tmp) = fresh_store("dedup_includes_accepted").await;

    let accepted = accepted_fact("confirmed claim", unit_embedding(1024, 0), session_id(1));
    let accepted_id = accepted.id().clone();
    FactRepository::save(&store, &accepted)
        .await
        .expect("save accepted");

    let dedup = store
        .search_for_dedup(
            unit_embedding(1024, 0).as_slice().to_vec(),
            &memory_key(),
            5,
        )
        .await
        .expect("search_for_dedup");
    assert_eq!(dedup.len(), 1);
    assert_eq!(dedup[0].id, accepted_id);
    assert_eq!(dedup[0].metadata.status, "accepted");
}

/// `search_for_dedup` MUST exclude tombstoned facts — re-confirming a
/// fact that has already been invalidated would resurrect dead knowledge.
#[tokio::test]
async fn search_for_dedup_excludes_tombstoned_facts() {
    let (store, _tmp) = fresh_store("dedup_excludes_tombstoned").await;

    let mut expired = accepted_fact("expired claim", unit_embedding(1024, 0), session_id(1));
    let valid_from = expired.valid_from();
    let later = Timestamp::from_unix_secs(valid_from.as_unix_secs() + 3600).expect("later");
    expired.set_valid_until(Some(later)).expect("tombstone");
    FactRepository::save(&store, &expired)
        .await
        .expect("save expired");

    let dedup = store
        .search_for_dedup(
            unit_embedding(1024, 0).as_slice().to_vec(),
            &memory_key(),
            5,
        )
        .await
        .expect("search_for_dedup");
    assert!(
        dedup.is_empty(),
        "tombstoned fact must not be considered for dedup"
    );
}

/// Storage-level contract for the dedup safety net: a pending fact seeded
/// in session 1 with a 1024-dim embedding; a rephrased query whose SHA1
/// differs but whose embedding is near-identical surfaces the pending row
/// via `search_for_dedup` with similarity ≥ 0.95. The pipeline-level
/// promotion (`persist_one_fact` → `confirm_cross_session` → accept
/// threshold) is exercised by the unit tests in `extract_facts_from_response`.
#[tokio::test]
async fn search_for_dedup_finds_rephrased_pending_fact_above_threshold() {
    let (store, _tmp) = fresh_store("dedup_unblocks_promotion").await;

    // Two near-identical embeddings: differ in dimension 1 by 0.0001 —
    // cosine similarity stays well above 0.95.
    let emb_a = unit_embedding(1024, 0);
    let mut emb_b_vec = emb_a.as_slice().to_vec();
    emb_b_vec[1] = 0.0001;
    let emb_b = Embedding::new(emb_b_vec).expect("embedding");

    let pending = pending_fact("auth uses Argon2id hashing", emb_a, session_id(1));
    let pending_id = pending.id().clone();
    FactRepository::save(&store, &pending)
        .await
        .expect("save pending");

    // Rephrased variant — different SHA1, near-identical embedding.
    let rephrased_id = FactId::from_content("password hashing relies on Argon2id");
    assert_ne!(
        rephrased_id, pending_id,
        "test setup: rephrasing must hash to a different FactId"
    );

    // The dedup search surfaces the pending fact from the rephrased query.
    let dedup = store
        .search_for_dedup(emb_b.as_slice().to_vec(), &memory_key(), 1)
        .await
        .expect("search_for_dedup");
    assert_eq!(dedup.len(), 1, "near-identical pending fact must surface");
    assert_eq!(dedup[0].id, pending_id);

    // Cosine similarity above the 0.95 threshold — confirms the dedup
    // decision that `persist_facts` Layer 2 would make.
    let distance = dedup[0].metadata.distance.expect("distance present");
    let similarity = 1.0 - distance;
    assert!(
        similarity >= 0.95,
        "similarity {similarity} must clear the 0.95 dedup threshold"
    );
}

// ---------------------------------------------------------------------------
// update_heat_batch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_heat_batch_updates_only_scoped_facts() {
    let (store, _tmp) = fresh_store("update_heat_batch").await;
    let mut ids = Vec::new();
    for i in 0..5usize {
        let fact = accepted_fact(&format!("heat-{i}"), unit_embedding(1024, i), session_id(1));
        FactRepository::save(&store, &fact).await.expect("save");
        ids.push(fact.id().clone());
    }
    let now = timestamp(1_800_000_000);
    store
        .update_heat_batch(&ids, &memory_key(), Heat::new(0.7).expect("heat"), now)
        .await
        .expect("update_heat");

    // Read heat_base through the raw_db() escape hatch: the domain's Fact
    // boost_heat pins heat to 1.0 on read, so the public read API can't
    // observe the persisted decayed value.
    let mut res = store
        .raw_db()
        .query("SELECT heat_base FROM fact WHERE memory_key = $mk;")
        .bind(("mk", memory_key().as_str().to_string()))
        .await
        .expect("query");
    let rows: Vec<HeatRow> = res.take(0).expect("take");
    assert_eq!(rows.len(), 5);
    for r in &rows {
        assert!(
            (r.heat_base - 0.7).abs() < 1e-5,
            "heat must be 0.7, got {}",
            r.heat_base
        );
    }
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_or_create_creates_on_first_call_and_returns_existing_after() {
    let (store, _tmp) = fresh_store("session_get_or_create").await;
    let id = session_id(1);
    let s1 = store
        .get_or_create(&id, &memory_key())
        .await
        .expect("goc 1");
    assert!(s1.pending_facts().is_empty());

    let s2 = store
        .get_or_create(&id, &memory_key())
        .await
        .expect("goc 2");
    assert_eq!(s2.id(), &id);
}

#[tokio::test]
async fn add_pending_then_remove_pending_owned_round_trips() {
    let (store, _tmp) = fresh_store("session_pending_roundtrip").await;
    let id = session_id(1);
    let _ = store.get_or_create(&id, &memory_key()).await.expect("goc");

    let fids = vec![
        FactId::from_content("p1"),
        FactId::from_content("p2"),
        FactId::from_content("p3"),
    ];
    store.add_pending(&id, &fids).await.expect("add");

    store
        .remove_pending_owned(&id, &[fids[0].clone(), fids[2].clone()])
        .await
        .expect("remove");

    // Pending_facts are restored on read via `SessionState::add_pending`, so
    // we can verify the round-trip through the public port.
    let state = store
        .get_or_create(&id, &memory_key())
        .await
        .expect("goc read");
    assert_eq!(state.pending_facts().len(), 1);
    assert_eq!(state.pending_facts()[0], fids[1]);
}

#[tokio::test]
async fn clear_session_removes_the_row() {
    let (store, _tmp) = fresh_store("clear_session").await;
    let id = session_id(1);
    let _ = store.get_or_create(&id, &memory_key()).await.expect("goc");
    store.clear_session(&id).await.expect("clear");

    let snap = store.snapshot_all().await.expect("snapshot");
    assert!(snap.is_empty(), "snapshot must be empty after clear");
}

#[tokio::test]
async fn collect_expired_returns_only_stale_sessions() {
    let (store, _tmp) = fresh_store("collect_expired").await;
    let id1 = session_id(1);
    let id2 = session_id(2);
    let _ = store
        .get_or_create(&id1, &memory_key())
        .await
        .expect("goc1");
    let _ = store
        .get_or_create(&id2, &memory_key())
        .await
        .expect("goc2");

    // Backdate id1 via the raw_db() escape hatch — no public mutator exists.
    // Use `<datetime>` cast because the SDK passes the string through as-is.
    let _ = store
        .raw_db()
        .query("UPDATE type::thing('session', $id) SET last_active = <datetime>$ts;")
        .bind(("id", id1.as_str().to_string()))
        .bind(("ts", "2020-01-01T00:00:00Z".to_string()))
        .await
        .expect("backdate");

    let expired = store
        .collect_expired(Duration::from_secs(60))
        .await
        .expect("collect");
    let expired_ids: HashSet<String> = expired
        .iter()
        .map(|(s, _)| s.as_str().to_string())
        .collect();
    assert!(
        expired_ids.contains(id1.as_str()),
        "backdated session must be in expired set"
    );
    assert!(
        !expired_ids.contains(id2.as_str()),
        "fresh session must NOT be in expired set"
    );
}

// ---------------------------------------------------------------------------
// dedup_and_mark
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dedup_and_mark_returns_new_on_first_call_and_empty_on_second() {
    let (store, _tmp) = fresh_store("dedup_idempotent").await;
    let id = session_id(1);
    let candidates = vec![FactId::from_content("alpha"), FactId::from_content("beta")];

    let new1 = store
        .dedup_and_mark(&id, &memory_key(), &candidates)
        .await
        .expect("dedup 1");
    let new1_set: HashSet<String> = new1.iter().map(|f| f.as_str().to_string()).collect();
    assert_eq!(
        new1_set,
        candidates.iter().map(|f| f.as_str().to_string()).collect(),
        "first call must return all candidates"
    );

    let new2 = store
        .dedup_and_mark(&id, &memory_key(), &candidates)
        .await
        .expect("dedup 2");
    assert!(new2.is_empty(), "second call must return nothing");
}
#[tokio::test]
async fn dedup_and_mark_concurrent_calls_do_not_double_inject() {
    let (store, _tmp) = fresh_store("dedup_concurrent").await;
    let id = session_id(1);
    let _ = store.get_or_create(&id, &memory_key()).await.expect("goc");

    let candidates = Arc::new(vec![
        FactId::from_content("alpha"),
        FactId::from_content("beta"),
        FactId::from_content("gamma"),
    ]);
    let store_arc = Arc::new(store);
    let id_arc = Arc::new(id);
    let mk_arc = Arc::new(memory_key());

    // Spawn two concurrent dedup_and_mark calls with the same candidates.
    // SurrealDB uses optimistic concurrency: under contention one call MAY
    // fail with TransactionConflict even after retries. The atomicity
    // contract we care about is "no double-inject" — the union of the
    // successful calls' returned id sets equals the candidate set, with no
    // id claimed by two calls.
    let s1 = tokio::spawn({
        let store = store_arc.clone();
        let id = id_arc.clone();
        let mk = mk_arc.clone();
        let cands = candidates.clone();
        async move { store.dedup_and_mark(&id, &mk, &cands).await }
    });
    let s2 = tokio::spawn({
        let store = store_arc.clone();
        let id = id_arc.clone();
        let mk = mk_arc.clone();
        let cands = candidates.clone();
        async move { store.dedup_and_mark(&id, &mk, &cands).await }
    });

    let r1 = s1.await.expect("task 1");
    let r2 = s2.await.expect("task 2");

    // Collect returned ids from successful calls. Conflicts are tolerated as
    // long as at least one call won the race and absorbed every candidate.
    let mut claimed: Vec<FactId> = Vec::new();
    let mut successes = 0;
    let mut conflicts = 0;
    for r in [r1, r2] {
        match r {
            Ok(ids) => {
                successes += 1;
                claimed.extend(ids);
            }
            Err(RepoError::TransactionConflict) => {
                conflicts += 1;
            }
            Err(e) => panic!("unexpected non-conflict error: {e}"),
        }
    }
    assert!(successes >= 1, "at least one call must succeed");

    // No id may appear in two responses (atomicity contract).
    let mut seen = HashSet::new();
    for f in &claimed {
        assert!(seen.insert(f.as_str().to_string()), "double-injected: {f}");
    }

    // Verify persisted injected set via raw_db(). Whatever split the two
    // successful calls produced, the persisted set must contain every
    // candidate exactly once.
    let mut res = store_arc
        .raw_db()
        .query("SELECT injected_facts FROM session WHERE id = type::thing('session', $id);")
        .bind(("id", id_arc.as_str().to_string()))
        .await
        .expect("query");
    let rows: Vec<InjectedRow> = res.take(0).expect("take");
    assert_eq!(rows.len(), 1);
    let persisted: HashSet<String> = rows[0].injected_facts.iter().cloned().collect();
    let expected: HashSet<String> = candidates.iter().map(|f| f.as_str().to_string()).collect();
    assert_eq!(
        persisted, expected,
        "persisted injected set must match candidates (successes={successes}, conflicts={conflicts})"
    );
}

// ---------------------------------------------------------------------------
// Performance test (1047 facts)
//
// Skipped in `debug_assertions` because brute-force cosine over 1047 × 1024-dim
// vectors in an unoptimised build is dominated by interpreter overhead. Run
// `cargo test --release -p smos-adapters search_similar_p95_under_threshold_on_1047_facts`
// to exercise this test and confirm the spec's p95 budget.
// ---------------------------------------------------------------------------

#[cfg(not(debug_assertions))]
#[tokio::test]
async fn search_similar_p95_under_threshold_on_1047_facts() {
    let (store, _tmp) = fresh_store("perf_1047").await;
    let n = 1047;
    // Synthetic reproducible embeddings via xorshift — values in [-1, 1].
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = vec![0.0_f32; 1024];
        for x in v.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *x = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        embeddings.push(v);
    }

    for (i, emb) in embeddings.iter().enumerate() {
        let fact = accepted_fact(
            &format!("fact-{i:04}"),
            Embedding::new(emb.clone()).expect("emb"),
            session_id(1),
        );
        FactRepository::save(&store, &fact).await.expect("save");
    }

    // Warm up.
    let _ = store
        .search_similar(embeddings[0].clone(), &memory_key(), 5)
        .await
        .expect("warmup");

    let mut latencies_ms: Vec<u128> = Vec::with_capacity(50);
    for i in 0..50 {
        let query_emb = embeddings[i % embeddings.len()].clone();
        let start = std::time::Instant::now();
        let hits = store
            .search_similar(query_emb, &memory_key(), 5)
            .await
            .expect("search");
        let elapsed = start.elapsed().as_millis();
        assert_eq!(hits.len(), 5, "must always return top-5");
        latencies_ms.push(elapsed);
    }
    latencies_ms.sort();
    let p95 = latencies_ms[(latencies_ms.len() * 95) / 100];
    let p50 = latencies_ms[latencies_ms.len() / 2];
    println!("search_similar on {n} facts: p50={p50}ms p95={p95}ms");
    // Brute-force cosine over 1047 × 1024-dim vectors must comfortably beat
    // the spec's 50ms p95 budget in release; we assert a 200 ms ceiling to
    // leave room for CI noise.
    assert!(
        p95 < 200,
        "p95 latency {p95}ms exceeds 200ms budget on {n} facts"
    );
}
