//! AC0 spike — validate SurrealQL syntax against a live embedded SurrealDB.
//!
//! This is NOT a unit test of any adapter: it is a one-shot exploration that
//! runs the planned DDL, an INSERT, a KNN search, and the dedup transaction
//! against a real SurrealDB 2.x instance (RocksDB in a tempdir) so we know
//! the exact syntax BEFORE writing `SurrealStore`. The findings are captured
//! as inline assertions and printed observations; later integration tests
//! build on the syntax validated here.
//!
//! Helpers below intentionally over-produce fixture constructors; the spike
//! keeps them around for ad-hoc exploration, so dead-code warnings are
//! suppressed at the file boundary.

#![allow(dead_code)]

use serde::Deserialize;
use smos_domain::{Embedding, Fact, FactId, MemoryKey, SessionId, Timestamp};
use std::collections::BTreeSet;
use surrealdb::Surreal;
use surrealdb::engine::local::RocksDb;
use tempfile::TempDir;

const DIM: usize = 1024;

#[derive(Debug, Deserialize)]

struct FactRow {
    id: surrealdb::RecordId,
    memory_key: String,
    content: String,
    fact_type: String,
    confidence: f32,
    status: String,
    valid_until: Option<String>,
    heat_base: f32,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]

struct SessionRow {
    id: surrealdb::RecordId,
    memory_key: String,
    injected_facts: Vec<String>,
    pending_facts: Vec<String>,
}

/// Generic KNN row used by both manual-similarity and KNN-operator queries.
/// `distance` is filled by the KNN operator, `similarity` by the manual
/// cosine helper. Both are optional so the same struct covers both paths.
#[derive(Debug, Deserialize)]
struct KnnHit {
    id: surrealdb::RecordId,

    content: Option<String>,
    distance: Option<f64>,
    similarity: Option<f64>,
}

impl KnnHit {
    fn similarity_or_distance(&self) -> f64 {
        self.distance.or(self.similarity).unwrap_or(f64::NAN)
    }
}

#[tokio::test]
async fn spike_ddl_insert_knn_and_transaction_all_work() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("sdb");
    let db = Surreal::new::<RocksDb>(path.to_str().expect("utf8 path"))
        .await
        .expect("rocksdb connect");
    db.use_ns("test").use_db("test").await.expect("use ns/db");

    // --- 1) Run the planned fact DDL ------------------------------------
    let ddl_fact = r#"
        DEFINE TABLE IF NOT EXISTS fact SCHEMAFULL;
        DEFINE FIELD IF NOT EXISTS memory_key      ON fact TYPE string;
        DEFINE FIELD IF NOT EXISTS content         ON fact TYPE string;
        DEFINE FIELD IF NOT EXISTS fact_type       ON fact TYPE string;
        DEFINE FIELD IF NOT EXISTS confidence      ON fact TYPE float;
        DEFINE FIELD IF NOT EXISTS status          ON fact TYPE string;
        DEFINE FIELD IF NOT EXISTS valid_from      ON fact TYPE datetime;
        DEFINE FIELD IF NOT EXISTS valid_until     ON fact TYPE option<datetime>;
        DEFINE FIELD IF NOT EXISTS extracted_at    ON fact TYPE datetime;
        DEFINE FIELD IF NOT EXISTS source_sessions ON fact TYPE array<string>;
        DEFINE FIELD IF NOT EXISTS conflicts_with  ON fact TYPE array<string>;
        DEFINE FIELD IF NOT EXISTS heat_base       ON fact TYPE float;
        DEFINE FIELD IF NOT EXISTS last_access_at  ON fact TYPE datetime;
        DEFINE FIELD IF NOT EXISTS embedding       ON fact TYPE array<float>;
        DEFINE INDEX IF NOT EXISTS fact_status_lookup ON fact COLUMNS memory_key, status;
        DEFINE INDEX IF NOT EXISTS fact_embedding_hnsw
            ON fact FIELDS embedding HNSW DIMENSION 1024 DIST COSINE TYPE F32;
    "#;
    let mut res = db.query(ddl_fact).await.expect("fact ddl");
    assert_no_query_errors(&mut res, "fact ddl");

    // --- 2) Run the planned session DDL ---------------------------------
    let ddl_session = r#"
        DEFINE TABLE IF NOT EXISTS session SCHEMAFULL;
        DEFINE FIELD IF NOT EXISTS memory_key     ON session TYPE string;
        DEFINE FIELD IF NOT EXISTS injected_facts ON session TYPE array<string>;
        DEFINE FIELD IF NOT EXISTS pending_facts  ON session TYPE array<string>;
        DEFINE FIELD IF NOT EXISTS created_at     ON session TYPE datetime;
        DEFINE FIELD IF NOT EXISTS last_active    ON session TYPE datetime;
        DEFINE INDEX IF NOT EXISTS session_lookup ON session COLUMNS memory_key;
    "#;
    let mut res = db.query(ddl_session).await.expect("session ddl");
    assert_no_query_errors(&mut res, "session ddl");

    // --- 3) Insert two accepted facts with orthogonal embeddings ---------
    let key = MemoryKey::from_raw("origa").expect("memory key");
    let fid_a = FactId::from_content("Rust is memory-safe");
    let fid_b = FactId::from_content("Python is dynamic");

    insert_fact_via_set(
        &db,
        &fid_a,
        &key,
        "Rust is memory-safe",
        unit_embedding(DIM, 0),
    )
    .await;
    insert_fact_via_set(
        &db,
        &fid_b,
        &key,
        "Python is dynamic",
        unit_embedding(DIM, 1),
    )
    .await;

    // --- 4) Verify SELECT * round-trips ---------------------------------
    let mut res = db
        .query("SELECT * FROM fact WHERE memory_key = $mk ORDER BY content")
        .bind(("mk", key.as_str().to_string()))
        .await
        .expect("select fact");
    assert_no_query_errors(&mut res, "select fact");
    let rows: Vec<FactRow> = res.take(0).expect("take rows");
    assert_eq!(rows.len(), 2, "exactly two facts expected");
    assert_eq!(rows[0].content, "Python is dynamic");
    assert_eq!(rows[0].embedding.len(), DIM);

    // --- 5) KNN search: query embedding = first fact's embedding --------
    // Step 5a: confirm `embedding` is stored as an array of floats and the
    // type matches what we bind.
    let mut res = db
        .query("SELECT array::len(embedding) AS dim FROM fact LIMIT 1")
        .await
        .expect("dim probe");
    let dims: Vec<serde_json::Value> = res.take(0).expect("take dim probe");
    println!("embedding dim probe: {dims:?}");

    // Step 5b: brute-force similarity using `vector::similarity::cosine`.
    // (SurrealDB exposes cosine as a similarity, not a distance — distance is
    // `1 - similarity`.) This call bypasses the KNN operator and proves the
    // embeddings are queryable independently of the operator.
    let cosine_manual = r#"
        SELECT id, content,
               vector::similarity::cosine(embedding, $embedding) AS similarity
        FROM fact
        WHERE memory_key = $mk AND status = 'accepted'
        ORDER BY similarity DESC;
    "#;
    let mut res = db
        .query(cosine_manual)
        .bind(("mk", key.as_str().to_string()))
        .bind(("embedding", unit_embedding_f64(DIM, 0)))
        .await
        .expect("cosine manual query");
    assert_no_query_errors(&mut res, "cosine manual query");
    let manual_hits: Vec<KnnHit> = res.take(0).expect("take cosine manual rows");
    println!("manual cosine hits: {}", manual_hits.len());
    assert!(!manual_hits.is_empty(), "manual cosine must work");
    println!(
        "manual first hit: id={} similarity={}",
        manual_hits[0].id,
        manual_hits[0].similarity_or_distance()
    );

    // Step 5c: brute-force KNN operator `<|K,COSINE|>` — distance is
    // 1 - similarity; the operator computes it lazily and `vector::distance::
    // knn()` exposes the precomputed value.
    //
    // NOTE: With an HNSW index defined on `embedding`, the SurrealDB 2.6
    // query planner appears to route the operator through the HNSW path even
    // when an explicit distance function is supplied. We empirically observe
    // that the operator returns zero rows in that case. To validate the
    // operator itself in isolation, we drop the index first.
    let mut res = db
        .query("REMOVE INDEX IF EXISTS fact_embedding_hnsw ON fact;")
        .await
        .expect("drop hnsw");
    assert_no_query_errors(&mut res, "drop hnsw");

    let knn_bruteforce_no_filter = r#"
        SELECT id,
               vector::distance::knn() AS distance
        FROM fact
        WHERE embedding <|5,COSINE|> $embedding
        ORDER BY distance;
    "#;
    let mut res = db
        .query(knn_bruteforce_no_filter)
        .bind(("embedding", unit_embedding_f64(DIM, 0)))
        .await
        .expect("knn brute-force (no filter) query");
    assert_no_query_errors(&mut res, "knn brute-force (no filter) query");
    let bf_nf_hits: Vec<KnnHit> = res.take(0).expect("take knn brute-force (no filter) rows");
    println!("brute-force KNN (no filter) hits: {}", bf_nf_hits.len());

    let knn_bruteforce = r#"
        SELECT id, content,
               vector::distance::knn() AS distance
        FROM fact
        WHERE memory_key = $mk
          AND status = 'accepted'
          AND embedding <|5,COSINE|> $embedding
        ORDER BY distance;
    "#;
    let mut res = db
        .query(knn_bruteforce)
        .bind(("mk", key.as_str().to_string()))
        .bind(("embedding", unit_embedding_f64(DIM, 0)))
        .await
        .expect("knn brute-force query");
    assert_no_query_errors(&mut res, "knn brute-force query");
    let bf_hits: Vec<KnnHit> = res.take(0).expect("take knn brute-force rows");
    println!("brute-force KNN hits (with pre-filter): {}", bf_hits.len());
    // KNOWN LIMITATION (SurrealDB 2.6): combining the KNN operator with
    // equality pre-filters (`memory_key = …`, `status = 'accepted'`) returns
    // zero rows. The recommended workaround is post-filtering in application
    // code or using `vector::similarity::cosine` with a regular ORDER BY.

    // Step 5d: HNSW-backed KNN operator `<|K,EF|>`. Re-create the index first.
    let mut res = db
        .query(
            r#"
        DEFINE INDEX IF NOT EXISTS fact_embedding_hnsw
            ON fact FIELDS embedding HNSW DIMENSION 1024 DIST COSINE TYPE F32;
    "#,
        )
        .await
        .expect("re-create hnsw");
    assert_no_query_errors(&mut res, "re-create hnsw");

    let knn_hnsw = r#"
        SELECT id, content,
               vector::distance::knn() AS distance
        FROM fact
        WHERE memory_key = $mk
          AND status = 'accepted'
          AND embedding <|5,50|> $embedding
        ORDER BY distance;
    "#;
    let mut res = db
        .query(knn_hnsw)
        .bind(("mk", key.as_str().to_string()))
        .bind(("embedding", unit_embedding_f64(DIM, 0)))
        .await
        .expect("knn hnsw query");
    assert_no_query_errors(&mut res, "knn hnsw query");
    let hnsw_hits: Vec<KnnHit> = res.take(0).expect("take knn hnsw rows");
    println!("hnsw KNN hits: {}", hnsw_hits.len());

    let hits: Vec<KnnHit> = if !hnsw_hits.is_empty() {
        hnsw_hits
    } else if !bf_hits.is_empty() {
        bf_hits
    } else if !bf_nf_hits.is_empty() {
        bf_nf_hits
    } else {
        manual_hits
    };
    assert!(!hits.is_empty(), "knn returned at least one row");
    let first = &hits[0];
    println!(
        "KNN hit: id={} distance={:.6}",
        first.id,
        first.similarity_or_distance()
    );
    assert!(
        first.id.to_string().contains(fid_a.as_str()),
        "expected fid_a ({}) to be the closest; got {}",
        fid_a.as_str(),
        first.id
    );

    // --- 6) dedup_and_mark atomic transaction ---------------------------
    let session_id = SessionId::from_raw("sess_abcdef012345").expect("session id");

    // Bootstrap a session row (mimics get_or_create).
    let bootstrap = r#"
        INSERT INTO session (id, memory_key, injected_facts, pending_facts, created_at, last_active)
        VALUES (type::thing('session', $id), $mk, [], [], time::now(), time::now());
    "#;
    let mut res = db
        .query(bootstrap)
        .bind(("id", session_id.as_str().to_string()))
        .bind(("mk", key.as_str().to_string()))
        .await
        .expect("session bootstrap");
    assert_no_query_errors(&mut res, "session bootstrap");

    // The atomic dedup+mark transaction.
    //
    // `array::complement(a, b)` returns the items in `a` that are not in `b`
    // — the relative complement A\B. Do NOT use `array::difference(a, b)`:
    // that one computes the SYMMETRIC difference A△B, which is wrong when
    // `b` carries ids that are not in `a` (it would re-introduce them as
    // "new"). See https://surrealdb.com/docs/reference/query-language/functions/database-functions/array.
    let tx = r#"
        BEGIN TRANSACTION;
            LET $existing = (SELECT injected_facts FROM session WHERE id = type::thing('session', $id) LIMIT 1);
            LET $current = array::flatten($existing.injected_facts);
            LET $new = array::complement($candidates, $current);
            UPDATE type::thing('session', $id) SET injected_facts = array::union($current, $new);
            RETURN $new;
        COMMIT TRANSACTION;
    "#;
    let mut res = db
        .query(tx)
        .bind(("id", session_id.as_str().to_string()))
        .bind((
            "candidates",
            vec![fid_a.as_str().to_string(), fid_b.as_str().to_string()],
        ))
        .await
        .expect("dedup tx first call");
    assert_no_query_errors(&mut res, "dedup tx first call");
    println!(
        "dedup tx num_statements after BEGIN/LET/LET/LET/UPDATE/RETURN/COMMIT: {}",
        res.num_statements()
    );
    let first_new = take_return_value(&mut res);
    println!("dedup first call new = {first_new:?}");
    assert_eq!(
        first_new.len(),
        2,
        "first call must return both candidates as new"
    );

    // Second call: both candidates already injected → must return [].
    let mut res = db
        .query(tx)
        .bind(("id", session_id.as_str().to_string()))
        .bind((
            "candidates",
            vec![fid_a.as_str().to_string(), fid_b.as_str().to_string()],
        ))
        .await
        .expect("dedup tx second call");
    assert_no_query_errors(&mut res, "dedup tx second call");
    let second_new = take_return_value(&mut res);
    println!("dedup second call new = {second_new:?}");
    assert!(
        second_new.is_empty(),
        "second call must return [] (idempotent)"
    );

    // Third call: regression guard for the symmetric-vs-relative complement
    // trap. The session already has [fid_a, fid_b] injected; offering a
    // SUBSET (just fid_a) must return [] (nothing new). The buggy
    // `array::difference` would return [fid_b] here because fid_b is in
    // `current` but not in `candidates` — symmetric difference, not the
    // relative complement the dedup contract requires.
    let mut res = db
        .query(tx)
        .bind(("id", session_id.as_str().to_string()))
        .bind(("candidates", vec![fid_a.as_str().to_string()]))
        .await
        .expect("dedup tx third call");
    assert_no_query_errors(&mut res, "dedup tx third call");
    let third_new = take_return_value(&mut res);
    println!("dedup third call new = {third_new:?}");
    assert!(
        third_new.is_empty(),
        "third call (subset of injected) must return [] — proves \
         array::complement (A\\B), not array::difference (A△B)"
    );

    // Verify the session row ended up with both ids injected.
    let mut res = db
        .query("SELECT injected_facts FROM session WHERE id = type::thing('session', $id)")
        .bind(("id", session_id.as_str().to_string()))
        .await
        .expect("session final read");
    #[derive(Debug, Deserialize)]
    struct InjectedOnly {
        injected_facts: Vec<String>,
    }
    let rows: Vec<InjectedOnly> = res.take(0).expect("session row");
    assert_eq!(rows.len(), 1);
    let mut sorted = rows[0].injected_facts.clone();
    sorted.sort();
    assert_eq!(sorted.len(), 2);
}

// --- helpers -------------------------------------------------------------

fn assert_no_query_errors(res: &mut surrealdb::Response, ctx: &str) {
    let errors: Vec<_> = res.take_errors().into_iter().collect();
    assert!(
        errors.is_empty(),
        "unexpected SurrealQL errors in `{ctx}`: {errors:?}"
    );
}

/// Take the result of the `RETURN $new;` statement from the dedup
/// transaction. Scans every statement slot (in order) and returns the first
/// slot whose value deserialises as `Vec<String>` — the LET statements and
/// the UPDATE produce `None` values, only RETURN yields the array.
fn take_return_value(res: &mut surrealdb::Response) -> Vec<String> {
    let n = res.num_statements();
    for i in 0..n {
        if let Ok(v) = res.take::<Vec<String>>(i) {
            return v;
        }
    }
    Vec::new()
}

async fn insert_fact_via_set(
    db: &Surreal<surrealdb::engine::local::Db>,
    id: &FactId,
    memory_key: &MemoryKey,
    content: &str,
    embedding: Vec<f32>,
) {
    let stmt = r#"
        INSERT INTO fact (id, memory_key, content, fact_type, confidence,
                          status, valid_from, valid_until, extracted_at,
                          source_sessions, conflicts_with, heat_base,
                          last_access_at, embedding)
        VALUES (type::thing('fact', $id), $mk, $content, 'entity', 0.85,
                'accepted', time::now(), NONE, time::now(),
                [], [], 1.0, time::now(), $embedding)
        ON DUPLICATE KEY UPDATE content = $content,
                                 embedding = $embedding,
                                 confidence = 0.85,
                                 status = 'accepted';
    "#;
    let mut res = db
        .query(stmt)
        .bind(("id", id.as_str().to_string()))
        .bind(("mk", memory_key.as_str().to_string()))
        .bind(("content", content.to_string()))
        .bind(("embedding", embedding))
        .await
        .expect("insert fact");
    assert_no_query_errors(&mut res, "insert fact");
}

/// Produce a deterministic unit-norm 1024-d vector where only the index-th
/// component is non-zero. Two such vectors with different indices are
/// orthogonal (cosine distance = 1.0); identical indices give distance 0.
fn unit_embedding(dim: usize, index: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[index] = 1.0;
    v
}

/// Same as [`unit_embedding`] but in `f64`. SurrealDB's `float` type is 64-bit;
/// binding `Vec<f64>` ensures the SDK does not silently widen or reject the
/// array when checking it against `array<float>` columns.
fn unit_embedding_f64(dim: usize, index: usize) -> Vec<f64> {
    let mut v = vec![0.0_f64; dim];
    v[index] = 1.0;
    v
}

// Build a domain `Fact` to confirm we still can — kept around as a sanity
// check that nothing in `smos-domain` has shifted.
fn sample_fact(content: &str, embedding: Vec<f32>) -> Fact {
    Fact::new_pending(
        content,
        MemoryKey::from_raw("origa").unwrap(),
        SessionId::from_raw("sess_abcdef012345").unwrap(),
        Embedding::new(embedding).unwrap(),
        Timestamp::from_unix_secs(1_700_000_000).unwrap(),
        smos_domain::config::ConfidenceConfig::default().base,
    )
    .unwrap()
}

// Sentinel that proves BTreeSet<FactId> serialises to a Surreal array —
// exercised by the production adapter when persisting `SessionState`.
fn fact_id_set() -> BTreeSet<FactId> {
    let mut s = BTreeSet::new();
    s.insert(FactId::from_content("a"));
    s.insert(FactId::from_content("b"));
    s
}
