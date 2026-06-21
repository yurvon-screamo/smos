//! SurrealDB schema DDL.
//!
//! All statements are idempotent (`IF NOT EXISTS`) and run sequentially by
//! [`crate::storage::surreal_store::SurrealStore::run_migrations`]. The exact
//! syntax used here was validated by the AC0 spike
//! (`tests/spike_surrealdb_syntax.rs`) against SurrealDB 2.6 with the
//! embedded RocksDB engine.

/// DDL statements for the `fact` aggregate.
///
/// Schema decisions:
/// - `SCHEMAFULL` enforces column types at write time so malformed adapters
///   fail loudly instead of silently corrupting the store.
/// - `embedding` is `array<float>` (Surreal f64) — the embedding model emits
///   `f32`, but Surreal widens to `f64` on store. The HNSW index pins the
///   on-disk encoding to `TYPE F32` for memory efficiency.
/// - `valid_until` is `option<datetime>` because facts without a tombstone
///   are still current.
/// - The HNSW index uses `DIST COSINE` to match the embedding model's metric
///   space. The current adapter issues brute-force cosine queries (see
///   [`super::surreal_store`]), but the index is kept here so we can switch
///   to the KNN operator at production scale without a schema migration.
pub const FACT_DDL: &str = r#"
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

/// DDL statements for the `session` aggregate.
///
/// - `injected_facts` and `pending_facts` are arrays of strings (FactId
///   strings); Surreal arrays are mutable, so the dedup transaction can
///   `array::union` / `array::complement` them in place.
/// - `created_at` and `last_active` are `datetime`s so expiry queries can
///   compare against `time::now()` directly.
pub const SESSION_DDL: &str = r#"
    DEFINE TABLE IF NOT EXISTS session SCHEMAFULL;
    DEFINE FIELD IF NOT EXISTS memory_key     ON session TYPE string;
    DEFINE FIELD IF NOT EXISTS injected_facts ON session TYPE array<string>;
    DEFINE FIELD IF NOT EXISTS pending_facts  ON session TYPE array<string>;
    DEFINE FIELD IF NOT EXISTS created_at     ON session TYPE datetime;
    DEFINE FIELD IF NOT EXISTS last_active    ON session TYPE datetime;
    DEFINE INDEX IF NOT EXISTS session_lookup ON session COLUMNS memory_key;
"#;

/// Atomic dedup + mark transaction (POC parity with `select_new_facts`).
///
/// Returns the subset of `$candidates` that were NOT previously injected,
/// and records them as injected so concurrent calls cannot double-inject.
/// SurrealDB 2.6 collapses a BEGIN…COMMIT block into a single statement
/// from the SDK's perspective, so the RETURN value lands at slot 0 of the
/// query response.
///
/// SurrealQL notes:
/// - `array::complement(a, b)` returns items in `a` not in `b` (the
///   relative complement A\B). Do NOT confuse with `array::difference`,
///   which is the SYMMETRIC difference A△B and would re-introduce already
///   injected ids when `b` carries ids that are not in `a`.
/// - `array::flatten` normalises the `[[...]]` returned by the SELECT to
///   `[...]`.
/// - `type::thing('table', $id)` is the 2.x syntax for record-id
///   construction (renamed to `type::record` in 3.0).
pub const DEDUP_AND_MARK_TX: &str = r#"
    BEGIN TRANSACTION;
        LET $existing = (
            SELECT injected_facts FROM session
            WHERE id = type::thing('session', $id) LIMIT 1
        );
        LET $current = array::flatten($existing.injected_facts);
        LET $new = array::complement($candidates, $current);
        UPDATE type::thing('session', $id)
            SET injected_facts = array::union($current, $new),
                last_active = time::now();
        RETURN $new;
    COMMIT TRANSACTION;
"#;
