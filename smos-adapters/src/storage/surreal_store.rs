//! `SurrealStore` — concrete `FactRepository` + `SessionRepository` over
//! SurrealDB 2.x (embedded RocksDB by default; the same code works against a
//! remote server via the protocol engines).
//!
//! # Architecture
//!
//! The store owns one `Surreal<Db>` client. All port methods compile their
//! SurrealQL inline; the query text is the *single source of truth* for what
//! the adapter does. Datetimes are bound as ISO-8601 strings (parsed back
//! from the same format on read) to keep the row schema self-describing
//! without coupling the row structs to a specific datetime crate.
//!
//! # AC0 spike
//!
//! Every SurrealQL statement here was validated by
//! `tests/spike_surrealdb_syntax.rs` against SurrealDB 2.6 with the
//! embedded RocksDB engine. See `surreal_schema.rs` for the canonical DDL
//! strings and `DEDUP_AND_MARK_TX` for the atomic dedup transaction.

use std::time::Duration;

use smos_application::errors::RepoError;
use surrealdb::Surreal;
use surrealdb::engine::local::Db;

use crate::storage::surreal_schema::{FACT_DDL, SESSION_DDL};

// Submodules implementing the port traits. Declared in `storage/mod.rs` so
// the files live alongside `surreal_store.rs` (rows / mapping / vector_search
// / fact_repository / session_repository). The facade below keeps the public
// `SurrealStore` constructor surface; everything else is delegated.

/// SurrealDB-backed persistence for `Fact` and `SessionState`.
#[derive(Clone)]
pub struct SurrealStore {
    pub(crate) db: Surreal<Db>,
}

impl SurrealStore {
    /// Open (or create) a SurrealDB database at `path` (filesystem directory
    /// for RocksDB). Retries up to three attempts with exponential backoff
    /// (1 s after the first failure, 2 s after the second) — the engine
    /// occasionally returns a transient lock error on rapid re-opens in
    /// tests, and the doubling schedule means a hypothetical fourth attempt
    /// would wait 4 s without code changes.
    pub async fn connect(path: &str, namespace: &str, database: &str) -> Result<Self, RepoError> {
        let mut last_err: Option<String> = None;
        for attempt in 0..3u32 {
            if attempt > 0 {
                // Exponential backoff: `attempt = 1` waits 1 s, `attempt = 2`
                // waits 2 s. The doubling base means adding a fourth attempt
                // (e.g. for a flakier engine) would naturally sleep 4 s
                // without further tuning. `attempt` is bounded by the loop
                // constant (≤ 2), so the shift cannot overflow a u64.
                let backoff_ms = 1000_u64 << (attempt - 1);
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            }
            match Surreal::new::<surrealdb::engine::local::RocksDb>(path.to_string()).await {
                Ok(db) => {
                    db.use_ns(namespace.to_string())
                        .use_db(database.to_string())
                        .await
                        .map_err(|e| RepoError::ConnectFailed(e.to_string()))?;
                    return Ok(Self { db });
                }
                Err(e) => {
                    last_err = Some(e.to_string());
                }
            }
        }
        Err(RepoError::ConnectFailed(
            last_err.unwrap_or_else(|| "unknown connect failure".into()),
        ))
    }

    /// Wrap an existing `Surreal<Db>` handle. Useful for tests that spin up
    /// their own engine (Mem, RocksDb in tempdir, …) and want to skip
    /// `connect`'s retry loop.
    pub fn from_client(db: Surreal<Db>) -> Self {
        Self { db }
    }

    /// Apply all idempotent DDL statements (see [`super::surreal_schema`]).
    pub async fn run_migrations(&self) -> Result<(), RepoError> {
        let mut res = self
            .db
            .query(FACT_DDL)
            .query(SESSION_DDL)
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "run_migrations")?;
        Ok(())
    }

    /// Select namespace + database on the underlying client. Convenience for
    /// tests that share one engine across multiple namespaces.
    pub async fn use_ns_db(&self, namespace: &str, database: &str) -> Result<(), RepoError> {
        self.db
            .use_ns(namespace.to_string())
            .use_db(database.to_string())
            .await
            .map_err(|e| RepoError::QueryFailed(e.to_string()))
    }

    /// Read-only access to the underlying Surreal client.
    ///
    /// Exposed for tooling, observability, and integration tests that need
    /// raw SurrealQL (e.g. backdating a row to test `collect_expired`).
    /// Production code SHOULD go through the port-trait methods; this
    /// accessor is an escape hatch, not a primary API.
    pub fn raw_db(&self) -> &Surreal<Db> {
        &self.db
    }

    pub(crate) fn map_db_error(e: surrealdb::Error) -> RepoError {
        RepoError::QueryFailed(e.to_string())
    }

    /// Drain per-statement errors from a SurrealQL response and surface them
    /// as a single `RepoError::QueryFailed`. Used by every port method so the
    /// 11-line boilerplate is not duplicated (clean-code C5).
    pub(crate) fn check_errors(res: &mut surrealdb::Response, ctx: &str) -> Result<(), RepoError> {
        let errors: Vec<_> = res.take_errors().into_iter().collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(RepoError::QueryFailed(format!("{ctx}: {errors:?}")))
        }
    }

    /// Classify a SurrealDB transaction conflict (optimistic concurrency
    /// rollback). The SurrealDB 2.6 Rust SDK does not yet expose a typed
    /// variant for `QueryNotExecutedDetail`, so we fall back to a substring
    /// check on the error message. The tokens are kept in a single helper
    /// so a future SDK upgrade can replace the substring match with a
    /// structural one in one place.
    pub(crate) fn is_transaction_conflict(err: &surrealdb::Error) -> bool {
        let msg = err.to_string();
        Self::is_transaction_conflict_message(&msg)
    }

    /// Substring check used both for direct `surrealdb::Error` and for the
    /// embedded text inside a `RepoError::QueryFailed` returned by
    /// `check_errors`. Kept separate so the substring tokens live in one
    /// place.
    pub(crate) fn is_transaction_conflict_message(msg: &str) -> bool {
        msg.contains("read or write conflict") || msg.contains("transaction can be retried")
    }
}
