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

use serde::{Deserialize, Serialize};
use smos_application::{
    errors::RepoError,
    ports::{FactRepository, SessionRepository},
    types::{SearchHit, SearchHitMetadata},
};
use smos_domain::{
    Confidence, Embedding, Fact, FactContent, FactId, FactStatus, FactType, Heat, MemoryKey,
    SessionId, SessionState, SourceSessions, Timestamp,
};
use surrealdb::Surreal;
use surrealdb::engine::local::Db;
use time::OffsetDateTime;

use crate::storage::surreal_schema::{DEDUP_AND_MARK_TX, FACT_DDL, SESSION_DDL};

/// SurrealDB-backed persistence for `Fact` and `SessionState`.
#[derive(Clone)]
pub struct SurrealStore {
    db: Surreal<Db>,
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

    fn map_db_error(e: surrealdb::Error) -> RepoError {
        RepoError::QueryFailed(e.to_string())
    }

    /// Drain per-statement errors from a SurrealQL response and surface them
    /// as a single `RepoError::QueryFailed`. Used by every port method so the
    /// 11-line boilerplate is not duplicated (clean-code C5).
    fn check_errors(res: &mut surrealdb::Response, ctx: &str) -> Result<(), RepoError> {
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
    fn is_transaction_conflict(err: &surrealdb::Error) -> bool {
        let msg = err.to_string();
        Self::is_transaction_conflict_message(&msg)
    }

    /// Substring check used both for direct `surrealdb::Error` and for the
    /// embedded text inside a `RepoError::QueryFailed` returned by
    /// `check_errors`. Kept separate so the substring tokens live in one
    /// place.
    fn is_transaction_conflict_message(msg: &str) -> bool {
        msg.contains("read or write conflict") || msg.contains("transaction can be retried")
    }
}

// ---------------------------------------------------------------------------
// Fact row <-> domain mapping
// ---------------------------------------------------------------------------

/// Database projection of a `Fact` row.
///
/// `id` is the Surreal record id (`fact:<fact_id_string>`); the application-
/// level FactId is reconstructed from its key portion. All datetime fields
/// are ISO-8601 strings to keep the row self-describing. `id` is read by
/// serde from SurrealDB responses but the application never consults it —
/// `#[allow(dead_code)]` documents that intentional asymmetry.
#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
struct FactRow {
    #[serde(skip_serializing)]
    id: Option<surrealdb::RecordId>,
    memory_key: String,
    content: String,
    fact_type: String,
    confidence: f32,
    status: String,
    valid_from: String,
    valid_until: Option<String>,
    extracted_at: String,
    source_sessions: Vec<String>,
    conflicts_with: Vec<String>,
    heat_base: f32,
    last_access_at: String,
    embedding: Option<Vec<f32>>,
}

impl FactRow {
    fn from_fact(fact: &Fact) -> Result<Self, RepoError> {
        let valid_until = fact
            .valid_until()
            .map(|ts| format_iso(ts.as_offset_date_time()));
        let embedding = fact.embedding().map(|e| e.as_slice().to_vec());
        let source_sessions = fact
            .source_sessions()
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();
        let conflicts_with = fact
            .conflicts_with()
            .iter()
            .map(|c| c.as_str().to_string())
            .collect();
        Ok(Self {
            id: None,
            memory_key: fact.memory_key().as_str().to_string(),
            content: fact.content().to_string(),
            fact_type: fact.fact_type().as_str().to_string(),
            confidence: fact.confidence().value(),
            status: fact.status().as_str().to_string(),
            valid_from: format_iso(fact.valid_from().as_offset_date_time()),
            valid_until,
            extracted_at: format_iso(fact.extracted_at().as_offset_date_time()),
            source_sessions,
            conflicts_with,
            heat_base: fact.heat_base().value(),
            last_access_at: format_iso(fact.last_access_at().as_offset_date_time()),
            embedding,
        })
    }

    fn to_fact(&self, id: FactId) -> Result<Fact, RepoError> {
        let fact_type = parse_fact_type(&self.fact_type)?;
        let status = parse_fact_status(&self.status)?;
        let valid_from = parse_iso(&self.valid_from)?;
        let valid_until = match &self.valid_until {
            Some(s) => Some(parse_iso(s)?),
            None => None,
        };
        let extracted_at = parse_iso(&self.extracted_at)?;
        let last_access_at = parse_iso(&self.last_access_at)?;

        let memory_key = MemoryKey::from_raw(&self.memory_key)
            .map_err(|e| RepoError::SerializationFailed(e.to_string()))?;
        let content = FactContent::new(self.content.clone())
            .map_err(|e| RepoError::SerializationFailed(e.to_string()))?;
        let confidence = Confidence::new(self.confidence).map_err(domain_to_repo)?;
        let heat_base = Heat::new(self.heat_base).map_err(domain_to_repo)?;
        let embedding = self
            .embedding
            .as_ref()
            .map(|v| Embedding::new(v.clone()))
            .transpose()
            .map_err(domain_to_repo)?;

        let source_sessions_iter = self.source_sessions.iter().map(|s| {
            SessionId::from_raw(s).map_err(|e| RepoError::SerializationFailed(e.to_string()))
        });
        let source_sessions_vec: Vec<SessionId> = source_sessions_iter.collect::<Result<_, _>>()?;
        let source_sessions = SourceSessions::from_vec(source_sessions_vec);

        let conflicts_with: Vec<FactId> = self
            .conflicts_with
            .iter()
            .map(|c| FactId::from_raw(c))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| RepoError::SerializationFailed(e.to_string()))?;

        // Round-trip safe path: `Fact::rehydrate` rebuilds every field
        // verbatim with no recomputation (no `reclassify`, no `boost_heat`).
        // All invariants are enforced by the domain constructor.
        Fact::rehydrate(
            id,
            memory_key,
            content,
            fact_type,
            confidence,
            status,
            valid_from,
            valid_until,
            extracted_at,
            source_sessions,
            conflicts_with,
            heat_base,
            last_access_at,
            embedding,
        )
        .map_err(domain_to_repo)
    }
}

fn domain_to_repo(e: smos_domain::DomainError) -> RepoError {
    RepoError::SerializationFailed(e.to_string())
}

fn parse_fact_type(s: &str) -> Result<FactType, RepoError> {
    match s {
        "decision" => Ok(FactType::Decision),
        "preference" => Ok(FactType::Preference),
        "entity" => Ok(FactType::Entity),
        "event" => Ok(FactType::Event),
        "technical" => Ok(FactType::Technical),
        other => Err(RepoError::SerializationFailed(format!(
            "unknown fact_type: {other}"
        ))),
    }
}

fn parse_fact_status(s: &str) -> Result<FactStatus, RepoError> {
    match s {
        "pending" => Ok(FactStatus::Pending),
        "accepted" => Ok(FactStatus::Accepted),
        "rejected" => Ok(FactStatus::Rejected),
        other => Err(RepoError::SerializationFailed(format!(
            "unknown status: {other}"
        ))),
    }
}

fn format_iso(ts: OffsetDateTime) -> String {
    // `time`'s `Rfc3339` format is widely compatible and accepted by
    // SurrealDB's `datetime` parser. Formatting a `Rfc3339`-compatible
    // `OffsetDateTime` should never fail in practice, but the previous
    // silent fallback to `"1970-01-01T00:00:00Z"` would corrupt the
    // timestamp-dependent heat decay if it ever did. Surface the error at
    // ERROR level and fall back to the debug representation instead of
    // silently emitting the epoch.
    //
    // Caveat: the `time` crate's `Debug` for `OffsetDateTime` is NOT a
    // valid Rfc3339 string, so `parse_iso` will likely fail on a
    // round-trip. The branch is "should be unreachable" — the goal is to
    // avoid the silent epoch-corruption of heat decay timestamps and to
    // leave a forensic trail in the ERROR log + stored row, not to keep
    // the row readable.
    match ts.format(&time::format_description::well_known::Rfc3339) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(
                error = %e,
                timestamp = ?ts,
                "Rfc3339 formatting failed; this should be unreachable — please report"
            );
            format!("{:?}", ts)
        }
    }
}

fn parse_iso(s: &str) -> Result<Timestamp, RepoError> {
    OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339)
        .map_err(|e| RepoError::SerializationFailed(format!("invalid datetime {s:?}: {e}")))
        .and_then(|odt| {
            Timestamp::from_unix_secs(odt.unix_timestamp())
                .map_err(|e| RepoError::SerializationFailed(format!("unix out of range: {e}")))
        })
}

// (No `From<OffsetDateTime> for Timestamp` impl: both are foreign types, so
// the orphan rule forbids it. `parse_iso` does the same job via the public
// `Timestamp::from_unix_secs` constructor.)

// ---------------------------------------------------------------------------
// Session row <-> domain mapping
// ---------------------------------------------------------------------------

// `id` is the Surreal record id (`session:<session_id_string>`); serde reads
// it from SurrealDB responses but the application reconstructs the session
// id from the typed column below, so the field is intentionally unread.
#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
struct SessionRow {
    #[serde(skip_serializing)]
    id: Option<surrealdb::RecordId>,
    memory_key: String,
    injected_facts: Vec<String>,
    pending_facts: Vec<String>,
    created_at: String,
    last_active: String,
}

impl SessionRow {
    fn to_state(&self, id: SessionId) -> Result<SessionState, RepoError> {
        let memory_key = MemoryKey::from_raw(&self.memory_key)
            .map_err(|e| RepoError::SerializationFailed(e.to_string()))?;
        let created_at = parse_iso(&self.created_at)?;
        let last_active = parse_iso(&self.last_active)?;

        let injected_facts = self
            .injected_facts
            .iter()
            .map(|s| FactId::from_raw(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| RepoError::SerializationFailed(e.to_string()))?;
        let pending_facts = self
            .pending_facts
            .iter()
            .map(|s| FactId::from_raw(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| RepoError::SerializationFailed(e.to_string()))?;

        // Round-trip safe path: `SessionState::rehydrate` rebuilds every
        // field verbatim, including the injected_facts set that has no
        // public mutator. The `?` propagates the `last_active < created_at`
        // invariant failure as a `SerializationFailed` so a corrupt row is
        // surfaced loudly instead of silently producing an impossible state.
        SessionState::rehydrate(
            id,
            memory_key,
            injected_facts,
            pending_facts,
            created_at,
            last_active,
        )
        .map_err(|e| RepoError::SerializationFailed(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// FactRepository impl
// ---------------------------------------------------------------------------

impl FactRepository for SurrealStore {
    async fn save(&self, fact: &Fact) -> Result<(), RepoError> {
        // Build the row + extract datetime fields as ISO-8601 strings; the
        // SQL `<datetime>` casts coerce them to Surreal's native datetime
        // type (the SDK's serde path keeps them as strings otherwise, which
        // the SCHEMAFULL check rejects).
        //
        // For `valid_until` we generate two SQL variants because `<datetime>`
        // cannot cast the literal string `"NONE"`: when there is no tombstone
        // we explicitly assign the SurrealQL `NONE` keyword (which the
        // `option<datetime>` field accepts as "field not set").
        let row = FactRow::from_fact(fact)?;
        let memory_key_str = fact.memory_key().as_str().to_string();
        let content_str = fact.content().to_string();
        let fact_type_str = fact.fact_type().as_str().to_string();
        let confidence_val = fact.confidence().value();
        let status_str = fact.status().as_str().to_string();
        let valid_from_str = row.valid_from.clone();
        let valid_until_iso: Option<String> = row.valid_until.clone();
        let extracted_at_str = row.extracted_at.clone();
        let source_sessions = row.source_sessions.clone();
        let conflicts_with = row.conflicts_with.clone();
        let heat_val = fact.heat_base().value();
        let last_access_str = row.last_access_at.clone();
        let embedding: Option<Vec<f32>> = fact.embedding().map(|e| e.as_slice().to_vec());

        let valid_until_clause = match &valid_until_iso {
            Some(iso) => format!("<datetime>{iso:?}"),
            None => "NONE".to_string(),
        };
        let sql = format!(
            r#"UPSERT type::thing('fact', $id) SET
                    memory_key      = $mk,
                    content         = $content,
                    fact_type       = $fact_type,
                    confidence      = $confidence,
                    status          = $status,
                    valid_from      = <datetime>$valid_from,
                    valid_until     = {valid_until_clause},
                    extracted_at    = <datetime>$extracted_at,
                    source_sessions = $source_sessions,
                    conflicts_with  = $conflicts_with,
                    heat_base       = $heat,
                    last_access_at  = <datetime>$last_access,
                    embedding       = $embedding;"#
        );

        let mut res = self
            .db
            .query(&sql)
            .bind(("id", fact.id().as_str().to_string()))
            .bind(("mk", memory_key_str))
            .bind(("content", content_str))
            .bind(("fact_type", fact_type_str))
            .bind(("confidence", confidence_val))
            .bind(("status", status_str))
            .bind(("valid_from", valid_from_str))
            .bind(("extracted_at", extracted_at_str))
            .bind(("source_sessions", source_sessions))
            .bind(("conflicts_with", conflicts_with))
            .bind(("heat", heat_val))
            .bind(("last_access", last_access_str))
            .bind(("embedding", embedding))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "query")?;
        Ok(())
    }

    async fn get(&self, id: &FactId, memory_key: &MemoryKey) -> Result<Option<Fact>, RepoError> {
        let mut res = self
            .db
            .query(
                "SELECT * FROM fact
                 WHERE id = type::thing('fact', $id) AND memory_key = $mk
                 LIMIT 1;",
            )
            .bind(("id", id.as_str().to_string()))
            .bind(("mk", memory_key.as_str().to_string()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "query")?;
        let rows: Vec<FactRow> = res.take(0).map_err(Self::map_db_error)?;
        match rows.into_iter().next() {
            None => Ok(None),
            // Surface reconstruction errors as RepoError rather than masking
            // them as `None`: a corrupt row is a real problem the caller must
            // see, not a silent "missing fact" result.
            Some(r) => Ok(Some(r.to_fact(id.clone())?)),
        }
    }

    async fn list_accepted(&self, memory_key: &MemoryKey) -> Result<Vec<Fact>, RepoError> {
        self.list_by_status(memory_key, FactStatus::Accepted).await
    }

    async fn list_pending(&self, memory_key: &MemoryKey) -> Result<Vec<Fact>, RepoError> {
        self.list_by_status(memory_key, FactStatus::Pending).await
    }

    async fn list_memory_keys_for_session(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<MemoryKey>, RepoError> {
        // Cross-namespace scan: every fact whose `source_sessions` array
        // contains `session_id`, projected to the distinct set of
        // `memory_key` values. SurrealDB's `CONTAINS` operator is the
        // stable membership predicate on arrays (the `array::contains`
        // function does NOT exist in SurrealQL — using it raises a parse
        // error). The dedup happens in Rust so a future schema change
        // (e.g. indexing `source_sessions`) does not couple the query
        // shape to a DISTINCT variant that may or may not exist on a given
        // engine version.
        let mut res = self
            .db
            .query(
                "SELECT memory_key FROM fact
                 WHERE source_sessions CONTAINS $sid;",
            )
            .bind(("sid", session_id.as_str().to_string()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "list_memory_keys_for_session")?;

        #[derive(Debug, Deserialize)]
        struct MemoryKeyRow {
            memory_key: String,
        }
        let rows: Vec<MemoryKeyRow> = res.take(0).map_err(Self::map_db_error)?;

        let mut out: Vec<MemoryKey> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for row in rows {
            if !seen.insert(row.memory_key.clone()) {
                continue;
            }
            let mk = MemoryKey::from_raw(&row.memory_key)
                .map_err(|e| RepoError::SerializationFailed(e.to_string()))?;
            out.push(mk);
        }
        Ok(out)
    }

    async fn search_similar(
        &self,
        embedding: Vec<f32>,
        memory_key: &MemoryKey,
        limit: usize,
    ) -> Result<Vec<SearchHit>, RepoError> {
        self.vector_search(embedding, memory_key, limit, VectorSearchScope::Retrieval)
            .await
    }

    /// Semantic-dedup search across pending + accepted facts (no tombstones).
    /// Production override of [`FactRepository::search_for_dedup`] — backs
    /// the extraction pipeline's Layer 2 safety net. See the port docs for
    /// why this must include `pending` (otherwise a circular deadlock keeps
    /// single-source facts stuck below the accept threshold).
    async fn search_for_dedup(
        &self,
        embedding: Vec<f32>,
        memory_key: &MemoryKey,
        limit: usize,
    ) -> Result<Vec<SearchHit>, RepoError> {
        self.vector_search(embedding, memory_key, limit, VectorSearchScope::Dedup)
            .await
    }

    async fn update_heat_batch(
        &self,
        ids: &[FactId],
        memory_key: &MemoryKey,
        heat_base: Heat,
        last_access: Timestamp,
    ) -> Result<(), RepoError> {
        if ids.is_empty() {
            return Ok(());
        }
        // One UPDATE per id, scoped by `memory_key` so a foreign id can never
        // be rewarmed by accident. The SurrealDB Rust SDK does not cleanly
        // accept a record-id array binding (C4); revisit once it does to turn
        // this into a single round-trip.
        let last_access_iso = format_iso(last_access.as_offset_date_time());
        let heat_value = heat_base.value();
        let memory_key_str = memory_key.as_str().to_string();
        for id in ids {
            let mut res = self
                .db
                .query(
                    "UPDATE type::thing('fact', $id) SET
                        heat_base = $heat,
                        last_access_at = <datetime>$last
                     WHERE memory_key = $mk;",
                )
                .bind(("id", id.as_str().to_string()))
                .bind(("heat", heat_value))
                .bind(("last", last_access_iso.clone()))
                .bind(("mk", memory_key_str.clone()))
                .await
                .map_err(Self::map_db_error)?;
            Self::check_errors(&mut res, "update_heat_batch")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Private search-path helpers (split out from the trait impl for clarity)
// ---------------------------------------------------------------------------

/// Status set a vector search is allowed to return.
///
/// [`VectorSearchScope::Retrieval`] backs §3 enrichment: it must see only
/// accepted facts so an unconfirmed pending claim never leaks into a chat
/// context. [`VectorSearchScope::Dedup`] backs the extraction pipeline's
/// Layer 2 safety net: it must additionally include pending facts because
/// the cross-session confirmation that promotes them past the accept
/// threshold can only fire if the search finds them.
#[derive(Clone, Copy)]
enum VectorSearchScope {
    Retrieval,
    Dedup,
}

impl VectorSearchScope {
    /// SQL fragment for the equality-prefiltered brute-force pass.
    fn status_predicate(&self) -> &'static str {
        match self {
            Self::Retrieval => "status = 'accepted'",
            Self::Dedup => "(status = 'accepted' OR status = 'pending')",
        }
    }

    /// Rust predicate mirroring [`Self::status_predicate`] for the
    /// post-filtered HNSW pass.
    fn allows_status(&self, status: &str) -> bool {
        match self {
            Self::Retrieval => status == "accepted",
            Self::Dedup => status == "accepted" || status == "pending",
        }
    }
}

impl SurrealStore {
    /// Two-stage vector search (HNSW + brute-force fallback) shared by
    /// [`FactRepository::search_similar`] and
    /// [`FactRepository::search_for_dedup`]. The scope controls the status
    /// predicate applied in both passes.
    ///
    /// 1. Pull `over_fetch` nearest neighbours from the HNSW index WITHOUT
    ///    equality pre-filters. The AC0 spike proved that combining the
    ///    KNN operator with `memory_key = $mk AND status = 'accepted'`
    ///    returns zero rows on SurrealDB 2.6 — the planner can't fold
    ///    equality predicates into the HNSW traversal. Issuing the KNN
    ///    alone and post-filtering in Rust is the validated workaround.
    ///
    /// 2. Filter the HNSW candidates by memory_key + status + valid_until.
    ///
    /// 3. ALWAYS run the brute-force cosine scan as well, regardless of
    ///    how many hits the HNSW pass returned. The HNSW index is
    ///    approximate: under skewed namespaces or post-deletion churn it
    ///    can miss a true neighbour that an exact cosine scan finds. The
    ///    brute-force pass is the correctness backstop; the cost is one
    ///    extra round-trip per search (acceptable for the SMOS workload —
    ///    every search is per-request, not per-token).
    ///
    /// 4. Merge the two passes (dedup by FactId, prefer the smaller
    ///    `distance`), sort ascending, and truncate to `limit`.
    async fn vector_search(
        &self,
        embedding: Vec<f32>,
        memory_key: &MemoryKey,
        limit: usize,
        scope: VectorSearchScope,
    ) -> Result<Vec<SearchHit>, RepoError> {
        let over_fetch = (limit * 4).max(limit + 8);
        let embedding_f64: Vec<f64> = embedding.iter().map(|&x| x as f64).collect();

        let hnsw_hits = self
            .search_similar_hnsw(&embedding_f64, over_fetch, memory_key, scope)
            .await?;
        let bf_hits = self
            .search_similar_bruteforce(&embedding_f64, memory_key, limit, scope)
            .await?;

        // Dedup by FactId, keeping the hit with the smaller distance on
        // collisions. HNSW and brute-force often return the same FactId
        // with slightly different distances; preferring the smaller one
        // surfaces the most similar fact regardless of which pass found it.
        let mut merged: Vec<SearchHit> = Vec::with_capacity(hnsw_hits.len() + bf_hits.len());
        let mut seen: std::collections::HashMap<FactId, usize> =
            std::collections::HashMap::with_capacity(hnsw_hits.len() + bf_hits.len());
        for hit in hnsw_hits.into_iter().chain(bf_hits) {
            match seen.get(&hit.id) {
                None => {
                    seen.insert(hit.id.clone(), merged.len());
                    merged.push(hit);
                }
                Some(&idx) => {
                    let incumbent = &merged[idx];
                    let incumbent_dist = incumbent.metadata.distance.unwrap_or(f32::INFINITY);
                    let new_dist = hit.metadata.distance.unwrap_or(f32::INFINITY);
                    if new_dist < incumbent_dist {
                        merged[idx] = hit;
                    }
                }
            }
        }
        merged.sort_by(|a, b| {
            a.metadata
                .distance
                .partial_cmp(&b.metadata.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(limit);
        Ok(merged)
    }

    /// HNSW-backed KNN pass — no equality pre-filter, post-filtered in Rust.
    /// Returns up to `over_fetch` hits filtered by `memory_key`,
    /// `scope.allows_status`, and `valid_until = NONE`.
    async fn search_similar_hnsw(
        &self,
        embedding_f64: &[f64],
        over_fetch: usize,
        memory_key: &MemoryKey,
        scope: VectorSearchScope,
    ) -> Result<Vec<SearchHit>, RepoError> {
        // The KNN operator `<|K,EF|>` requires literal integers — SurrealQL's
        // parser rejects a bound parameter in that position. We interpolate
        // the values directly (they are derived from `limit`, which is
        // application-controlled, so this is safe).
        let sql = format!(
            "SELECT id, content, memory_key, status, confidence,
                    valid_until, heat_base, last_access_at,
                    vector::distance::knn() AS distance
             FROM fact
             WHERE embedding <|{over_fetch}, 64|> $embedding
             ORDER BY distance;"
        );
        let mut res = self
            .db
            .query(&sql)
            .bind(("embedding", embedding_f64.to_vec()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "search_similar_hnsw")?;
        let rows: Vec<SearchSimilarRow> = res.take(0).map_err(Self::map_db_error)?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r.to_hit(memory_key))
            .filter(|h| scope.allows_status(&h.metadata.status))
            .filter(|h| h.metadata.valid_until.is_none())
            .collect())
    }

    /// Brute-force cosine pass with equality pre-filters. Slower than HNSW
    /// but immune to the planner limitation that breaks KNN + filter.
    ///
    /// Returns `distance = 1.0 - similarity` so the metric is consistent with
    /// the HNSW pass (smaller distance = more similar) and the merge-sort in
    /// `vector_search` orders both passes by the same key.
    async fn search_similar_bruteforce(
        &self,
        embedding_f64: &[f64],
        memory_key: &MemoryKey,
        limit: usize,
        scope: VectorSearchScope,
    ) -> Result<Vec<SearchHit>, RepoError> {
        // Inline the status predicate so the planner can fold it together
        // with `memory_key` / `valid_until` into one index seek.
        let sql = format!(
            "SELECT id, content, memory_key, status, confidence,
                    valid_until, heat_base, last_access_at,
                    (1.0 - vector::similarity::cosine(embedding, $embedding)) AS distance
             FROM fact
             WHERE memory_key = $mk AND {status_pred} AND valid_until = NONE
             ORDER BY distance ASC
             LIMIT $limit;",
            status_pred = scope.status_predicate()
        );
        let mut res = self
            .db
            .query(&sql)
            .bind(("mk", memory_key.as_str().to_string()))
            .bind(("embedding", embedding_f64.to_vec()))
            .bind(("limit", limit as i64))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "search_similar_bruteforce")?;
        let rows: Vec<SearchSimilarRow> = res.take(0).map_err(Self::map_db_error)?;
        Ok(rows
            .into_iter()
            .filter_map(|r| r.to_hit(memory_key))
            .collect())
    }
}

/// Internal helper for `list_accepted` / `list_pending`.
impl SurrealStore {
    async fn list_by_status(
        &self,
        memory_key: &MemoryKey,
        status: FactStatus,
    ) -> Result<Vec<Fact>, RepoError> {
        let mut res = self
            .db
            .query(
                "SELECT * FROM fact
                 WHERE memory_key = $mk AND status = $status;",
            )
            .bind(("mk", memory_key.as_str().to_string()))
            .bind(("status", status.as_str().to_string()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "query")?;
        let rows: Vec<FactRow> = res.take(0).map_err(Self::map_db_error)?;
        rows.into_iter()
            .map(|r| {
                // Reconstruct the FactId from the row's content. The Fact
                // aggregate's invariant `id == FactId::from_content(content)`
                // is enforced by `Fact::rehydrate`, so this matches the row's
                // Surreal record id (`fact:<fact_id_string>`) by construction.
                let fact_id = FactId::from_content(&r.content);
                r.to_fact(fact_id)
            })
            .collect()
    }
}

/// Raw shape of a `search_similar` result row (subset of FactRow + distance).
#[derive(Debug, Deserialize)]
struct SearchSimilarRow {
    id: surrealdb::RecordId,
    content: String,
    memory_key: String,
    status: String,
    confidence: f32,
    valid_until: Option<String>,
    heat_base: f32,
    last_access_at: String,
    /// Cosine distance as reported by either the HNSW index
    /// (`vector::distance::knn()`) or the brute-force fallback
    /// (`1.0 - vector::similarity::cosine(...)`). Lower = more similar.
    /// Required: both query paths populate it, so a `None` here would
    /// indicate a query-shape regression.
    distance: f64,
}

impl SearchSimilarRow {
    fn to_hit(&self, expected_key: &MemoryKey) -> Option<SearchHit> {
        // Record id `fact:<FactId-string>` → FactId string is the key portion.
        let id_string = self.id.to_string();
        let fact_id_str = id_string.strip_prefix("fact:").unwrap_or(&id_string);
        let fact_id = FactId::from_raw(fact_id_str).ok()?;
        let memory_key = MemoryKey::from_raw(&self.memory_key).ok()?;
        // Defensive: filter out rows whose memory_key drifted (cross-key
        // leakage should be impossible given the post-filter, but cheap to
        // double-check).
        if memory_key != *expected_key {
            return None;
        }
        let metadata = SearchHitMetadata {
            status: self.status.clone(),
            confidence: self.confidence,
            valid_until: self.valid_until.clone(),
            heat_base: self.heat_base,
            last_access_at: parse_iso(&self.last_access_at)
                .map(|ts| ts.as_unix_secs() as f32)
                .unwrap_or(0.0),
            distance: Some(self.distance as f32),
        };
        Some(SearchHit {
            id: fact_id,
            document: self.content.clone(),
            memory_key,
            metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// SessionRepository impl
// ---------------------------------------------------------------------------

impl SessionRepository for SurrealStore {
    async fn get_or_create(
        &self,
        id: &SessionId,
        memory_key: &MemoryKey,
    ) -> Result<SessionState, RepoError> {
        // Atomic upsert via `INSERT ... ON DUPLICATE KEY UPDATE`. This avoids
        // the read-then-create race that two concurrent `get_or_create` calls
        // on a fresh session would otherwise hit (C3): both might miss the
        // SELECT, both would issue CREATE, and one would fail with a
        // record-id conflict.
        //
        // Two round-trips total (upsert + select) — we deliberately read the
        // row back rather than trust an `OUTPUT` clause so the code stays
        // portable across SurrealDB versions.
        let mut res = self
            .db
            .query(
                "INSERT INTO session (id, memory_key, injected_facts, pending_facts,
                                      created_at, last_active)
                 VALUES (type::thing('session', $id), $mk, [], [], time::now(), time::now())
                 ON DUPLICATE KEY UPDATE last_active = time::now();",
            )
            .bind(("id", id.as_str().to_string()))
            .bind(("mk", memory_key.as_str().to_string()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "get_or_create upsert")?;

        // Read back the canonical row to surface the post-upsert state.
        let mut res = self
            .db
            .query("SELECT * FROM session WHERE id = type::thing('session', $id) LIMIT 1;")
            .bind(("id", id.as_str().to_string()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "get_or_create select")?;
        let rows: Vec<SessionRow> = res.take(0).map_err(Self::map_db_error)?;
        let row = rows
            .into_iter()
            .next()
            .ok_or_else(|| RepoError::NotFound(format!("session {}", id)))?;
        row.to_state(id.clone())
    }

    async fn collect_expired(
        &self,
        timeout: Duration,
    ) -> Result<Vec<(SessionId, SessionState)>, RepoError> {
        let timeout_secs = timeout.as_secs() as i64;
        // Use the `<duration>` cast on a string parameter so SurrealDB parses
        // the literal properly. Direct `int * duration` is not supported.
        let timeout_str = format!("{timeout_secs}s");
        let mut res = self
            .db
            .query(
                "SELECT * FROM session
                 WHERE (time::now() - last_active) > <duration>$timeout;",
            )
            .bind(("timeout", timeout_str))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "collect_expired")?;
        let rows: Vec<SessionWithId> = res.take(0).map_err(Self::map_db_error)?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let id_str = r.id.to_string();
            let id_raw = id_str.strip_prefix("session:").unwrap_or(&id_str);
            let Ok(session_id) = SessionId::from_raw(id_raw) else {
                tracing::warn!(record_id = %id_str, "collect_expired: unparseable session id; skipping");
                continue;
            };
            // Skip delete-on-read; POC's `collect_expired` removes the
            // session, but for the Rust port the caller (FinalizeSession in
            // a later slice) decides whether to drop or refresh. We provide
            // `clear_session` for the explicit drop.
            match r.row.to_state(session_id.clone()) {
                Ok(state) => out.push((session_id, state)),
                Err(e) => tracing::warn!(
                    session_id = %session_id,
                    error = %e,
                    "collect_expired: corrupt session row; skipping"
                ),
            }
        }
        Ok(out)
    }

    async fn snapshot_all(&self) -> Result<Vec<(SessionId, SessionState)>, RepoError> {
        let mut res = self
            .db
            .query("SELECT * FROM session;")
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "snapshot_all")?;
        let rows: Vec<SessionWithId> = res.take(0).map_err(Self::map_db_error)?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let id_str = r.id.to_string();
            let id_raw = id_str.strip_prefix("session:").unwrap_or(&id_str);
            let Ok(session_id) = SessionId::from_raw(id_raw) else {
                tracing::warn!(record_id = %id_str, "snapshot_all: unparseable session id; skipping");
                continue;
            };
            match r.row.to_state(session_id.clone()) {
                Ok(state) => out.push((session_id, state)),
                Err(e) => tracing::warn!(
                    session_id = %session_id,
                    error = %e,
                    "snapshot_all: corrupt session row; skipping"
                ),
            }
        }
        Ok(out)
    }

    async fn add_pending(&self, id: &SessionId, fact_ids: &[FactId]) -> Result<(), RepoError> {
        if fact_ids.is_empty() {
            return Ok(());
        }
        let pending: Vec<String> = fact_ids.iter().map(|f| f.as_str().to_string()).collect();
        let mut res = self
            .db
            .query(
                "UPDATE type::thing('session', $id) SET
                    pending_facts = array::union(pending_facts, $pending),
                    last_active = time::now();",
            )
            .bind(("id", id.as_str().to_string()))
            .bind(("pending", pending))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "query")?;
        Ok(())
    }

    async fn remove_pending_owned(
        &self,
        id: &SessionId,
        owned: &[FactId],
    ) -> Result<(), RepoError> {
        if owned.is_empty() {
            return Ok(());
        }
        let owned_strings: Vec<String> = owned.iter().map(|f| f.as_str().to_string()).collect();
        // `array::complement(a, b)` returns the items in `a` that are NOT in
        // `b` (set relative complement, A\B). Do NOT confuse with
        // `array::difference(a, b)` which is the SYMMETRIC difference (A△B):
        // when `pending_facts` is already empty, `array::difference([], b)`
        // returns `b` instead of `[]`, restoring the very ids we are trying
        // to drop. See https://surrealdb.com/docs/reference/query-language/functions/database-functions/array.
        let mut res = self
            .db
            .query(
                "UPDATE type::thing('session', $id) SET
                    pending_facts = array::complement(pending_facts, $owned),
                    last_active = time::now();",
            )
            .bind(("id", id.as_str().to_string()))
            .bind(("owned", owned_strings))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "query")?;
        Ok(())
    }

    async fn clear_session(&self, id: &SessionId) -> Result<(), RepoError> {
        let mut res = self
            .db
            .query("DELETE FROM session WHERE id = type::thing('session', $id);")
            .bind(("id", id.as_str().to_string()))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "query")?;
        Ok(())
    }

    async fn dedup_and_mark(
        &self,
        id: &SessionId,
        _memory_key: &MemoryKey,
        candidate_ids: &[FactId],
    ) -> Result<Vec<FactId>, RepoError> {
        // Auto-create the session row so the transaction's UPDATE finds a
        // target even on a cold-cache session. The row is created with
        // empty injected/pending lists; the transaction then mutates it
        // atomically. Uses `time::now()` directly in SQL so datetimes are
        // stored natively (no string→datetime cast needed).
        let _ = self
            .db
            .query(
                "INSERT INTO session (id, memory_key, injected_facts, pending_facts,
                                      created_at, last_active)
                 VALUES (type::thing('session', $id), $mk, [], [], time::now(), time::now())
                 ON DUPLICATE KEY UPDATE id = id;",
            )
            .bind(("id", id.as_str().to_string()))
            .bind(("mk", _memory_key.as_str().to_string()))
            .await
            .map_err(Self::map_db_error)?;

        let candidates: Vec<String> = candidate_ids
            .iter()
            .map(|f| f.as_str().to_string())
            .collect();

        // SurrealDB uses optimistic concurrency: two transactions that touch
        // the same row may conflict at COMMIT time. We retry up to 5 times
        // with linear backoff — the second attempt almost always succeeds
        // because the first commit has resolved the conflict.
        //
        // Error tracking: a transaction-conflict error is the expected
        // "retry me" signal. Any OTHER error is also recorded into
        // `last_err` (and the loop still retries) because a transient
        // transport / connection error deserves the same retry budget as a
        // conflict — the alternative (return immediately on the first
        // non-conflict error) would surface an intermittent blip as a hard
        // failure even though the second attempt would have succeeded.
        let mut last_err: Option<RepoError> = None;
        for attempt in 0..5u32 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(5 * attempt as u64)).await;
            }
            let mut res = match self
                .db
                .query(DEDUP_AND_MARK_TX)
                .bind(("id", id.as_str().to_string()))
                .bind(("candidates", candidates.clone()))
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    if Self::is_transaction_conflict(&e) {
                        last_err = Some(RepoError::TransactionConflict);
                    } else {
                        // Save the underlying error so the final return is
                        // informative; keep retrying in case the failure is
                        // transient (connection blip, leader handoff, …).
                        last_err = Some(Self::map_db_error(e));
                    }
                    continue;
                }
            };
            if let Err(e) = Self::check_errors(&mut res, "dedup_and_mark") {
                // `check_errors` returns a `RepoError::QueryFailed` whose
                // message embeds the original SurrealDB conflict text.
                // Surface it as the more specific `TransactionConflict`
                // variant so callers (and tests) can match structurally.
                if Self::is_transaction_conflict_message(&e.to_string()) {
                    last_err = Some(RepoError::TransactionConflict);
                } else {
                    last_err = Some(e);
                }
                continue;
            }
            let new_strings: Vec<String> = res.take(0).map_err(Self::map_db_error)?;
            let mut out = Vec::with_capacity(new_strings.len());
            for s in new_strings {
                match FactId::from_raw(&s) {
                    Ok(fid) => out.push(fid),
                    Err(e) => {
                        return Err(RepoError::SerializationFailed(format!(
                            "dedup returned invalid FactId {s:?}: {e}"
                        )));
                    }
                }
            }
            return Ok(out);
        }
        Err(last_err.unwrap_or(RepoError::TransactionConflict))
    }

    async fn save(&self, id: &SessionId, state: &SessionState) -> Result<(), RepoError> {
        let memory_key_str = state.memory_key().as_str().to_string();
        let injected: Vec<String> = state
            .injected_facts()
            .iter()
            .map(|f| f.as_str().to_string())
            .collect();
        let pending: Vec<String> = state
            .pending_facts()
            .iter()
            .map(|f| f.as_str().to_string())
            .collect();
        let created_iso = format_iso(state.created_at().as_offset_date_time());
        let last_active_iso = format_iso(state.last_active().as_offset_date_time());

        let mut res = self
            .db
            .query(
                r#"UPSERT type::thing('session', $id) SET
                       memory_key     = $mk,
                       injected_facts = $injected,
                       pending_facts  = $pending,
                       created_at     = <datetime>$created,
                       last_active    = <datetime>$last_active;"#,
            )
            .bind(("id", id.as_str().to_string()))
            .bind(("mk", memory_key_str))
            .bind(("injected", injected))
            .bind(("pending", pending))
            .bind(("created", created_iso))
            .bind(("last_active", last_active_iso))
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "query")?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct SessionWithId {
    id: surrealdb::RecordId,
    #[serde(flatten)]
    row: SessionRow,
}
