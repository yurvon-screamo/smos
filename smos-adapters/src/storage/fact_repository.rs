use serde::Deserialize;
use smos_application::{errors::RepoError, ports::FactRepository, types::SearchHit};
use smos_domain::{Fact, FactId, FactStatus, Heat, MemoryKey, SessionId, Timestamp};

use super::mapping::format_iso;
use super::rows::FactRow;
use super::surreal_store::SurrealStore;
use super::vector_search::VectorSearchScope;

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

    async fn list_memory_keys(&self) -> Result<Vec<MemoryKey>, RepoError> {
        // Same shape as `list_memory_keys_for_session` minus the WHERE filter:
        // a single `SELECT memory_key` pass, with the distinct-set computed in
        // Rust so the query stays portable across engine versions (see the
        // rationale on `list_memory_keys_for_session`).
        let mut res = self
            .db
            .query("SELECT memory_key FROM fact;")
            .await
            .map_err(Self::map_db_error)?;
        Self::check_errors(&mut res, "list_memory_keys")?;

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
        let last_access_iso = format_iso(last_access.as_offset_date_time())?;
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
