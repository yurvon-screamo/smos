use std::time::Duration;

use smos_application::{errors::RepoError, ports::SessionRepository};
use smos_domain::{FactId, MemoryKey, SessionId, SessionState};

use super::mapping::format_iso;
use super::rows::{SessionRow, SessionWithId};
use super::surreal_store::SurrealStore;
use crate::storage::surreal_schema::DEDUP_AND_MARK_TX;

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
