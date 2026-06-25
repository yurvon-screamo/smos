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

        // SurrealDB uses optimistic concurrency: two transactions touching the
        // same row may conflict at COMMIT time. We retry ONLY on
        // `RepoError::TransactionConflict` (up to 5 attempts with linear
        // backoff); every other error propagates immediately. The previous
        // behaviour retried on ANY error, which masked genuine failures
        // (malformed query, schema drift, transport fault) behind five
        // redundant attempts and a ~50 ms backoff. The retry policy lives in
        // [`retry_on_conflict`] so it is unit-testable in isolation.
        let id_str = id.as_str().to_string();
        let new_strings: Vec<String> = retry_on_conflict(move || {
            let id_str = id_str.clone();
            let candidates = candidates.clone();
            async move {
                let mut res = match self
                    .db
                    .query(DEDUP_AND_MARK_TX)
                    .bind(("id", id_str))
                    .bind(("candidates", candidates))
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        return if Self::is_transaction_conflict(&e) {
                            Err(RepoError::TransactionConflict)
                        } else {
                            Err(Self::map_db_error(e))
                        };
                    }
                };
                if let Err(e) = Self::check_errors(&mut res, "dedup_and_mark") {
                    return if Self::is_transaction_conflict_message(&e.to_string()) {
                        Err(RepoError::TransactionConflict)
                    } else {
                        Err(e)
                    };
                }
                res.take::<Vec<String>>(0).map_err(Self::map_db_error)
            }
        })
        .await?;

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
        Ok(out)
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
        let created_iso = format_iso(state.created_at().as_offset_date_time())?;
        let last_active_iso = format_iso(state.last_active().as_offset_date_time())?;

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

// ---------------------------------------------------------------------------
// Retry policy for `dedup_and_mark`
// ---------------------------------------------------------------------------

/// Run `attempt` with a bounded retry budget, retrying ONLY on
/// [`RepoError::TransactionConflict`].
///
/// Up to 5 attempts with linear backoff (5 ms, 10 ms, 15 ms, 20 ms between
/// retries). Any non-conflict error propagates immediately — retrying a
/// malformed query or a schema error five times only delays the inevitable
/// failure and hides the real cause behind redundant round-trips. Conflict
/// is the SurrealDB "optimistic concurrency lost the race, try again"
/// signal and is the only error worth burning a retry on.
///
/// Extracted as a free function so the retry policy is unit-testable with a
/// counting fake attempt closure (deterministic, no embedded DB, no real
/// backoff on the happy path).
async fn retry_on_conflict<F, Fut, T>(mut attempt: F) -> Result<T, RepoError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, RepoError>>,
{
    for attempt_index in 0..5u32 {
        if attempt_index > 0 {
            tokio::time::sleep(Duration::from_millis(5 * attempt_index as u64)).await;
        }
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(RepoError::TransactionConflict) => continue,
            Err(other) => return Err(other),
        }
    }
    // Every attempt returned `TransactionConflict`: surface it so callers
    // can branch on the conflict variant rather than a generic query error.
    Err(RepoError::TransactionConflict)
}

#[cfg(test)]
mod retry_policy_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Regression (B4): a non-conflict error must propagate on the FIRST
    // attempt. The pre-B4 loop retried ANY error up to 5×, so against that
    // logic this fake would be invoked 5 times. The assertion pins the new
    // fail-fast-on-non-conflict contract.
    #[tokio::test]
    async fn dedup_propagates_non_conflict_error_without_retry() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let result: Result<u32, RepoError> = retry_on_conflict(move || {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Err(RepoError::QueryFailed("not a conflict".into()))
            }
        })
        .await;

        assert!(
            matches!(result, Err(RepoError::QueryFailed(_))),
            "non-conflict error must propagate unchanged"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a non-conflict error must NOT be retried"
        );
    }

    // Regression (B4): a conflict is retried, and the successful second
    // attempt is returned. Pins the preserved retry-on-conflict behaviour
    // alongside the new fail-fast-on-non-conflict contract.
    #[tokio::test]
    async fn dedup_retries_on_conflict_then_succeeds() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let result: Result<u32, RepoError> = retry_on_conflict(move || {
            let counter = counter.clone();
            async move {
                let n = counter.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    Err(RepoError::TransactionConflict)
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(
            result.unwrap(),
            42,
            "the retried attempt's value is returned"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "conflict is retried exactly once before success"
        );
    }

    // Regression (B4): a conflict that never resolves still terminates after
    // the 5-attempt cap and surfaces `TransactionConflict` (no infinite loop).
    #[tokio::test]
    async fn dedup_caps_conflict_retries_at_five_attempts() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let result: Result<u32, RepoError> = retry_on_conflict(move || {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Err(RepoError::TransactionConflict)
            }
        })
        .await;

        assert!(
            matches!(result, Err(RepoError::TransactionConflict)),
            "exhausted retries surface the conflict variant"
        );
        assert_eq!(
            calls.load(Ordering::SeqCst),
            5,
            "the retry budget is capped at 5 attempts"
        );
    }
}
