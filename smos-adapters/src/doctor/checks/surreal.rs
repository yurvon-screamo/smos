//! SurrealDB connect + migrations + stats snapshot.
//!
//! The doctor reuses [`SurrealStore::connect`] / [`SurrealStore::run_migrations`]
//! so the validation is DRY with the server bootstrap. Stats are pulled
//! via raw SurrealQL through [`SurrealStore::raw_db`] — the production port
//! traits do not expose aggregate counts (the server never needs them),
//! and reusing the store keeps the doctor from re-implementing the
//! SurrealDB connection path.
//!
//! Active-session detection uses the configured inactivity timeout so the
//! "active vs ended" split matches the proxy's own session-watcher policy.

use serde::Deserialize;
use smos_application::errors::RepoError;

use super::super::types::{CheckResult, StatsSnapshot};
use crate::SurrealStore;
use crate::config::{SessionConfig, SurrealConfig};

/// Raw count row projected from SurrealQL `GROUP BY status` aggregates.
#[derive(Debug, Deserialize)]
struct StatusCount {
    status: String,
    count: i64,
}

#[derive(Debug, Deserialize)]
struct TotalCount {
    total: i64,
}

/// Outcome of [`check_surreal`]: either a successful snapshot or a list of
/// failure rows. The Ok branch carries the rows that should still be added
/// to the report (connect + migrations PASS rows).
pub type CheckOutcome = Result<(Vec<CheckResult>, Option<StatsSnapshot>), Vec<CheckResult>>;

/// Connect to SurrealDB, run migrations, and (on success) snapshot stats.
///
/// `session_cfg` is the live `[session]` section from `smos.toml`, NOT a
/// `SessionConfig::default()`. Active-vs-ended split must match the
/// watcher's own policy so the doctor's report agrees with what the live
/// server observes; passing the configured timeout keeps that contract.
pub async fn check_surreal(config: &SurrealConfig, session_cfg: &SessionConfig) -> CheckOutcome {
    let mut rows = Vec::new();

    let store = match SurrealStore::connect(&config.path, &config.namespace, &config.database).await
    {
        Ok(s) => {
            rows.push(CheckResult::pass(
                "SurrealDB",
                format!(
                    "path: {}\nnamespace: {}\ndatabase: {}",
                    config.path, config.namespace, config.database
                ),
            ));
            s
        }
        Err(e) => {
            rows.push(
                CheckResult::fail("SurrealDB", format!("connect failed: {e}"))
                    .with_recommendation("delete ./data/smos.db and retry"),
            );
            return Err(rows);
        }
    };

    match store.run_migrations().await {
        Ok(()) => rows.push(CheckResult::pass(
            "SurrealDB migrations",
            "idempotent, applied",
        )),
        Err(e) => {
            rows.push(
                CheckResult::fail("SurrealDB migrations", format!("apply failed: {e}"))
                    .with_recommendation("check schema in surreal_schema.rs"),
            );
            return Err(rows);
        }
    }

    let snapshot = match snapshot_stats(&store, session_cfg).await {
        Ok(stats) => {
            rows.push(CheckResult::pass(
                "SurrealDB stats",
                format!(
                    "facts: {} (accepted: {}, pending: {}, rejected: {})",
                    stats.total_facts, stats.accepted, stats.pending, stats.rejected,
                ),
            ));
            Some(stats)
        }
        Err(e) => {
            rows.push(CheckResult::warn(
                "SurrealDB stats",
                format!("query failed: {e}"),
            ));
            None
        }
    };

    Ok((rows, snapshot))
}

/// Run the aggregate count queries against the store. Public so the doctor's
/// `--stats` subcommand can reuse it without re-doing connect + migrations.
pub async fn snapshot_stats(
    store: &SurrealStore,
    session_cfg: &SessionConfig,
) -> Result<StatsSnapshot, RepoError> {
    let db = store.raw_db();

    let mut res = db
        .query("SELECT status, count() AS count FROM fact GROUP BY status;")
        .await
        .map_err(|e| RepoError::QueryFailed(e.to_string()))?;
    let rows: Vec<StatusCount> = res
        .take(0)
        .map_err(|e| RepoError::QueryFailed(e.to_string()))?;

    let mut accepted = 0usize;
    let mut pending = 0usize;
    let mut rejected = 0usize;
    for row in rows {
        let c = row.count.max(0) as usize;
        match row.status.as_str() {
            "accepted" => accepted = c,
            "pending" => pending = c,
            "rejected" => rejected = c,
            _ => {}
        }
    }

    let mut res = db
        .query("SELECT count() AS total FROM session GROUP ALL;")
        .await
        .map_err(|e| RepoError::QueryFailed(e.to_string()))?;
    let total_rows: Vec<TotalCount> = res
        .take(0)
        .map_err(|e| RepoError::QueryFailed(e.to_string()))?;
    let total_sessions = total_rows.first().map(|r| r.total).unwrap_or(0).max(0) as usize;

    let timeout_str = format!("{}s", session_cfg.timeout_seconds);
    let mut res = db
        .query(
            "SELECT count() AS total FROM session
             WHERE (time::now() - last_active) < <duration>$timeout
             GROUP ALL;",
        )
        .bind(("timeout", timeout_str))
        .await
        .map_err(|e| RepoError::QueryFailed(e.to_string()))?;
    let active_rows: Vec<TotalCount> = res
        .take(0)
        .map_err(|e| RepoError::QueryFailed(e.to_string()))?;
    let active_sessions = active_rows.first().map(|r| r.total).unwrap_or(0).max(0) as usize;

    Ok(StatsSnapshot {
        total_facts: accepted + pending + rejected,
        accepted,
        pending,
        rejected,
        total_sessions,
        active_sessions,
        ended_sessions: total_sessions.saturating_sub(active_sessions),
    })
}

/// Build a synthetic snapshot for unit tests so the formatter tests do not
/// need a real database.
#[cfg(test)]
pub(crate) fn sample_snapshot_for_test() -> StatsSnapshot {
    StatsSnapshot {
        total_facts: 5,
        accepted: 3,
        pending: 2,
        rejected: 0,
        total_sessions: 1,
        active_sessions: 0,
        ended_sessions: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_snapshot_sums_are_consistent() {
        let s = sample_snapshot_for_test();
        assert_eq!(s.total_facts, s.accepted + s.pending + s.rejected);
        assert_eq!(s.total_sessions, s.active_sessions + s.ended_sessions);
    }
}
