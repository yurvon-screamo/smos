//! `SessionRepository` port — per-session dedup and pending bookkeeping.
//!
//! Mirrors the POC `SessionStore` (`smos/session.py`), with the in-memory
//! `select_new_facts` replaced by an atomic SurrealQL transaction. The trait
//! stays synchronous-agnostic about how atomicity is achieved; the SurrealDB
//! adapter uses a single-statement transaction.

use std::time::Duration;

use smos_domain::{FactId, MemoryKey, SessionId, SessionState};

use crate::errors::RepoError;

/// Persistence + dedup boundary for `SessionState`.
pub trait SessionRepository {
    /// Fetch a session by id, creating it if absent (POC `get_or_create`).
    async fn get_or_create(
        &self,
        id: &SessionId,
        memory_key: &MemoryKey,
    ) -> Result<SessionState, RepoError>;

    /// Return and remove sessions inactive for longer than `timeout`
    /// (POC `collect_expired`). Pairs `(id, state)` are returned so the caller
    /// can drain pending facts before the state is forgotten.
    async fn collect_expired(
        &self,
        timeout: Duration,
    ) -> Result<Vec<(SessionId, SessionState)>, RepoError>;

    /// Read-only snapshot of every session (POC `snapshot_all`).
    async fn snapshot_all(&self) -> Result<Vec<(SessionId, SessionState)>, RepoError>;

    /// Append fact ids to the pending list (deduplicating in place).
    async fn add_pending(&self, id: &SessionId, fact_ids: &[FactId]) -> Result<(), RepoError>;

    /// Remove only the supplied ids, preserving any concurrent additions
    /// (parity with `SessionState::remove_owned`).
    async fn remove_pending_owned(&self, id: &SessionId, owned: &[FactId])
    -> Result<(), RepoError>;

    /// Drop the session entirely.
    async fn clear_session(&self, id: &SessionId) -> Result<(), RepoError>;

    /// ATOMIC dedup + mark in a single transaction (POC `select_new_facts`).
    ///
    /// Returns the subset of `candidate_ids` not previously injected, and
    /// records them as injected so concurrent calls cannot double-inject.
    /// Implementations MUST guarantee that two concurrent calls with the same
    /// candidates return disjoint subsets.
    async fn dedup_and_mark(
        &self,
        id: &SessionId,
        memory_key: &MemoryKey,
        candidate_ids: &[FactId],
    ) -> Result<Vec<FactId>, RepoError>;

    /// Persist a (possibly mutated) `SessionState` back to storage.
    async fn save(&self, id: &SessionId, state: &SessionState) -> Result<(), RepoError>;
}
