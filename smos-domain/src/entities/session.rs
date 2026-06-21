//! `SessionState` aggregate — per-session bookkeeping.
//!
//! Tracks dedup sets and pending fact ids. The POC's `SessionStore` (thread-safe
//! registry of these) lives in the application layer; the domain layer only
//! owns the value type and its pure update operations.

use crate::error::DomainError;
use crate::value_objects::{FactId, MemoryKey, SessionId, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// In-memory bookkeeping for a single SMOS session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    id: SessionId,
    memory_key: MemoryKey,
    injected_facts: BTreeSet<FactId>,
    pending_facts: Vec<FactId>,
    created_at: Timestamp,
    last_active: Timestamp,
}

impl SessionState {
    pub fn new(id: SessionId, memory_key: MemoryKey, now: Timestamp) -> Self {
        Self {
            id,
            memory_key,
            injected_facts: BTreeSet::new(),
            pending_facts: Vec::new(),
            created_at: now,
            last_active: now,
        }
    }

    /// Rehydrate a `SessionState` from a persisted representation.
    ///
    /// Mirrors [`crate::entities::Fact::rehydrate`]: the only path that lets a
    /// persistence adapter rebuild the full session state — including
    /// `injected_facts`, which has no public mutator — without recomputation.
    /// The `injected_facts` iterator is consumed into a `BTreeSet` (duplicates
    /// collapse); `pending_facts` is taken as-is.
    ///
    /// Returns [`DomainError::InvalidTimestamp`] when `last_active < created_at`:
    /// a session that was last touched before it was created is an impossible
    /// state and indicates a corrupt row. Fail-fast at rehydrate keeps the
    /// invariant pinned in one place rather than relying on every caller to
    /// remember the check.
    pub fn rehydrate(
        id: SessionId,
        memory_key: MemoryKey,
        injected_facts: impl IntoIterator<Item = FactId>,
        pending_facts: Vec<FactId>,
        created_at: Timestamp,
        last_active: Timestamp,
    ) -> Result<Self, DomainError> {
        if last_active < created_at {
            return Err(DomainError::InvalidTimestamp(format!(
                "last_active ({last_active}) < created_at ({created_at})"
            )));
        }
        Ok(Self {
            id,
            memory_key,
            injected_facts: injected_facts.into_iter().collect(),
            pending_facts,
            created_at,
            last_active,
        })
    }

    /// Refresh `last_active` to `now`.
    pub fn touch(&mut self, now: Timestamp) {
        self.last_active = now;
    }

    /// Append fact ids to the pending list, deduplicating in place.
    pub fn add_pending(&mut self, ids: &[FactId]) {
        for id in ids {
            if !self.pending_facts.contains(id) {
                self.pending_facts.push(id.clone());
            }
        }
    }

    /// Remove only the supplied ids (preserving any concurrent additions).
    ///
    /// Used by session-end processing to drain exactly the fact ids it owns,
    /// even if another flow has appended new ones in the meantime.
    pub fn remove_owned(&mut self, owned: &[FactId]) {
        self.pending_facts.retain(|id| !owned.contains(id));
    }

    /// `true` if the session has been inactive for longer than `timeout`.
    pub fn is_expired(&self, now: Timestamp, timeout: std::time::Duration) -> bool {
        let elapsed = now.as_offset_date_time() - self.last_active.as_offset_date_time();
        elapsed > timeout
    }

    pub fn id(&self) -> &SessionId {
        &self.id
    }

    pub fn memory_key(&self) -> &MemoryKey {
        &self.memory_key
    }

    pub fn injected_facts(&self) -> &BTreeSet<FactId> {
        &self.injected_facts
    }

    pub fn pending_facts(&self) -> &[FactId] {
        &self.pending_facts
    }

    pub fn created_at(&self) -> Timestamp {
        self.created_at
    }

    pub fn last_active(&self) -> Timestamp {
        self.last_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn fid(tag: &str) -> FactId {
        FactId::from_content(tag)
    }

    fn sid() -> SessionId {
        SessionId::from_raw("sess_000000000001").unwrap()
    }

    fn key() -> MemoryKey {
        MemoryKey::from_raw("origa").unwrap()
    }

    fn at(secs: i64) -> Timestamp {
        Timestamp::from_unix_secs(secs).unwrap()
    }

    #[test]
    fn new_initialises_defaults() {
        let state = SessionState::new(sid(), key(), at(1000));
        assert!(state.injected_facts().is_empty());
        assert!(state.pending_facts().is_empty());
        assert_eq!(state.created_at().as_unix_secs(), 1000);
        assert_eq!(state.last_active().as_unix_secs(), 1000);
    }

    #[test]
    fn touch_updates_last_active() {
        let mut state = SessionState::new(sid(), key(), at(1000));
        state.touch(at(2000));
        assert_eq!(state.last_active().as_unix_secs(), 2000);
    }

    #[test]
    fn add_pending_appends_unique_ids() {
        let mut state = SessionState::new(sid(), key(), at(0));
        let id1 = fid("first");
        let id2 = fid("second");
        state.add_pending(&[id1.clone(), id2.clone()]);
        assert_eq!(state.pending_facts(), &[id1.clone(), id2.clone()]);
        state.add_pending(&[id1.clone(), id2.clone()]);
        assert_eq!(state.pending_facts().len(), 2);
    }

    #[test]
    fn remove_owned_removes_only_owned_ids() {
        let mut state = SessionState::new(sid(), key(), at(0));
        let id1 = fid("first");
        let id2 = fid("second");
        let id3 = fid("third");
        state.add_pending(&[id1.clone(), id2.clone(), id3.clone()]);
        state.remove_owned(&[id1.clone(), id3.clone()]);
        assert_eq!(state.pending_facts(), std::slice::from_ref(&id2));
    }

    #[test]
    fn remove_owned_preserves_concurrent_additions() {
        let mut state = SessionState::new(sid(), key(), at(0));
        let id1 = fid("first");
        let id2 = fid("second");
        state.add_pending(&[id1.clone(), id2.clone()]);
        // Caller snapshots `owned = [id1, id2]`, then a concurrent flow adds id3.
        let id3 = fid("third");
        state.add_pending(std::slice::from_ref(&id3));
        state.remove_owned(&[id1.clone(), id2.clone()]);
        assert_eq!(state.pending_facts(), std::slice::from_ref(&id3));
    }

    #[test]
    fn is_expired_returns_false_within_30_minutes() {
        let created = at(1_700_000_000);
        let mut state = SessionState::new(sid(), key(), created);
        state.touch(at(1_700_000_000 + 29 * 60));
        assert!(!state.is_expired(at(1_700_000_000 + 30 * 60), Duration::from_secs(30 * 60)));
    }

    #[test]
    fn is_expired_returns_true_after_31_minutes() {
        let created = at(1_700_000_000);
        let mut state = SessionState::new(sid(), key(), created);
        state.touch(at(1_700_000_000));
        let now = at(1_700_000_000 + 31 * 60);
        assert!(state.is_expired(now, Duration::from_secs(30 * 60)));
    }

    #[test]
    fn is_expired_boundary_30_minutes_is_not_expired() {
        let created = at(1_700_000_000);
        let mut state = SessionState::new(sid(), key(), created);
        state.touch(created);
        let now = at(1_700_000_000 + 30 * 60);
        // elapsed > timeout is strict — exactly 30 minutes is still alive.
        assert!(!state.is_expired(now, Duration::from_secs(30 * 60)));
    }

    #[test]
    fn rehydrate_roundtrips_every_field_verbatim() {
        // Persistence adapters call `rehydrate` on read; this test pins the
        // round-trip contract — every field must survive unchanged.
        let mut state = SessionState::new(sid(), key(), at(1_700_000_000));
        state.add_pending(&[fid("p1"), fid("p2")]);
        let injected = [fid("i1"), fid("i2"), fid("i1")]; // duplicate intentional
        state.touch(at(1_700_000_999));

        let rehydrated = SessionState::rehydrate(
            state.id().clone(),
            state.memory_key().clone(),
            injected.iter().cloned(),
            state.pending_facts().to_vec(),
            state.created_at(),
            state.last_active(),
        )
        .expect("valid timestamps: last_active >= created_at");

        assert_eq!(rehydrated.id(), state.id());
        assert_eq!(
            rehydrated.memory_key().as_str(),
            state.memory_key().as_str()
        );
        // `injected` had a duplicate; BTreeSet collapses it to 2 distinct ids.
        assert_eq!(rehydrated.injected_facts().len(), 2);
        assert!(rehydrated.injected_facts().contains(&fid("i1")));
        assert!(rehydrated.injected_facts().contains(&fid("i2")));
        assert_eq!(rehydrated.pending_facts(), state.pending_facts(),);
        assert_eq!(rehydrated.created_at(), state.created_at());
        assert_eq!(rehydrated.last_active(), state.last_active());
    }

    #[test]
    fn rehydrate_rejects_last_active_before_created_at() {
        // A session that was last touched before it was created is an
        // impossible state — `rehydrate` must fail loudly so a corrupt row
        // never reaches the rest of the system.
        let created = at(2_000);
        let last_active = at(1_000);
        let result = SessionState::rehydrate(
            sid(),
            key(),
            std::iter::empty(),
            Vec::new(),
            created,
            last_active,
        );
        let err = result.expect_err("last_active < created_at must fail rehydrate");
        let msg = err.to_string();
        assert!(msg.contains("last_active"), "msg = {msg}");
        assert!(msg.contains("created_at"), "msg = {msg}");
    }

    #[test]
    fn rehydrate_accepts_equal_created_and_last_active() {
        // Boundary: created_at == last_active is the freshly-created state —
        // must NOT trip the `last_active < created_at` invariant.
        let now = at(1_700_000_000);
        let state = SessionState::rehydrate(sid(), key(), std::iter::empty(), Vec::new(), now, now)
            .expect("created_at == last_active is valid");
        assert_eq!(state.created_at(), state.last_active());
    }
}
