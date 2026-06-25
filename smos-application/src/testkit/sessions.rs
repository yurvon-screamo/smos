//! In-memory `SessionRepository` double.
//!
//! Lifted from the former `finalize_session::tests` local copy (the only
//! in-tree `InMemorySessions`). Unlike that copy, `dedup_and_mark` implements
//! the trait's atomic dedup-and-mark contract (first call returns the
//! candidates and records them; a repeat call with the same candidates returns
//! nothing) — the three use cases migrated onto this testkit never call
//! `dedup_and_mark` (it is exercised only by `enrich_request`), so the
//! stronger semantics are strictly safer.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use smos_domain::{FactId, MemoryKey, SessionId, SessionState};

use crate::errors::RepoError;
use crate::ports::SessionRepository;

#[derive(Default, Clone)]
pub struct InMemorySessions {
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
    injected: Arc<Mutex<HashMap<String, HashSet<String>>>>,
}

impl InMemorySessions {
    pub fn seed(&self, state: SessionState) {
        self.sessions
            .lock()
            .unwrap()
            .insert(state.id().as_str().to_string(), state);
    }

    pub fn pending_of(&self, id: &SessionId) -> Vec<FactId> {
        self.sessions
            .lock()
            .unwrap()
            .get(id.as_str())
            .map(|s| s.pending_facts().to_vec())
            .unwrap_or_default()
    }
}

impl SessionRepository for InMemorySessions {
    async fn get_or_create(
        &self,
        id: &SessionId,
        memory_key: &MemoryKey,
    ) -> Result<SessionState, RepoError> {
        Ok(self
            .sessions
            .lock()
            .unwrap()
            .entry(id.as_str().to_string())
            .or_insert_with(|| {
                SessionState::new(
                    id.clone(),
                    memory_key.clone(),
                    smos_domain::Timestamp::from_unix_secs(0).unwrap(),
                )
            })
            .clone())
    }

    async fn collect_expired(
        &self,
        _timeout: Duration,
    ) -> Result<Vec<(SessionId, SessionState)>, RepoError> {
        Ok(Vec::new())
    }

    async fn snapshot_all(&self) -> Result<Vec<(SessionId, SessionState)>, RepoError> {
        Ok(self
            .sessions
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (SessionId::from_raw(k).unwrap(), v.clone()))
            .collect())
    }

    async fn add_pending(&self, id: &SessionId, fact_ids: &[FactId]) -> Result<(), RepoError> {
        if let Some(state) = self.sessions.lock().unwrap().get_mut(id.as_str()) {
            state.add_pending(fact_ids);
        }
        Ok(())
    }

    async fn remove_pending_owned(
        &self,
        id: &SessionId,
        owned: &[FactId],
    ) -> Result<(), RepoError> {
        if let Some(state) = self.sessions.lock().unwrap().get_mut(id.as_str()) {
            state.remove_owned(owned);
        }
        Ok(())
    }

    async fn clear_session(&self, id: &SessionId) -> Result<(), RepoError> {
        self.sessions.lock().unwrap().remove(id.as_str());
        Ok(())
    }

    async fn dedup_and_mark(
        &self,
        id: &SessionId,
        _memory_key: &MemoryKey,
        candidate_ids: &[FactId],
    ) -> Result<Vec<FactId>, RepoError> {
        let mut injected = self.injected.lock().unwrap();
        let seen = injected.entry(id.as_str().to_string()).or_default();
        let mut new_ids = Vec::new();
        for cid in candidate_ids {
            if seen.insert(cid.as_str().to_string()) {
                new_ids.push(cid.clone());
            }
        }
        Ok(new_ids)
    }

    async fn save(&self, id: &SessionId, state: &SessionState) -> Result<(), RepoError> {
        self.sessions
            .lock()
            .unwrap()
            .insert(id.as_str().to_string(), state.clone());
        Ok(())
    }
}
