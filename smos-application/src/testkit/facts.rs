//! In-memory `FactRepository` double.
//!
//! Unifies the three former in-tree copies (`finalize_session::tests`,
//! `extract_facts_from_response::tests`, `import_opencode_session::tests`).
//! The list/search methods follow the [`FactRepository`] contract literally:
//! `list_accepted` / `list_pending` filter the store by status, and
//! `list_memory_keys_for_session` deduplicates memory keys in insertion order.
//!
//! This is a deliberate, safe widening over the former `extract`/`import`
//! copies, which stubbed those methods to `Ok(Vec::new())`: neither use case
//! calls them (verified — only `finalize` reads the accepted/pending pools),
//! so the finalize-driving real implementation preserves every observable
//! behavior while letting one type back all three suites.
//!
//! `search_similar` intentionally returns an empty `Vec`: none of the three
//! use cases exercise it (they go through `search_for_dedup`), and mirroring
//! the production accepted-only contract here keeps the fake honest. Tests
//! that need to drive Layer 2 dedup script the response via
//! [`InMemoryFacts::script_dedup_hits`].

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use smos_domain::{Fact, FactId, FactStatus, Heat, MemoryKey, SessionId, Timestamp};

use crate::errors::RepoError;
use crate::ports::FactRepository;
use crate::types::SearchHit;

#[derive(Default, Clone)]
pub struct InMemoryFacts {
    store: Arc<Mutex<HashMap<String, Fact>>>,
    /// Optional scripted `search_for_dedup` response (semantic-dedup tests
    /// only). Empty by default so Layer 2 stays inert for exact-match and
    /// new-fact tests.
    dedup_hits: Arc<Mutex<Vec<SearchHit>>>,
}

impl InMemoryFacts {
    /// Insert a fact bypassing `save` (no async, no `Result`) — used to seed
    /// fixtures before the use case runs.
    pub fn seed(&self, fact: Fact) {
        self.store
            .lock()
            .unwrap()
            .insert(fact.id().as_str().to_string(), fact);
    }

    /// Read-only snapshot of a stored fact by id.
    pub fn get_clone(&self, id: &FactId) -> Option<Fact> {
        self.store.lock().unwrap().get(id.as_str()).cloned()
    }

    /// Program the response returned by `search_for_dedup`.
    pub fn script_dedup_hits(&self, hits: Vec<SearchHit>) {
        *self.dedup_hits.lock().unwrap() = hits;
    }

    pub fn is_empty(&self) -> bool {
        self.store.lock().unwrap().is_empty()
    }

    pub fn contains(&self, id: &FactId) -> bool {
        self.store.lock().unwrap().contains_key(id.as_str())
    }
}

impl FactRepository for InMemoryFacts {
    async fn save(&self, fact: &Fact) -> Result<(), RepoError> {
        self.store
            .lock()
            .unwrap()
            .insert(fact.id().as_str().to_string(), fact.clone());
        Ok(())
    }

    async fn get(&self, id: &FactId, _memory_key: &MemoryKey) -> Result<Option<Fact>, RepoError> {
        Ok(self.get_clone(id))
    }

    async fn list_accepted(&self, _memory_key: &MemoryKey) -> Result<Vec<Fact>, RepoError> {
        Ok(self
            .store
            .lock()
            .unwrap()
            .values()
            .filter(|f| f.status() == FactStatus::Accepted)
            .cloned()
            .collect())
    }

    async fn list_pending(&self, _memory_key: &MemoryKey) -> Result<Vec<Fact>, RepoError> {
        Ok(self
            .store
            .lock()
            .unwrap()
            .values()
            .filter(|f| f.status() == FactStatus::Pending)
            .cloned()
            .collect())
    }

    async fn list_memory_keys_for_session(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<MemoryKey>, RepoError> {
        let mut out: Vec<MemoryKey> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for fact in self.store.lock().unwrap().values() {
            if !fact.source_sessions().iter().any(|s| s == session_id) {
                continue;
            }
            let mk_str = fact.memory_key().as_str().to_string();
            if seen.insert(mk_str) {
                out.push(fact.memory_key().clone());
            }
        }
        Ok(out)
    }

    async fn list_memory_keys(&self) -> Result<Vec<MemoryKey>, RepoError> {
        let mut out: Vec<MemoryKey> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for fact in self.store.lock().unwrap().values() {
            let mk_str = fact.memory_key().as_str().to_string();
            if seen.insert(mk_str) {
                out.push(fact.memory_key().clone());
            }
        }
        Ok(out)
    }

    async fn search_similar(
        &self,
        _embedding: Vec<f32>,
        _memory_key: &MemoryKey,
        _limit: usize,
    ) -> Result<Vec<SearchHit>, RepoError> {
        Ok(Vec::new())
    }

    async fn search_for_dedup(
        &self,
        _embedding: Vec<f32>,
        _memory_key: &MemoryKey,
        _limit: usize,
    ) -> Result<Vec<SearchHit>, RepoError> {
        Ok(self.dedup_hits.lock().unwrap().clone())
    }

    async fn update_heat_batch(
        &self,
        _ids: &[FactId],
        _memory_key: &MemoryKey,
        _heat_base: Heat,
        _last_access: Timestamp,
    ) -> Result<(), RepoError> {
        Ok(())
    }
}
