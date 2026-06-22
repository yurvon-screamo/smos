//! `FactRepository` port — persistence of the `Fact` aggregate.
//!
//! Seven operations covering the full lifecycle: save, get, list-by-status,
//! semantic search, bulk heat updates, and the cross-namespace session
//! provenance scan used by the finalize trigger when the caller knows only
//! the session id. Implementations are responsible for transactional
//! integrity; the trait surface is intentionally minimal so it can be backed
//! by any vector-aware store.

use smos_domain::{Fact, FactId, Heat, MemoryKey, SessionId, Timestamp};

use crate::errors::RepoError;
use crate::types::SearchHit;

/// Persistence boundary for the `Fact` aggregate.
pub trait FactRepository {
    /// Insert or replace a fact (idempotent by `FactId`).
    async fn save(&self, fact: &Fact) -> Result<(), RepoError>;

    /// Look up a fact by id within a memory namespace.
    ///
    /// Cross-namespace lookups return `None` even if the id exists elsewhere,
    /// matching the POC's per-namespace storage layout.
    async fn get(&self, id: &FactId, memory_key: &MemoryKey) -> Result<Option<Fact>, RepoError>;

    /// All currently-accepted facts in a namespace (§3 retrieval pool).
    async fn list_accepted(&self, memory_key: &MemoryKey) -> Result<Vec<Fact>, RepoError>;

    /// All currently-pending facts in a namespace (§5 session-end input).
    async fn list_pending(&self, memory_key: &MemoryKey) -> Result<Vec<Fact>, RepoError>;

    /// Distinct memory_keys whose fact set references `session_id` in
    /// `source_sessions`.
    ///
    /// Used by the manual `--finalize <session_id>` trigger when the operator
    /// does not pass `--memory-key`: the trigger scans every matching
    /// namespace and runs [`crate::use_cases::FinalizeSession`] once per key.
    /// Production callers that already know the memory_key (the watcher, the
    /// CLI with `--memory-key`) skip this scan and call `list_pending`
    /// directly.
    ///
    /// HTTP extraction persists only `fact.source_sessions` — the
    /// `SessionState` row is never written on the request path — so this
    /// method is the only reliable way to discover which namespaces a session
    /// touched without the operator naming one.
    async fn list_memory_keys_for_session(
        &self,
        session_id: &SessionId,
    ) -> Result<Vec<MemoryKey>, RepoError>;

    /// Distinct `memory_key` values across **every** namespace in the store.
    ///
    /// Backs the dreaming agent's `list_memory_keys` discovery tool: the LLM
    /// has no other way to learn which namespaces exist, and without it the
    /// audit prompt's "for every memory_key" instruction sends the model
    /// straight into `count_facts("")` (rejected as an invalid key).
    ///
    /// Required (not defaulted): a future production adapter that forgets to
    /// override it would silently surface zero namespaces — the audit would
    /// emit an empty report with no error. Forcing every implementor to
    /// acknowledge the method makes that failure mode impossible. The
    /// test stubs return `Vec::new()` because they are never exercised on
    /// the dreaming path.
    async fn list_memory_keys(&self) -> Result<Vec<MemoryKey>, RepoError>;

    /// K-nearest-neighbour vector search against accepted facts.
    ///
    /// Returns hits ordered by ascending distance from `embedding`.
    async fn search_similar(
        &self,
        embedding: Vec<f32>,
        memory_key: &MemoryKey,
        limit: usize,
    ) -> Result<Vec<SearchHit>, RepoError>;

    /// Semantic-dedup search across **pending AND accepted** facts (no
    /// tombstones). Used by the extraction pipeline's safety-net Layer 2
    /// (`persist_facts` step 2): a rephrased re-observation hashes to a
    /// different `FactId`, so the exact match misses — but the embedding is
    /// still near-identical and a cross-session confirmation can promote the
    /// existing fact past the accept threshold.
    ///
    /// Without this method, retrieval-only `search_similar` (accepted-only)
    /// creates a **circular deadlock**: a pending fact can reach the accept
    /// threshold only through cross-session confirmation, but confirmation
    /// requires finding the existing fact, which lives in `pending` — a
    /// status `search_similar` filters out by contract.
    ///
    /// The default implementation falls back to `search_similar` so existing
    /// stubs/fakes keep compiling. Production `SurrealStore` overrides it to
    /// include pending facts; tests that exercise Layer 2 must override it
    /// too (otherwise they mask the constraint and give false confidence).
    async fn search_for_dedup(
        &self,
        embedding: Vec<f32>,
        memory_key: &MemoryKey,
        limit: usize,
    ) -> Result<Vec<SearchHit>, RepoError> {
        self.search_similar(embedding, memory_key, limit).await
    }

    /// Bulk-rewrite heat fields for a set of facts (§7 retrieval rewarm).
    ///
    /// `heat_base` and `last_access` are applied uniformly to every supplied
    /// id within the namespace. Ids that do not exist (or live in another
    /// namespace) are silently skipped — callers treat heat as best-effort.
    async fn update_heat_batch(
        &self,
        ids: &[FactId],
        memory_key: &MemoryKey,
        heat_base: Heat,
        last_access: Timestamp,
    ) -> Result<(), RepoError>;
}
