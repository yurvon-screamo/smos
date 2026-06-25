use serde::{Deserialize, Serialize};
use smos_application::errors::RepoError;
use smos_application::types::{SearchHit, SearchHitMetadata};
use smos_domain::{
    Confidence, Embedding, Fact, FactContent, FactId, FactRecord, Heat, MemoryKey, SessionId,
    SessionRecord, SessionState, SourceSessions,
};

use super::mapping::{domain_to_repo, format_iso, parse_fact_status, parse_fact_type, parse_iso};

/// Database projection of a `Fact` row.
///
/// `id` is the Surreal record id (`fact:<fact_id_string>`); the application-
/// level FactId is reconstructed from its key portion. All datetime fields
/// are ISO-8601 strings to keep the row self-describing. `id` is read by
/// serde from SurrealDB responses but the application never consults it —
/// `#[allow(dead_code)]` documents that intentional asymmetry.
#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub(crate) struct FactRow {
    #[serde(skip_serializing)]
    pub(crate) id: Option<surrealdb::RecordId>,
    pub(crate) memory_key: String,
    pub(crate) content: String,
    pub(crate) fact_type: String,
    pub(crate) confidence: f32,
    pub(crate) status: String,
    pub(crate) valid_from: String,
    pub(crate) valid_until: Option<String>,
    pub(crate) extracted_at: String,
    pub(crate) source_sessions: Vec<String>,
    pub(crate) conflicts_with: Vec<String>,
    pub(crate) heat_base: f32,
    pub(crate) last_access_at: String,
    pub(crate) embedding: Option<Vec<f32>>,
}

impl FactRow {
    pub(crate) fn from_fact(fact: &Fact) -> Result<Self, RepoError> {
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

    pub(crate) fn to_fact(&self, id: FactId) -> Result<Fact, RepoError> {
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
        Fact::rehydrate(FactRecord {
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
        })
        .map_err(domain_to_repo)
    }
}

// ---------------------------------------------------------------------------
// Session row <-> domain mapping
// ---------------------------------------------------------------------------

// `id` is the Surreal record id (`session:<session_id_string>`); serde reads
// it from SurrealDB responses but the application reconstructs the session
// id from the typed column below, so the field is intentionally unread.
#[derive(Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub(crate) struct SessionRow {
    #[serde(skip_serializing)]
    id: Option<surrealdb::RecordId>,
    memory_key: String,
    injected_facts: Vec<String>,
    pending_facts: Vec<String>,
    created_at: String,
    last_active: String,
}

impl SessionRow {
    pub(crate) fn to_state(&self, id: SessionId) -> Result<SessionState, RepoError> {
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
        SessionState::rehydrate(SessionRecord {
            id,
            memory_key,
            injected_facts,
            pending_facts,
            created_at,
            last_active,
        })
        .map_err(|e| RepoError::SerializationFailed(e.to_string()))
    }
}

/// Raw shape of a `search_similar` result row (subset of FactRow + distance).
#[derive(Debug, Deserialize)]
pub(crate) struct SearchSimilarRow {
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
    pub(crate) fn to_hit(&self, expected_key: &MemoryKey) -> Option<SearchHit> {
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

#[derive(Debug, Deserialize)]
pub(crate) struct SessionWithId {
    pub(crate) id: surrealdb::RecordId,
    #[serde(flatten)]
    pub(crate) row: SessionRow,
}
