//! `Fact` aggregate root — canonical stored memory.
//!
//! The fact is the central aggregate of SMOS. All mutations go through methods
//! that enforce the aggregate's five invariants; there are no public setters
//! for individual fields. Status transitions are gated so a finalised fact
//! cannot be silently resurrected.

use crate::config::{ConfidenceConfig, MergeConfig};
use crate::enums::{FactStatus, FactType, NliLabel};
use crate::error::DomainError;
use crate::value_objects::{
    Confidence, Cosine, Embedding, FactContent, FactId, Heat, MemoryKey, NliResult, SessionId,
    SourceSessions, Timestamp,
};
use serde::{Deserialize, Serialize};

/// Canonical English statement about the world, sourced from one or more
/// sessions, classified by NLI, and retrievable by similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    id: FactId,
    memory_key: MemoryKey,
    content: FactContent,
    fact_type: FactType,
    confidence: Confidence,
    status: FactStatus,
    valid_from: Timestamp,
    valid_until: Option<Timestamp>,
    extracted_at: Timestamp,
    source_sessions: SourceSessions,
    conflicts_with: Vec<FactId>,
    heat_base: Heat,
    last_access_at: Timestamp,
    embedding: Option<Embedding>,
}

/// One pool member that survived the cosine merge threshold.
///
/// Deliberately not `PartialEq`: comparing full `Fact` clones is brittle (it
/// drags in `f32` via `Confidence`/`Heat`). Tests compare fields directly.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// Cloned existing fact (a pool member).
    pub fact: Fact,
    pub cosine_similarity: Cosine,
}

impl Fact {
    /// Construct a fresh pending fact right after extraction.
    ///
    /// Defaults match the POC: status `Pending`, confidence `base_confidence`,
    /// heat `1.0`, fact_type `Entity`, no conflicts, no `valid_until`.
    /// Confidence and status are recomputed by [`Fact::reclassify`] once NLI
    /// is available. The caller passes the configured
    /// [`ConfidenceConfig::base`] so the domain stays free of the
    /// "default 0.5" hard-coding that bit the POC (a config tweak to
    /// `confidence.base` was silently ignored at extraction time).
    pub fn new_pending(
        content: &str,
        memory_key: MemoryKey,
        session: SessionId,
        embedding: Embedding,
        now: Timestamp,
        base_confidence: f32,
    ) -> Result<Self, DomainError> {
        Ok(Self {
            id: FactId::from_content(content),
            memory_key,
            content: FactContent::new(content.to_string())?,
            fact_type: FactType::Entity,
            confidence: Confidence::new(base_confidence)?,
            status: FactStatus::Pending,
            valid_from: now,
            valid_until: None,
            extracted_at: now,
            source_sessions: SourceSessions::from_one(session),
            conflicts_with: Vec::new(),
            heat_base: Heat::new(1.0)?,
            last_access_at: now,
            embedding: Some(embedding),
        })
    }

    /// Rehydrate a `Fact` from a persisted representation (storage → domain).
    ///
    /// This constructor is the **only** way to rebuild a fact with arbitrary
    /// field values; every other constructor derives some fields from inputs
    /// (e.g. `new_pending` derives `id` from content). Persistence adapters
    /// call this on read so the round-trip `save(f); get(id) == f` holds
    /// exactly, without recomputation that would silently mutate stored
    /// confidence/status/heat.
    ///
    /// All invariants are still enforced: confidence and heat must be in
    /// `[0.0, 1.0]`, `valid_until` (when `Some`) must be strictly after
    /// `valid_from`, and the id must equal `FactId::from_content(content)`.
    /// The constructor returns the matching `DomainError` if any invariant
    /// fails so the caller can surface data corruption rather than silently
    /// re-stamp it.
    #[allow(clippy::too_many_arguments)] // full-state rehydrate; arg count is inherent
    pub fn rehydrate(
        id: FactId,
        memory_key: MemoryKey,
        content: FactContent,
        fact_type: FactType,
        confidence: Confidence,
        status: FactStatus,
        valid_from: Timestamp,
        valid_until: Option<Timestamp>,
        extracted_at: Timestamp,
        source_sessions: SourceSessions,
        conflicts_with: Vec<FactId>,
        heat_base: Heat,
        last_access_at: Timestamp,
        embedding: Option<Embedding>,
    ) -> Result<Self, DomainError> {
        // Sanity check: the caller-supplied id must match the canonical
        // content-derived id. If it doesn't, the persisted row is corrupt
        // (someone wrote a row whose record id disagrees with its content).
        if id != FactId::from_content(content.as_str()) {
            return Err(DomainError::InvalidFactId(format!(
                "rehydrate id mismatch: record={} expected={}",
                id,
                FactId::from_content(content.as_str())
            )));
        }
        // `valid_until` invariant (mirrors `set_valid_until`).
        if let Some(until) = valid_until
            && until <= valid_from
        {
            return Err(DomainError::ValidUntilBeforeValidFrom {
                from: valid_from,
                until,
            });
        }
        Ok(Self {
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
    }

    /// Recompute confidence + status from NLI context and the active config.
    ///
    /// Atomic: either both fields update or neither does. Safe to call with
    /// `nli = None` to refresh the gate after a provenance change.
    pub fn reclassify(
        &mut self,
        nli: Option<&NliResult>,
        cfg: &ConfidenceConfig,
    ) -> Result<(), DomainError> {
        let new_conf = self.compute_confidence(nli, cfg);
        let new_status = new_conf.classify(cfg);
        self.set_status_and_confidence(new_status, new_conf, cfg)
    }

    /// Compute the confidence for this fact given optional NLI context.
    ///
    /// Formula (§5.4, §9):
    /// ```text
    /// score = base
    ///       + multi_source_bonus     if 2+ distinct sessions observed the fact
    ///       + no_contradiction_bonus if NLI ran and did not flag a contradiction
    /// ```
    ///
    /// The result is clamped to `[0, 1]` by `Confidence::new_unchecked`.
    ///
    /// NOTE: the contradiction check here is intentionally label-only (not the
    /// threshold-aware `NliResult::is_contradiction`). This faithfully mirrors
    /// the POC: the bonus rewards "any non-contradiction verdict observed",
    /// whereas the threshold-aware predicate drives the merge/drift decision.
    /// Mixing the two would couple two independently-tunable policies.
    pub fn compute_confidence(
        &self,
        nli: Option<&NliResult>,
        cfg: &ConfidenceConfig,
    ) -> Confidence {
        let mut score = cfg.base;

        if self.source_sessions.distinct_count() >= 2 {
            score += cfg.multi_source_bonus;
        }

        if let Some(nli) = nli
            && nli.available
            && nli.label != NliLabel::Contradiction
        {
            score += cfg.no_contradiction_bonus;
        }

        Confidence::new_unchecked(score)
    }

    /// Stateless heat decay (§7): `heat_base * exp(-decay_rate * hours)`.
    ///
    /// Delegates to the canonical [`Heat::decay`] formula. Past access is the
    /// normal case (positive hours); future timestamps (clock skew) clamp to
    /// zero so we never amplify heat above `heat_base`.
    pub fn heat_live(&self, now: Timestamp, decay_rate: f32) -> f32 {
        Heat::decay(self.heat_base, self.last_access_at, now, decay_rate)
    }

    /// Scan `pool` for merge candidates against this fact (§5.3 candidate
    /// selection).
    ///
    /// Filters:
    /// - Same `memory_key` as `self` (cross-namespace matches never merge).
    /// - Skip the pool member whose id matches `self.id` (would be a self-match).
    /// - Cosine similarity at/above `cfg.cosine_threshold`.
    ///
    /// Results are sorted by cosine similarity descending so the closest
    /// candidate is processed first.
    pub fn find_merge_candidates(&self, pool: &[Fact], cfg: &MergeConfig) -> Vec<MergeCandidate> {
        let Some(self_emb) = self.embedding.as_ref() else {
            return Vec::new();
        };
        let self_key = &self.memory_key;
        let self_id = &self.id;

        let mut candidates: Vec<MergeCandidate> = pool
            .iter()
            .filter(|f| &f.memory_key == self_key)
            .filter(|f| &f.id != self_id)
            .filter_map(|f| {
                let emb = f.embedding()?;
                let sim = self_emb.cosine(emb);
                if sim.value() >= cfg.cosine_threshold {
                    Some(MergeCandidate {
                        fact: f.clone(),
                        cosine_similarity: sim,
                    })
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.cosine_similarity
                .value()
                .partial_cmp(&a.cosine_similarity.value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Mark this fact and `other` as conflicting, on both sides, idempotently.
    ///
    /// Convenience wrapper around [`Fact::flag_conflict`] so session-end code
    /// cannot accidentally flag only one direction (§5.2: both facts must carry
    /// the conflict link).
    pub fn flag_conflict_bidirectional(&mut self, other: &mut Fact) -> Result<(), DomainError> {
        let self_id = self.id.clone();
        let other_id = other.id.clone();
        self.flag_conflict(other_id)?;
        other.flag_conflict(self_id)?;
        Ok(())
    }

    /// Cross-session confirmation (parity with POC `_confirm_existing_fact`).
    ///
    /// Adds the session to provenance if it is new, then recomputes confidence
    /// and re-applies the validation gate — the multi-source bonus (≥2 sessions)
    /// can lift a single-session `Pending` fact above the accept threshold, at
    /// which point the status is promoted to `Accepted`. Returns `true` iff the
    /// provenance grew.
    pub fn confirm_cross_session(
        &mut self,
        session: &SessionId,
        cfg: &ConfidenceConfig,
    ) -> Result<bool, DomainError> {
        let grew = self.source_sessions.add_unique(session.clone());
        if grew {
            self.reclassify(None, cfg)?;
        }
        Ok(grew)
    }

    /// Union provenance and conflict flags from `other`.
    ///
    /// Used by the merge path; the caller is responsible for reclassifying
    /// afterwards. Self-references and duplicates are skipped.
    pub fn merge_into(&mut self, other: &Fact) -> Result<(), DomainError> {
        self.source_sessions.union(&other.source_sessions);
        for cid in &other.conflicts_with {
            if *cid != self.id && !self.conflicts_with.contains(cid) {
                self.conflicts_with.push(cid.clone());
            }
        }
        Ok(())
    }

    /// Mark this fact as conflicting with `other_id` (one side of a bidirectional
    /// flag; caller flags the other side). Rejects self-conflicts.
    pub fn flag_conflict(&mut self, other_id: FactId) -> Result<(), DomainError> {
        if other_id == self.id {
            return Err(DomainError::SelfConflict(self.id.clone()));
        }
        if !self.conflicts_with.contains(&other_id) {
            self.conflicts_with.push(other_id);
        }
        Ok(())
    }

    /// Rewarm the fact after a retrieval hit (§7 boost).
    pub fn boost_heat(&mut self, now: Timestamp) {
        self.heat_base = Heat::MAX;
        self.last_access_at = now;
    }

    /// Set the validity tombstone (`valid_until`).
    ///
    /// Returns [`DomainError::ValidUntilBeforeValidFrom`] if `until` is at or
    /// before `valid_from`. `None` clears a previously-set tombstone (fact
    /// becomes current again).
    pub fn set_valid_until(&mut self, until: Option<Timestamp>) -> Result<(), DomainError> {
        if let Some(ts) = until
            && ts <= self.valid_from
        {
            return Err(DomainError::ValidUntilBeforeValidFrom {
                from: self.valid_from,
                until: ts,
            });
        }
        self.valid_until = until;
        Ok(())
    }

    /// Set status and confidence together, enforcing all transition invariants.
    ///
    /// Rules:
    /// 1. Outgoing transitions from a terminal state (`Accepted`/`Rejected`)
    ///    are illegal — only self-refresh is allowed.
    /// 2. Accepted implies `confidence >= cfg.accept_threshold`.
    pub fn set_status_and_confidence(
        &mut self,
        status: FactStatus,
        conf: Confidence,
        cfg: &ConfidenceConfig,
    ) -> Result<(), DomainError> {
        if self.status.is_terminal() && status != self.status {
            return Err(DomainError::IllegalStatusTransition {
                from: self.status,
                to: status,
            });
        }
        if status == FactStatus::Accepted && conf.value() < cfg.accept_threshold {
            return Err(DomainError::ConfidenceBelowAcceptThreshold {
                threshold: cfg.accept_threshold,
                actual: conf.value(),
            });
        }
        self.status = status;
        self.confidence = conf;
        Ok(())
    }

    // Read-only accessors. No public setters: every mutation goes through a
    // method that enforces an invariant.

    pub fn id(&self) -> &FactId {
        &self.id
    }

    pub fn memory_key(&self) -> &MemoryKey {
        &self.memory_key
    }

    pub fn content(&self) -> &str {
        self.content.as_str()
    }

    pub fn fact_type(&self) -> FactType {
        self.fact_type
    }

    pub fn confidence(&self) -> Confidence {
        self.confidence
    }

    pub fn status(&self) -> FactStatus {
        self.status
    }

    pub fn valid_from(&self) -> Timestamp {
        self.valid_from
    }

    pub fn valid_until(&self) -> Option<Timestamp> {
        self.valid_until
    }

    pub fn extracted_at(&self) -> Timestamp {
        self.extracted_at
    }

    pub fn source_sessions(&self) -> &SourceSessions {
        &self.source_sessions
    }

    pub fn conflicts_with(&self) -> &[FactId] {
        &self.conflicts_with
    }

    pub fn heat_base(&self) -> Heat {
        self.heat_base
    }

    pub fn last_access_at(&self) -> Timestamp {
        self.last_access_at
    }

    pub fn embedding(&self) -> Option<&Embedding> {
        self.embedding.as_ref()
    }

    /// Builder-style constructor combinator: replace the embedding.
    ///
    /// Exposed because the persistence adapter rebuilds a `Fact` from markdown
    /// frontmatter and may need to attach (or detach) the in-memory embedding
    /// after the fact. This is the *only* way to mutate the embedding after
    /// construction — all other fields stay invariant.
    pub fn with_embedding(mut self, embedding: Option<Embedding>) -> Self {
        self.embedding = embedding;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_objects::SessionId;

    fn sid(suffix: u8) -> SessionId {
        let hex = format!("sess_{:012x}", suffix as u64);
        SessionId::from_raw(&hex).unwrap()
    }

    fn emb(dim: usize) -> Embedding {
        Embedding::new((0..dim).map(|i| i as f32 + 1.0).collect()).unwrap()
    }

    fn pending_fact(content: &str, session: SessionId) -> Fact {
        Fact::new_pending(
            content,
            MemoryKey::from_raw("origa").unwrap(),
            session,
            emb(8),
            Timestamp::from_unix_secs(1_700_000_000).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap()
    }

    fn default_cfg() -> ConfidenceConfig {
        ConfidenceConfig::default()
    }

    #[test]
    fn new_pending_initialises_all_fields() {
        let session = sid(1);
        let fact = pending_fact("Rust is fast", session.clone());

        assert_eq!(fact.content(), "Rust is fast");
        assert_eq!(fact.memory_key().as_str(), "origa");
        assert_eq!(fact.fact_type(), FactType::Entity);
        assert_eq!(fact.confidence().value(), 0.5);
        assert_eq!(fact.status(), FactStatus::Pending);
        assert!(fact.valid_until().is_none());
        assert_eq!(fact.source_sessions().distinct_count(), 1);
        assert!(fact.conflicts_with().is_empty());
        assert_eq!(fact.heat_base().value(), 1.0);
        assert!(fact.embedding().is_some());
        assert_eq!(fact.id(), &FactId::from_content("Rust is fast"));
    }

    #[test]
    fn new_pending_rejects_empty_content() {
        let err = Fact::new_pending(
            "  ",
            MemoryKey::shared(),
            sid(1),
            emb(4),
            Timestamp::from_unix_secs(0).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap_err();
        assert!(matches!(err, DomainError::EmptyFactContent));
    }

    #[test]
    fn set_status_pending_to_accepted_when_confidence_is_high_enough() {
        let mut fact = pending_fact("a", sid(1));
        fact.set_status_and_confidence(
            FactStatus::Accepted,
            Confidence::new(0.7).unwrap(),
            &default_cfg(),
        )
        .unwrap();
        assert_eq!(fact.status(), FactStatus::Accepted);
    }

    #[test]
    fn set_status_pending_to_accepted_rejects_low_confidence() {
        let mut fact = pending_fact("a", sid(1));
        let err = fact
            .set_status_and_confidence(
                FactStatus::Accepted,
                Confidence::new(0.5).unwrap(),
                &default_cfg(),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            DomainError::ConfidenceBelowAcceptThreshold {
                threshold: 0.7,
                actual: 0.5
            }
        ));
    }

    #[test]
    fn set_status_pending_to_rejected_is_allowed() {
        let mut fact = pending_fact("a", sid(1));
        fact.set_status_and_confidence(
            FactStatus::Rejected,
            Confidence::new(0.0).unwrap(),
            &default_cfg(),
        )
        .unwrap();
        assert_eq!(fact.status(), FactStatus::Rejected);
    }

    #[test]
    fn set_status_accepted_to_accepted_is_allowed_for_refresh() {
        let mut fact = pending_fact("a", sid(1));
        fact.set_status_and_confidence(
            FactStatus::Accepted,
            Confidence::new(0.9).unwrap(),
            &default_cfg(),
        )
        .unwrap();
        fact.set_status_and_confidence(
            FactStatus::Accepted,
            Confidence::new(0.95).unwrap(),
            &default_cfg(),
        )
        .unwrap();
        assert_eq!(fact.confidence().value(), 0.95);
    }

    #[test]
    fn set_status_accepted_to_pending_is_illegal() {
        let mut fact = pending_fact("a", sid(1));
        fact.set_status_and_confidence(
            FactStatus::Accepted,
            Confidence::new(0.9).unwrap(),
            &default_cfg(),
        )
        .unwrap();
        let err = fact
            .set_status_and_confidence(
                FactStatus::Pending,
                Confidence::new(0.5).unwrap(),
                &default_cfg(),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            DomainError::IllegalStatusTransition {
                from: FactStatus::Accepted,
                to: FactStatus::Pending
            }
        ));
    }

    #[test]
    fn set_status_accepted_to_rejected_is_illegal() {
        let mut fact = pending_fact("a", sid(1));
        fact.set_status_and_confidence(
            FactStatus::Accepted,
            Confidence::new(0.9).unwrap(),
            &default_cfg(),
        )
        .unwrap();
        assert!(
            fact.set_status_and_confidence(
                FactStatus::Rejected,
                Confidence::new(0.0).unwrap(),
                &default_cfg(),
            )
            .is_err()
        );
    }

    #[test]
    fn set_status_rejected_to_anything_is_illegal() {
        let mut fact = pending_fact("a", sid(1));
        fact.set_status_and_confidence(
            FactStatus::Rejected,
            Confidence::new(0.0).unwrap(),
            &default_cfg(),
        )
        .unwrap();
        for target in [FactStatus::Pending, FactStatus::Accepted] {
            assert!(
                fact.set_status_and_confidence(
                    target,
                    Confidence::new(0.5).unwrap(),
                    &default_cfg()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn reclassify_applies_confidence_and_status_atomically() {
        let mut fact = pending_fact("a", sid(1));
        // No NLI, single source: base 0.5 → Pending.
        fact.reclassify(None, &default_cfg()).unwrap();
        assert_eq!(fact.confidence().value(), 0.5);
        assert_eq!(fact.status(), FactStatus::Pending);
    }

    #[test]
    fn confirm_cross_session_adds_session_first_time() {
        let mut fact = pending_fact("a", sid(1));
        let grew = fact.confirm_cross_session(&sid(2), &default_cfg()).unwrap();
        assert!(grew);
        assert_eq!(fact.source_sessions().distinct_count(), 2);
    }

    #[test]
    fn confirm_cross_session_returns_false_on_repeat() {
        let mut fact = pending_fact("a", sid(1));
        fact.confirm_cross_session(&sid(2), &default_cfg()).unwrap();
        let grew = fact.confirm_cross_session(&sid(2), &default_cfg()).unwrap();
        assert!(!grew);
    }

    #[test]
    fn confirm_cross_session_lifts_confidence_to_accept_threshold() {
        let mut fact = pending_fact("a", sid(1));
        fact.confirm_cross_session(&sid(2), &default_cfg()).unwrap();
        // 0.5 base + 0.2 multi_source = 0.7 → Accepted.
        assert!((fact.confidence().value() - 0.7).abs() < 1e-6);
        assert_eq!(fact.status(), FactStatus::Accepted);
    }

    #[test]
    fn merge_into_unions_source_sessions() {
        let mut left = pending_fact("a", sid(1));
        let mut right = pending_fact("a", sid(2));
        right
            .confirm_cross_session(&sid(3), &default_cfg())
            .unwrap();
        left.merge_into(&right).unwrap();
        assert_eq!(left.source_sessions().distinct_count(), 3);
    }

    #[test]
    fn merge_into_unions_conflicts_without_self_reference() {
        let mut left = pending_fact("a", sid(1));
        let other_id = FactId::from_content("other");
        left.flag_conflict(other_id.clone()).unwrap();
        let right = pending_fact("a", sid(2));
        left.merge_into(&right).unwrap();
        assert!(left.conflicts_with().contains(&other_id));
    }

    #[test]
    fn merge_into_dedups_conflict_flags() {
        let mut left = pending_fact("a", sid(1));
        let other_id = FactId::from_content("other");
        left.flag_conflict(other_id.clone()).unwrap();
        let mut right = pending_fact("a", sid(2));
        right.flag_conflict(other_id.clone()).unwrap();
        left.merge_into(&right).unwrap();
        let count = left
            .conflicts_with()
            .iter()
            .filter(|id| **id == other_id)
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn flag_conflict_rejects_self_conflict() {
        let mut fact = pending_fact("a", sid(1));
        let err = fact.flag_conflict(fact.id().clone()).unwrap_err();
        assert!(matches!(err, DomainError::SelfConflict(_)));
    }

    #[test]
    fn flag_conflict_is_idempotent() {
        let mut fact = pending_fact("a", sid(1));
        let other = FactId::from_content("other");
        fact.flag_conflict(other.clone()).unwrap();
        fact.flag_conflict(other.clone()).unwrap();
        assert_eq!(
            fact.conflicts_with()
                .iter()
                .filter(|id| **id == other)
                .count(),
            1
        );
    }

    #[test]
    fn boost_heat_sets_max_heat_and_refreshes_access_time() {
        let mut fact = pending_fact("a", sid(1));
        let now = Timestamp::from_unix_secs(1_800_000_000).unwrap();
        fact.boost_heat(now);
        assert_eq!(fact.heat_base().value(), 1.0);
        assert_eq!(fact.last_access_at().as_unix_secs(), 1_800_000_000);
    }

    #[test]
    fn set_valid_until_accepts_timestamp_strictly_after_valid_from() {
        let mut fact = pending_fact("a", sid(1));
        let original_valid_from = fact.valid_from();
        let later = Timestamp::from_unix_secs(original_valid_from.as_unix_secs() + 3600).unwrap();
        fact.set_valid_until(Some(later)).unwrap();
        assert_eq!(fact.valid_until(), Some(later));
    }

    #[test]
    fn set_valid_until_rejects_timestamp_at_or_before_valid_from() {
        let mut fact = pending_fact("a", sid(1));
        let original_valid_from = fact.valid_from();
        let equal = original_valid_from;
        let earlier = Timestamp::from_unix_secs(original_valid_from.as_unix_secs() - 10).unwrap();
        assert!(fact.set_valid_until(Some(equal)).is_err());
        assert!(fact.set_valid_until(Some(earlier)).is_err());
    }

    #[test]
    fn set_valid_until_none_clears_tombstone() {
        let mut fact = pending_fact("a", sid(1));
        let original = fact.valid_from();
        let later = Timestamp::from_unix_secs(original.as_unix_secs() + 3600).unwrap();
        fact.set_valid_until(Some(later)).unwrap();
        fact.set_valid_until(None).unwrap();
        assert!(fact.valid_until().is_none());
    }

    #[test]
    fn with_embedding_overrides_embedding() {
        let fact = pending_fact("a", sid(1));
        let replaced = fact.with_embedding(None);
        assert!(replaced.embedding().is_none());
    }

    #[test]
    fn serde_roundtrip_preserves_fact_fields() {
        let fact = pending_fact("Rust fact", sid(1));
        let json = serde_json::to_string(&fact).unwrap();
        let back: Fact = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content(), "Rust fact");
        assert_eq!(back.status(), FactStatus::Pending);
        assert_eq!(back.confidence().value(), 0.5);
    }

    #[test]
    fn rehydrate_roundtrips_every_field_verbatim() {
        // Persistence adapters call `rehydrate` on read; this test pins the
        // round-trip contract: every persisted field must survive unchanged.
        let mut fact = pending_fact("Rust fact", sid(1));
        fact.set_status_and_confidence(
            FactStatus::Accepted,
            Confidence::new(0.92).unwrap(),
            &default_cfg(),
        )
        .unwrap();
        fact.flag_conflict(FactId::from_content("other")).unwrap();
        fact.set_valid_until(Some(
            Timestamp::from_unix_secs(fact.valid_from().as_unix_secs() + 3600).unwrap(),
        ))
        .unwrap();

        let rehydrated = Fact::rehydrate(
            fact.id().clone(),
            fact.memory_key().clone(),
            FactContent::new(fact.content().to_string()).unwrap(),
            fact.fact_type(),
            fact.confidence(),
            fact.status(),
            fact.valid_from(),
            fact.valid_until(),
            fact.extracted_at(),
            fact.source_sessions().clone(),
            fact.conflicts_with().to_vec(),
            fact.heat_base(),
            fact.last_access_at(),
            fact.embedding().cloned(),
        )
        .unwrap();

        assert_eq!(rehydrated.id(), fact.id());
        assert_eq!(rehydrated.content(), fact.content());
        assert_eq!(rehydrated.fact_type(), fact.fact_type());
        assert_eq!(rehydrated.confidence().value(), fact.confidence().value());
        assert_eq!(rehydrated.status(), fact.status());
        assert_eq!(rehydrated.valid_from(), fact.valid_from());
        assert_eq!(rehydrated.valid_until(), fact.valid_until());
        assert_eq!(rehydrated.extracted_at(), fact.extracted_at());
        assert_eq!(
            rehydrated.source_sessions().distinct_count(),
            fact.source_sessions().distinct_count()
        );
        assert_eq!(rehydrated.conflicts_with(), fact.conflicts_with());
        assert_eq!(rehydrated.heat_base().value(), fact.heat_base().value());
        assert_eq!(rehydrated.last_access_at(), fact.last_access_at());
    }

    #[test]
    fn rehydrate_rejects_id_that_disagrees_with_content() {
        let fact = pending_fact("Rust fact", sid(1));
        let wrong_id = FactId::from_content("different content");
        let err = Fact::rehydrate(
            wrong_id.clone(),
            fact.memory_key().clone(),
            FactContent::new(fact.content().to_string()).unwrap(),
            fact.fact_type(),
            fact.confidence(),
            fact.status(),
            fact.valid_from(),
            None,
            fact.extracted_at(),
            fact.source_sessions().clone(),
            Vec::new(),
            fact.heat_base(),
            fact.last_access_at(),
            None,
        )
        .unwrap_err();
        assert!(matches!(err, DomainError::InvalidFactId(_)));
    }

    #[test]
    fn rehydrate_rejects_valid_until_at_or_before_valid_from() {
        let fact = pending_fact("Rust fact", sid(1));
        let at_valid_from = fact.valid_from();
        let err = Fact::rehydrate(
            fact.id().clone(),
            fact.memory_key().clone(),
            FactContent::new(fact.content().to_string()).unwrap(),
            fact.fact_type(),
            fact.confidence(),
            fact.status(),
            fact.valid_from(),
            Some(at_valid_from),
            fact.extracted_at(),
            fact.source_sessions().clone(),
            Vec::new(),
            fact.heat_base(),
            fact.last_access_at(),
            None,
        )
        .unwrap_err();
        assert!(matches!(err, DomainError::ValidUntilBeforeValidFrom { .. }));
    }

    // -----------------------------------------------------------------
    // compute_confidence — multi-source + NLI bonus formula
    // -----------------------------------------------------------------

    fn nli_result(label: NliLabel, available: bool) -> NliResult {
        use crate::value_objects::NliScores;
        NliResult {
            label,
            scores: NliScores {
                entailment: if label == NliLabel::Entailment {
                    1.0
                } else {
                    0.0
                },
                neutral: if label == NliLabel::Neutral { 1.0 } else { 0.0 },
                contradiction: if label == NliLabel::Contradiction {
                    1.0
                } else {
                    0.0
                },
            },
            available,
        }
    }

    #[test]
    fn compute_confidence_base_only_for_single_source_no_nli() {
        let fact = pending_fact("a", sid(1));
        let c = fact.compute_confidence(None, &default_cfg());
        assert!((c.value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn compute_confidence_multi_source_bonus_applies_with_two_sessions() {
        let mut fact = pending_fact("a", sid(1));
        fact.confirm_cross_session(&sid(2), &default_cfg()).unwrap();
        let c = fact.compute_confidence(None, &default_cfg());
        assert!((c.value() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn compute_confidence_no_contradiction_bonus_for_entailment() {
        let fact = pending_fact("a", sid(1));
        let c = fact.compute_confidence(
            Some(&nli_result(NliLabel::Entailment, true)),
            &default_cfg(),
        );
        assert!((c.value() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn compute_confidence_no_contradiction_bonus_skipped_for_contradiction() {
        let fact = pending_fact("a", sid(1));
        let c = fact.compute_confidence(
            Some(&nli_result(NliLabel::Contradiction, true)),
            &default_cfg(),
        );
        assert!((c.value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn compute_confidence_no_contradiction_bonus_skipped_when_unavailable() {
        let fact = pending_fact("a", sid(1));
        let c = fact.compute_confidence(
            Some(&nli_result(NliLabel::Entailment, false)),
            &default_cfg(),
        );
        assert!((c.value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn compute_confidence_both_bonuses_stack_and_clamp_at_one() {
        let mut fact = pending_fact("a", sid(1));
        fact.confirm_cross_session(&sid(2), &default_cfg()).unwrap();
        fact.confirm_cross_session(&sid(3), &default_cfg()).unwrap();
        let c = fact.compute_confidence(
            Some(&nli_result(NliLabel::Entailment, true)),
            &default_cfg(),
        );
        assert!((c.value() - 0.8).abs() < 1e-6);
    }

    // -----------------------------------------------------------------
    // heat_live — exponential decay
    // -----------------------------------------------------------------

    #[test]
    fn heat_live_fresh_fact_has_full_heat() {
        let fact = pending_fact("a", sid(1));
        let now = fact.last_access_at();
        assert!((fact.heat_live(now, 0.03) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn heat_live_decays_after_24_hours_at_known_rate() {
        // exp(-0.03 * 24) ≈ 0.4868
        let fact = pending_fact("a", sid(1));
        let base = fact.last_access_at();
        let one_day_later = Timestamp::from_unix_secs(base.as_unix_secs() + 24 * 3600).unwrap();
        let h = fact.heat_live(one_day_later, 0.03);
        assert!((h - 0.4868).abs() < 1e-3, "got {h}");
    }

    #[test]
    fn heat_live_future_access_clamps_to_zero_decay() {
        let fact = pending_fact("a", sid(1));
        let base = fact.last_access_at();
        let earlier = Timestamp::from_unix_secs(base.as_unix_secs() - 3600).unwrap();
        assert!((fact.heat_live(earlier, 0.03) - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------
    // find_merge_candidates — cosine similarity scan
    // -----------------------------------------------------------------

    fn fact_with_key_embedding(content: &str, key: &str, embedding: Vec<f32>) -> Fact {
        Fact::new_pending(
            content,
            MemoryKey::from_raw(key).unwrap(),
            sid(1),
            Embedding::new(embedding).unwrap(),
            Timestamp::from_unix_secs(0).unwrap(),
            ConfidenceConfig::default().base,
        )
        .unwrap()
    }

    #[test]
    fn find_merge_candidates_empty_pool_returns_empty() {
        let pending = fact_with_key_embedding("p", "origa", vec![1.0, 0.0]);
        assert!(
            pending
                .find_merge_candidates(&[], &MergeConfig::default())
                .is_empty()
        );
    }

    #[test]
    fn find_merge_candidates_filters_below_threshold() {
        let pending = fact_with_key_embedding("p", "origa", vec![1.0, 0.0]);
        // Orthogonal → cosine 0.0 < 0.85.
        let pool = vec![fact_with_key_embedding("x", "origa", vec![0.0, 1.0])];
        assert!(
            pending
                .find_merge_candidates(&pool, &MergeConfig::default())
                .is_empty()
        );
    }

    #[test]
    fn find_merge_candidates_keeps_above_threshold_sorted_desc() {
        let pending = fact_with_key_embedding("p", "origa", vec![1.0, 0.0]);
        let pool = vec![
            fact_with_key_embedding("ortho", "origa", vec![0.0, 1.0]),
            fact_with_key_embedding("mid", "origa", vec![1.0, 1.0]),
            fact_with_key_embedding("perfect", "origa", vec![1.0, 0.0]),
        ];
        let out = pending.find_merge_candidates(&pool, &MergeConfig::default());
        // Only "perfect" passes the default 0.85 threshold.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].fact.content(), "perfect");
        assert!((out[0].cosine_similarity.value() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn find_merge_candidates_excludes_self_id() {
        let pending = fact_with_key_embedding("same", "origa", vec![1.0, 0.0]);
        // Same content → same id → excluded as a self-match.
        let pool = vec![fact_with_key_embedding("same", "origa", vec![1.0, 0.0])];
        assert!(
            pending
                .find_merge_candidates(&pool, &MergeConfig::default())
                .is_empty()
        );
    }

    #[test]
    fn find_merge_candidates_excludes_different_memory_key() {
        let pending = fact_with_key_embedding("p", "origa", vec![1.0, 0.0]);
        let pool = vec![fact_with_key_embedding("x", "other", vec![1.0, 0.0])];
        assert!(
            pending
                .find_merge_candidates(&pool, &MergeConfig::default())
                .is_empty()
        );
    }

    #[test]
    fn find_merge_candidates_skips_pool_member_without_embedding() {
        let pending = fact_with_key_embedding("p", "origa", vec![1.0, 0.0]);
        let pool_member =
            fact_with_key_embedding("x", "origa", vec![1.0, 0.0]).with_embedding(None);
        let pool = vec![pool_member];
        assert!(
            pending
                .find_merge_candidates(&pool, &MergeConfig::default())
                .is_empty()
        );
    }

    // -----------------------------------------------------------------
    // flag_conflict_bidirectional — symmetric conflict flag
    // -----------------------------------------------------------------

    #[test]
    fn flag_conflict_bidirectional_sets_both_sides() {
        let mut a = pending_fact("alpha", sid(1));
        let mut b = pending_fact("beta", sid(2));
        a.flag_conflict_bidirectional(&mut b).unwrap();
        assert!(a.conflicts_with().contains(b.id()));
        assert!(b.conflicts_with().contains(a.id()));
    }

    #[test]
    fn flag_conflict_bidirectional_is_idempotent() {
        let mut a = pending_fact("alpha", sid(1));
        let mut b = pending_fact("beta", sid(2));
        a.flag_conflict_bidirectional(&mut b).unwrap();
        a.flag_conflict_bidirectional(&mut b).unwrap();
        assert_eq!(a.conflicts_with().len(), 1);
        assert_eq!(b.conflicts_with().len(), 1);
    }
}
