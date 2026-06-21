//! `SourceSessions` — provenance list of unique sessions that observed a fact.

use crate::value_objects::SessionId;
use serde::{Deserialize, Serialize};

/// Multi-value provenance of a fact.
///
/// Drives the `multi_source_bonus` in [`crate::entities::Fact::compute_confidence`]:
/// once 2+ distinct sessions have observed the same fact, its confidence is
/// bumped. Order is preserved (insertion order) so the *first* observation is
/// stable across merge unions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceSessions(Vec<SessionId>);

impl SourceSessions {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn from_one(id: SessionId) -> Self {
        Self(vec![id])
    }

    /// Wrap a vec, deduplicating in place while preserving first-seen order.
    pub fn from_vec(ids: Vec<SessionId>) -> Self {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if !out.contains(&id) {
                out.push(id);
            }
        }
        Self(out)
    }

    /// Add `id` if it is not already present.
    ///
    /// Returns `true` when the id was new (provenance grew) — callers use that
    /// signal to decide whether to recompute confidence.
    pub fn add_unique(&mut self, id: SessionId) -> bool {
        if self.0.contains(&id) {
            return false;
        }
        self.0.push(id);
        true
    }

    /// Union another set into this one. Returns `true` when any new id was added.
    pub fn union(&mut self, other: &SourceSessions) -> bool {
        let mut grew = false;
        for id in &other.0 {
            if self.add_unique(id.clone()) {
                grew = true;
            }
        }
        grew
    }

    pub fn distinct_count(&self) -> usize {
        self.0.len()
    }

    pub fn contains(&self, id: &SessionId) -> bool {
        self.0.contains(id)
    }

    pub fn as_slice(&self) -> &[SessionId] {
        &self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, SessionId> {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(n: u8) -> SessionId {
        // Deterministic session id for tests, bypassing the RNG.
        SessionId::from_raw(&format!("sess_{:012x}", n as u64)).expect("valid test session id")
    }

    #[test]
    fn add_unique_returns_true_first_time() {
        let mut s = SourceSessions::new();
        let id = sid(1);
        assert!(s.add_unique(id.clone()));
    }

    #[test]
    fn add_unique_returns_false_second_time() {
        let mut s = SourceSessions::from_one(sid(1));
        assert!(!s.add_unique(sid(1)));
    }

    #[test]
    fn distinct_count_after_adds() {
        let mut s = SourceSessions::new();
        s.add_unique(sid(1));
        s.add_unique(sid(2));
        s.add_unique(sid(1));
        assert_eq!(s.distinct_count(), 2);
    }

    #[test]
    fn union_dedups() {
        let mut a = SourceSessions::from_one(sid(1));
        a.add_unique(sid(2));
        let mut b = SourceSessions::from_one(sid(2));
        b.add_unique(sid(3));
        assert!(a.union(&b));
        assert_eq!(a.distinct_count(), 3);
    }

    #[test]
    fn union_returns_false_when_nothing_new() {
        let mut a = SourceSessions::from_one(sid(1));
        let b = SourceSessions::from_one(sid(1));
        assert!(!a.union(&b));
    }

    #[test]
    fn contains_works() {
        let id = sid(7);
        let s = SourceSessions::from_one(id.clone());
        assert!(s.contains(&id));
        assert!(!s.contains(&sid(8)));
    }

    #[test]
    fn from_vec_dedups_preserving_order() {
        let a = sid(1);
        let b = sid(2);
        let s = SourceSessions::from_vec(vec![a.clone(), b.clone(), a.clone(), b.clone()]);
        assert_eq!(s.as_slice(), [a, b]);
    }

    #[test]
    fn empty_source_has_zero_count() {
        assert_eq!(SourceSessions::new().distinct_count(), 0);
    }
}
