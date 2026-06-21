//! Domain-level errors.
//!
//! Every operation that can fail at the domain boundary returns [`DomainError`].
//! No IO errors live here — the domain layer is pure.

use crate::enums::FactStatus;
use crate::value_objects::{FactId, Timestamp};

/// All invariants and parse failures the domain layer can report.
///
/// Variants are intentionally exhaustive: each one corresponds to a single
/// well-defined invariant of a value object or aggregate. Callers match
/// exhaustively so the compiler flags any new invariant we forget to handle.
#[derive(Debug, thiserror::Error)]
pub enum DomainError {
    #[error("fact id format invalid: {0}")]
    InvalidFactId(String),

    #[error("memory key unsafe or invalid: {0}")]
    UnsafeMemoryKey(String),

    #[error("session id format invalid: {0}")]
    InvalidSessionId(String),

    #[error("confidence out of range [0,1]: {0}")]
    ConfidenceOutOfRange(f32),

    #[error("heat out of range [0,1]: {0}")]
    HeatOutOfRange(f32),

    #[error("cosine out of range [-1,1]: {0}")]
    CosineOutOfRange(f32),

    #[error("fact content empty")]
    EmptyFactContent,

    #[error("embedding empty")]
    EmptyEmbedding,

    #[error("timestamp out of representable range: {0}")]
    InvalidTimestamp(String),

    #[error("illegal status transition: {from} -> {to}")]
    IllegalStatusTransition { from: FactStatus, to: FactStatus },

    #[error("invariant violation: Accepted fact must have confidence >= {threshold}, got {actual}")]
    ConfidenceBelowAcceptThreshold { threshold: f32, actual: f32 },

    #[error("invariant violation: valid_until ({until}) <= valid_from ({from})")]
    ValidUntilBeforeValidFrom { from: Timestamp, until: Timestamp },

    #[error("invariant violation: fact cannot conflict with itself: {0}")]
    SelfConflict(FactId),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enums::FactStatus;
    use crate::value_objects::{FactId, Timestamp};

    #[test]
    fn display_invalid_fact_id_contains_input() {
        let err = DomainError::InvalidFactId("bad".to_string());
        assert_eq!(err.to_string(), "fact id format invalid: bad");
    }

    #[test]
    fn display_unsafe_memory_key_contains_input() {
        let err = DomainError::UnsafeMemoryKey("../etc".to_string());
        assert_eq!(err.to_string(), "memory key unsafe or invalid: ../etc");
    }

    #[test]
    fn display_confidence_out_of_range_includes_value() {
        let err = DomainError::ConfidenceOutOfRange(1.5);
        assert_eq!(err.to_string(), "confidence out of range [0,1]: 1.5");
    }

    #[test]
    fn display_empty_fact_content_is_static_text() {
        let err = DomainError::EmptyFactContent;
        assert_eq!(err.to_string(), "fact content empty");
    }

    #[test]
    fn display_illegal_transition_includes_both_statuses() {
        let err = DomainError::IllegalStatusTransition {
            from: FactStatus::Accepted,
            to: FactStatus::Pending,
        };
        assert_eq!(
            err.to_string(),
            "illegal status transition: accepted -> pending"
        );
    }

    #[test]
    fn display_confidence_below_threshold_includes_threshold_and_actual() {
        let err = DomainError::ConfidenceBelowAcceptThreshold {
            threshold: 0.7,
            actual: 0.5,
        };
        assert_eq!(
            err.to_string(),
            "invariant violation: Accepted fact must have confidence >= 0.7, got 0.5"
        );
    }

    #[test]
    fn display_self_conflict_includes_fact_id() {
        let id = FactId::from_content("x");
        let err = DomainError::SelfConflict(id.clone());
        assert_eq!(
            err.to_string(),
            format!(
                "invariant violation: fact cannot conflict with itself: {}",
                id
            )
        );
    }

    #[test]
    fn display_valid_until_before_valid_from_mentions_invariant() {
        let from = Timestamp::from_unix_secs(1000).unwrap();
        let until = Timestamp::from_unix_secs(500).unwrap();
        let err = DomainError::ValidUntilBeforeValidFrom { from, until };
        let msg = err.to_string();
        // Display carries the invariant text and both timestamps (the exact
        // OffsetDateTime rendering is crate-version dependent, so we assert on
        // the stable substring rather than a specific datetime format).
        assert!(msg.contains("valid_until"), "msg = {msg}");
        assert!(msg.contains("valid_from"), "msg = {msg}");
        assert!(msg.contains("invariant violation"), "msg = {msg}");
    }
}
