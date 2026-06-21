//! Persistence-layer errors.
//!
//! Returned by `FactRepository` / `SessionRepository` implementations. Each
//! variant maps to a distinct recovery strategy: serialization errors are
//! caller bugs, transaction conflicts are retryable, connect failures are
//! infrastructure.

use thiserror::Error;

/// Errors returned by storage adapters (currently the SurrealDB adapter).
#[derive(Debug, Error)]
pub enum RepoError {
    #[error("surrealdb connect failed: {0}")]
    ConnectFailed(String),

    #[error("surrealdb query failed: {0}")]
    QueryFailed(String),

    #[error("surrealdb serialization failed: {0}")]
    SerializationFailed(String),

    #[error("surrealdb transaction conflict")]
    TransactionConflict,

    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },

    #[error("record not found: {0}")]
    NotFound(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_inner_message() {
        let e = RepoError::QueryFailed("syntax error near 'FOO'".into());
        assert!(e.to_string().contains("syntax error near 'FOO'"));
    }

    #[test]
    fn transaction_conflict_display_is_stable() {
        assert_eq!(
            RepoError::TransactionConflict.to_string(),
            "surrealdb transaction conflict"
        );
    }

    #[test]
    fn vector_dimension_mismatch_display_shows_both_sides() {
        let e = RepoError::VectorDimensionMismatch {
            expected: 1024,
            actual: 768,
        };
        let msg = e.to_string();
        assert!(msg.contains("1024") && msg.contains("768"));
    }
}
