//! `Cosine` — f32 clamped to `[-1.0, 1.0]`.

use crate::error::DomainError;
use serde::{Deserialize, Serialize};

/// Cosine similarity score between two fact embeddings.
///
/// Stored as a value object because the domain layer reasons about thresholds
/// (`cosine >= 0.85` for merge candidates) and the cosine formula can produce
/// tiny floating-point excursions past `[-1, 1]` for identical vectors.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Cosine(f32);

impl Cosine {
    pub fn new(v: f32) -> Result<Self, DomainError> {
        if v.is_nan() || !(-1.0..=1.0).contains(&v) {
            return Err(DomainError::CosineOutOfRange(v));
        }
        Ok(Self(v))
    }

    /// Build a [`Cosine`] from raw cosine-similarity math, snapping tiny
    /// floating-point overshoots back into the valid interval.
    pub(crate) fn from_raw_clamped(v: f32) -> Self {
        if v.is_nan() {
            return Self(0.0);
        }
        Self(v.clamp(-1.0, 1.0))
    }

    pub fn value(self) -> f32 {
        self.0
    }
}

impl std::fmt::Display for Cosine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DomainError;

    #[test]
    fn new_accepts_minus_one() {
        assert_eq!(Cosine::new(-1.0).unwrap().value(), -1.0);
    }

    #[test]
    fn new_accepts_plus_one() {
        assert_eq!(Cosine::new(1.0).unwrap().value(), 1.0);
    }

    #[test]
    fn new_accepts_zero() {
        assert_eq!(Cosine::new(0.0).unwrap().value(), 0.0);
    }

    #[test]
    fn new_rejects_below_minus_one() {
        assert!(matches!(
            Cosine::new(-1.1),
            Err(DomainError::CosineOutOfRange(_))
        ));
    }

    #[test]
    fn new_rejects_above_plus_one() {
        assert!(matches!(
            Cosine::new(1.1),
            Err(DomainError::CosineOutOfRange(_))
        ));
    }

    #[test]
    fn new_rejects_nan() {
        assert!(Cosine::new(f32::NAN).is_err());
    }

    #[test]
    fn from_raw_clamped_snaps_overflow() {
        assert_eq!(Cosine::from_raw_clamped(1.0001).value(), 1.0);
        assert_eq!(Cosine::from_raw_clamped(-1.0001).value(), -1.0);
    }

    #[test]
    fn from_raw_clamped_translates_nan_to_zero() {
        assert_eq!(Cosine::from_raw_clamped(f32::NAN).value(), 0.0);
    }
}
