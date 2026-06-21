//! `Confidence` — f32 clamped to `[0.0, 1.0]`.

use crate::config::ConfidenceConfig;
use crate::enums::FactStatus;
use crate::error::DomainError;
use serde::{Deserialize, Serialize};

/// Confidence score produced by [`crate::entities::Fact::compute_confidence`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Confidence(f32);

impl Confidence {
    /// Validate-and-wrap. Rejects NaN, infinities, and out-of-range values.
    pub fn new(v: f32) -> Result<Self, DomainError> {
        if v.is_nan() || !(0.0..=1.0).contains(&v) {
            return Err(DomainError::ConfidenceOutOfRange(v));
        }
        Ok(Self(v))
    }

    /// Wrap a value that is *already known* to be in range (e.g. right after a
    /// clamp). Used by pure scorers that compute then clamp internally.
    pub(crate) fn new_unchecked(v: f32) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn value(self) -> f32 {
        self.0
    }

    /// Classify into a lifecycle status based on the configured thresholds.
    ///
    /// - `>= accept_threshold`    → `Accepted`
    /// - `>= pending_threshold`   → `Pending`
    /// - otherwise                → `Rejected`
    pub fn classify(self, cfg: &ConfidenceConfig) -> FactStatus {
        if self.value() >= cfg.accept_threshold {
            FactStatus::Accepted
        } else if self.value() >= cfg.pending_threshold {
            FactStatus::Pending
        } else {
            FactStatus::Rejected
        }
    }
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DomainError;

    #[test]
    fn new_accepts_zero() {
        assert_eq!(Confidence::new(0.0).unwrap().value(), 0.0);
    }

    #[test]
    fn new_accepts_one() {
        assert_eq!(Confidence::new(1.0).unwrap().value(), 1.0);
    }

    #[test]
    fn new_accepts_half() {
        assert_eq!(Confidence::new(0.5).unwrap().value(), 0.5);
    }

    #[test]
    fn new_rejects_negative() {
        assert!(matches!(
            Confidence::new(-0.1),
            Err(DomainError::ConfidenceOutOfRange(_))
        ));
    }

    #[test]
    fn new_rejects_above_one() {
        assert!(matches!(
            Confidence::new(1.1),
            Err(DomainError::ConfidenceOutOfRange(_))
        ));
    }

    #[test]
    fn new_rejects_nan() {
        assert!(Confidence::new(f32::NAN).is_err());
    }

    #[test]
    fn new_rejects_infinity() {
        assert!(Confidence::new(f32::INFINITY).is_err());
        assert!(Confidence::new(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn new_unchecked_clamps_overflow() {
        assert_eq!(Confidence::new_unchecked(2.0).value(), 1.0);
        assert_eq!(Confidence::new_unchecked(-1.0).value(), 0.0);
        assert_eq!(Confidence::new_unchecked(0.3).value(), 0.3);
    }

    #[test]
    fn partial_ord_orders_correctly() {
        let a = Confidence::new(0.3).unwrap();
        let b = Confidence::new(0.7).unwrap();
        assert!(a < b);
        assert!(b > a);
    }

    #[test]
    fn serde_roundtrip_preserves_value() {
        let c = Confidence::new(0.42).unwrap();
        let json = serde_json::to_string(&c).unwrap();
        let back: Confidence = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn classify_above_accept_threshold_is_accepted() {
        let cfg = ConfidenceConfig::default();
        assert_eq!(
            Confidence::new(0.7).unwrap().classify(&cfg),
            FactStatus::Accepted
        );
        assert_eq!(
            Confidence::new(1.0).unwrap().classify(&cfg),
            FactStatus::Accepted
        );
    }

    #[test]
    fn classify_between_pending_and_accept_is_pending() {
        let cfg = ConfidenceConfig::default();
        assert_eq!(
            Confidence::new(0.4).unwrap().classify(&cfg),
            FactStatus::Pending
        );
        assert_eq!(
            Confidence::new(0.5).unwrap().classify(&cfg),
            FactStatus::Pending
        );
        assert_eq!(
            Confidence::new(0.69).unwrap().classify(&cfg),
            FactStatus::Pending
        );
    }

    #[test]
    fn classify_below_pending_threshold_is_rejected() {
        let cfg = ConfidenceConfig::default();
        assert_eq!(
            Confidence::new(0.0).unwrap().classify(&cfg),
            FactStatus::Rejected
        );
        assert_eq!(
            Confidence::new(0.39).unwrap().classify(&cfg),
            FactStatus::Rejected
        );
    }
}
