//! `Heat` — f32 clamped to `[0.0, 1.0]`, retrieval recency/decay base.

use crate::error::DomainError;
use crate::value_objects::Timestamp;
use serde::{Deserialize, Serialize};

/// Heat bookkeeping value stored on each fact.
///
/// `1.0` means "just accessed / maximally warm"; values decay over time via
/// [`Heat::decay`] / [`crate::entities::Fact::heat_live`]. Bound to `[0.0, 1.0]`
/// so the live decay formula cannot drift outside the unit interval.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Heat(f32);

impl Heat {
    /// Maximum heat (`1.0`) — the post-retrieval re-warming target used by
    /// the enrichment pipeline (`§7`). Exposed as a `const` so callers avoid
    /// the runtime `Heat::new(1.0).expect(...)` pattern in hot paths.
    pub const MAX: Heat = Heat(1.0);

    pub fn new(v: f32) -> Result<Self, DomainError> {
        if v.is_nan() || !(0.0..=1.0).contains(&v) {
            return Err(DomainError::HeatOutOfRange(v));
        }
        Ok(Self(v))
    }

    pub fn value(self) -> f32 {
        self.0
    }

    /// Stateless heat decay formula (§7): `heat_base * exp(-decay_rate * hours)`.
    ///
    /// Single source of truth for the heat decay curve. Used by both the domain
    /// ([`crate::entities::Fact::heat_live`]) and the application-layer
    /// retrieval projection (`RetrievalHit::heat_live`) so a future change to
    /// the formula propagates from one place.
    ///
    /// Past access is the normal case (positive hours); future timestamps
    /// (clock skew) clamp to zero so we never amplify heat above `heat_base`.
    pub fn decay(
        heat_base: Heat,
        last_access_at: Timestamp,
        now: Timestamp,
        decay_rate: f32,
    ) -> f32 {
        let delta = now.as_offset_date_time() - last_access_at.as_offset_date_time();
        let hours = (delta.as_seconds_f64() / 3600.0).max(0.0);
        let decay = (-(decay_rate as f64) * hours).exp();
        (heat_base.value() as f64 * decay) as f32
    }
}

impl std::fmt::Display for Heat {
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
        assert_eq!(Heat::new(0.0).unwrap().value(), 0.0);
    }

    #[test]
    fn new_accepts_one() {
        assert_eq!(Heat::new(1.0).unwrap().value(), 1.0);
    }

    #[test]
    fn new_accepts_half() {
        assert_eq!(Heat::new(0.5).unwrap().value(), 0.5);
    }

    #[test]
    fn new_rejects_negative() {
        assert!(matches!(
            Heat::new(-0.1),
            Err(DomainError::HeatOutOfRange(_))
        ));
    }

    #[test]
    fn new_rejects_above_one() {
        assert!(matches!(
            Heat::new(1.1),
            Err(DomainError::HeatOutOfRange(_))
        ));
    }

    #[test]
    fn new_rejects_nan() {
        assert!(Heat::new(f32::NAN).is_err());
    }

    #[test]
    fn serde_roundtrip_preserves_value() {
        let h = Heat::new(0.42).unwrap();
        let json = serde_json::to_string(&h).unwrap();
        let back: Heat = serde_json::from_str(&json).unwrap();
        assert_eq!(h, back);
    }

    #[test]
    fn decay_fresh_access_yields_full_heat() {
        let now = Timestamp::from_unix_secs(1_700_000_000).unwrap();
        let h = Heat::decay(Heat::MAX, now, now, 0.03);
        assert!((h - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decay_decays_after_24_hours_at_known_rate() {
        // exp(-0.03 * 24) ≈ 0.4868
        let base = Timestamp::from_unix_secs(1_700_000_000).unwrap();
        let one_day_later = Timestamp::from_unix_secs(base.as_unix_secs() + 24 * 3600).unwrap();
        let h = Heat::decay(Heat::MAX, base, one_day_later, 0.03);
        assert!((h - 0.4868).abs() < 1e-3, "got {h}");
    }

    #[test]
    fn decay_future_access_clamps_to_zero_decay() {
        let base = Timestamp::from_unix_secs(1_700_001_000).unwrap();
        let earlier = Timestamp::from_unix_secs(1_700_000_000).unwrap();
        let h = Heat::decay(Heat::MAX, base, earlier, 0.03);
        assert!((h - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decay_rate_zero_keeps_full_heat() {
        let base = Timestamp::from_unix_secs(0).unwrap();
        let later = Timestamp::from_unix_secs(10_000_000).unwrap();
        let h = Heat::decay(Heat::new(0.5).unwrap(), base, later, 0.0);
        assert!((h - 0.5).abs() < 1e-6);
    }
}
