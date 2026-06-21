//! `SessionId` — proxy-issued, marker-encoded session identifier.
//!
//! Format: `sess_<12 lowercase hex>`. The marker (`<!-- smos:sess_xxx -->`)
//! carries the id through the conversation; SMOS re-reads it on each request
//! (§3 step 2). 12 hex chars = 48 bits of entropy, plenty to avoid collisions
//! across concurrent sessions.

use crate::error::DomainError;
use serde::{Deserialize, Serialize};
use std::cell::Cell;

thread_local! {
    /// Thread-local simple counter-based RNG seeded from system entropy.
    /// Good enough for an opaque session id; we are not modelling an adversary
    /// that can predict session ids (the marker is in-band by design).
    static RNG_STATE: Cell<u64> = const { Cell::new(0) };
}

/// Identifier of an active SMOS session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    /// Generate a fresh `sess_<12 hex>` id.
    ///
    /// `pub(crate)` on purpose: id generation reads system entropy, which
    /// is an IO concern. Production code routes through the `IdGenerator`
    /// port in `smos-application` (with `SystemIdGenerator` as the
    /// adapter impl); only domain-internal tests call this constructor
    /// directly so they can mint an id without threading a port through
    /// every fixture.
    #[allow(dead_code)] // only called from in-crate tests
    pub(crate) fn new() -> Self {
        let value = next_random_u64();
        let hex = format!("{:012x}", value & 0xFFFF_FFFF_FFFF);
        Self(format!("sess_{hex}"))
    }

    /// Parse from string. Pattern: `sess_[0-9a-f]{12}`.
    pub fn from_raw(s: &str) -> Result<Self, DomainError> {
        if is_valid_session_id(s) {
            Ok(Self(s.to_string()))
        } else {
            Err(DomainError::InvalidSessionId(s.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Render the trailing marker SMOS appends to assistant responses.
    ///
    /// Format: `\n<!-- smos:{session_id} -->`. The leading newline separates it
    /// from any inline content; the marker round-trips the session id through
    /// the conversation history (§3 step 2, §4 step 2).
    pub fn to_marker(&self) -> String {
        format!("\n<!-- smos:{} -->", self.0)
    }
}

fn is_valid_session_id(s: &str) -> bool {
    let Some(hex) = s.strip_prefix("sess_") else {
        return false;
    };
    hex.len() == 12
        && hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

#[allow(dead_code)] // only exercised via SessionId::new() in tests
fn next_random_u64() -> u64 {
    RNG_STATE.with(|cell| {
        let mut state = cell.get();
        if state == 0 {
            // Seed from system time + stack address entropy. Two threads will
            // almost certainly diverge because of the address mix-in.
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(KNUTH_GOLDEN_GAMMA);
            let addr_salt = &cell as *const _ as u64;
            // Odd multiplier keeps the low bit set (xorshift needs non-zero).
            state = nanos ^ addr_salt.wrapping_mul(ADDR_SALT_MULTIPLIER);
            if state == 0 {
                state = KNUTH_GOLDEN_GAMMA;
            }
        }
        // xorshift64 — cheap, statistically fine for opaque ids.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        cell.set(state);
        state
    })
}

/// Knuth's golden-ratio constant (0x9E3779B97F4A7C15) — well-known odd
/// multiplier for hash mixing, used here as the RNG's non-zero fallback seed.
#[allow(dead_code)] // only referenced via SessionId::new() in tests
const KNUTH_GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Salt for mixing the stack-address entropy into the seed. Any odd constant
/// works; this one was picked arbitrarily and just needs to be non-trivial.
#[allow(dead_code)] // only referenced via SessionId::new() in tests
const ADDR_SALT_MULTIPLIER: u64 = 0x517C_C1B7_2722_0A95;

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DomainError;

    #[test]
    fn new_returns_well_formed_id() {
        let id = SessionId::new();
        let s = id.as_str();
        assert!(s.starts_with("sess_"));
        let hex = &s["sess_".len()..];
        assert_eq!(hex.len(), 12);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn new_generates_distinct_ids_in_a_loop() {
        let mut ids = std::collections::HashSet::new();
        for _ in 0..32 {
            ids.insert(SessionId::new().as_str().to_string());
        }
        // 48-bit ids: 32 distinct draws with high probability.
        assert!(ids.len() > 28, "got {} distinct ids", ids.len());
    }

    #[test]
    fn new_produces_canonical_shape() {
        let id = SessionId::new();
        assert!(id.as_str().starts_with("sess_"));
    }

    #[test]
    fn from_raw_accepts_well_formed_id() {
        let parsed = SessionId::from_raw("sess_abcdef012345").unwrap();
        assert_eq!(parsed.as_str(), "sess_abcdef012345");
    }

    #[test]
    fn from_raw_rejects_missing_prefix() {
        assert!(matches!(
            SessionId::from_raw("abcdef012345"),
            Err(DomainError::InvalidSessionId(_))
        ));
    }

    #[test]
    fn from_raw_rejects_wrong_length() {
        assert!(SessionId::from_raw("sess_abc").is_err());
        assert!(SessionId::from_raw("sess_abcdef0123456789").is_err());
    }

    #[test]
    fn from_raw_rejects_uppercase() {
        assert!(SessionId::from_raw("sess_ABCDEF012345").is_err());
    }

    #[test]
    fn from_raw_rejects_non_hex() {
        assert!(SessionId::from_raw("sess_zzzzzzzzzzzz").is_err());
    }

    #[test]
    fn display_returns_raw_string() {
        let id = SessionId::from_raw("sess_abcdef012345").unwrap();
        assert_eq!(id.to_string(), "sess_abcdef012345");
    }

    #[test]
    fn to_marker_renders_template_with_session_id() {
        let id = SessionId::from_raw("sess_abcdef012345").unwrap();
        assert_eq!(id.to_marker(), "\n<!-- smos:sess_abcdef012345 -->");
    }
}
