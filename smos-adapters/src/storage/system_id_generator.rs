//! `SystemIdGenerator` — production id generator backed by a thread-local
//! xorshift RNG seeded from system time + thread id.
//!
//! The domain owns the `SessionId` value type but stops short of generating
//! fresh ids — `SessionId::new()` is `pub(crate)` so production callers
//! route through this adapter (id generation is conceptually IO: it mixes
//! system time + thread identity). Tests inside the domain crate can still
//! call `SessionId::new()` directly without going through a port.

use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

use smos_application::ports::IdGenerator;
use smos_domain::SessionId;

/// Knuth's golden-ratio constant — well-known odd multiplier for hash
/// mixing, reused here as the RNG's non-zero fallback seed.
const KNUTH_GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Salt for mixing the thread id into the seed. Any odd constant works;
/// this one was picked arbitrarily and just needs to be non-trivial.
const THREAD_SALT_MULTIPLIER: u64 = 0x517C_C1B7_2722_0A95;

/// Id generator that delegates to a thread-local xorshift64 RNG.
///
/// Cheap to construct and `Copy`-able; inject a fake in tests so session
/// ids are predictable and assertions do not race on entropy. The
/// algorithm mirrors the one the domain used internally so the wire
/// format (`sess_<12 hex>`) stays stable across the move.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemIdGenerator;

impl IdGenerator for SystemIdGenerator {
    fn new_session_id(&self) -> SessionId {
        let value = next_random_u64();
        // The hex mask keeps the id inside the canonical 12-char shape
        // that `SessionId::from_raw` validates.
        let formatted = format!("sess_{:012x}", value & 0xFFFF_FFFF_FFFF);
        SessionId::from_raw(&formatted).expect("generated id matches the canonical pattern")
    }
}

thread_local! {
    /// Thread-local xorshift64 state, seeded lazily from system time +
    /// thread id. Each thread draws a non-overlapping sequence.
    static RNG_STATE: Cell<u64> = const { Cell::new(0) };
}

fn next_random_u64() -> u64 {
    RNG_STATE.with(|cell| {
        let mut state = cell.get();
        if state == 0 {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(KNUTH_GOLDEN_GAMMA);
            // Mix the thread id in so two concurrent threads diverge.
            let mut hasher = DefaultHasher::new();
            std::thread::current().id().hash(&mut hasher);
            let thread_bits = hasher.finish();
            state = nanos ^ thread_bits.wrapping_mul(THREAD_SALT_MULTIPLIER);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_id_is_well_formed() {
        let generator = SystemIdGenerator;
        let id = generator.new_session_id();
        let s = id.as_str();
        assert!(s.starts_with("sess_"));
        let hex = &s["sess_".len()..];
        assert_eq!(hex.len(), 12);
        assert!(
            hex.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn many_ids_are_mostly_distinct() {
        let generator = SystemIdGenerator;
        let mut seen = HashSet::new();
        for _ in 0..32 {
            seen.insert(generator.new_session_id().as_str().to_string());
        }
        // 48-bit ids: 32 distinct draws with very high probability.
        assert!(seen.len() > 28, "got {} distinct ids", seen.len());
    }
}
