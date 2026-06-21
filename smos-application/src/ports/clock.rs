//! `Clock` port — injectable wall-clock for deterministic time in tests.
//!
//! Production code uses `SystemClock` from `smos-adapters`; tests inject a
//! fake to advance time deterministically. `now` is intentionally synchronous
//! — fetching UTC time is a pure read with no async IO.

use smos_domain::Timestamp;

/// Wall-clock boundary.
pub trait Clock {
    /// Return the current UTC instant.
    fn now(&self) -> Timestamp;
}
