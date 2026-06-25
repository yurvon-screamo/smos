//! Clock and delay test doubles.
//!
//! `FixedClock` always reports the same instant — used by every use-case unit
//! test so timestamps are deterministic. `NoOpDelay` collapses retry backoff to
//! zero so the suite never pays the real sleeps; timing is still verified
//! end-to-end against the real `TokioDelay` adapter.

use std::time::Duration;

use smos_domain::Timestamp;

use crate::ports::{Clock, Delay};

#[derive(Clone)]
pub struct FixedClock(pub Timestamp);

impl Clock for FixedClock {
    fn now(&self) -> Timestamp {
        self.0
    }
}

#[derive(Clone, Copy)]
pub struct NoOpDelay;

impl Delay for NoOpDelay {
    async fn delay(&self, _duration: Duration) {}
}
