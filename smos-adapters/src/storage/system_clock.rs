//! `SystemClock` — production wall-clock backed by the `time` crate.

use smos_application::ports::Clock;
use smos_domain::Timestamp;
use time::OffsetDateTime;

/// Wall-clock that reads the system UTC time. Inject a fake in tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        // Reads the system clock here (in the adapter layer) and hands the
        // domain a pure value — the domain crate itself never touches
        // wall-clock time, preserving its IO-free invariant.
        Timestamp::from_offset_date_time(OffsetDateTime::now_utc())
    }
}
