//! `Delay` port — injectable async timer (§12 retry backoff).
//!
//! Production uses [`crate`]'s adapter `TokioDelay` (a thin `tokio::time::sleep`
//! wrapper). Tests inject a no-op delay so the retry loop runs instantaneously
//! — only the retry *logic* is asserted, never wall-clock timing.
//!
//! Keeping this as a port lets the application layer orchestrate retry
//! backoff WITHOUT pulling a tokio runtime into the IO-free core: the use
//! case decides *when* and *how long* to wait, the adapter decides *how* the
//! wait is implemented.

use std::time::Duration;

/// Async timer boundary.
pub trait Delay {
    /// Wait for `duration` before resolving.
    fn delay(&self, duration: Duration) -> impl std::future::Future<Output = ()> + Send;
}
