//! `TokioDelay` — production [`Delay`](smos_application::ports::Delay) impl.
//!
//! A thin wrapper over `tokio::time::sleep`. Lives in the adapter layer so the
//! application core stays runtime-agnostic; the extraction retry loop decides
//! *when* to wait, this impl decides *how*.

use std::time::Duration;

use smos_application::ports::Delay;

/// `Delay` backed by the tokio runtime timer.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokioDelay;

impl Delay for TokioDelay {
    async fn delay(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }
}
