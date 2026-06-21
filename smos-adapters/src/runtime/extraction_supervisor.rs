//! `ExtractionSupervisor` — tracks background extraction tasks so the HTTP
//! server can drain them for `shutdown_extraction_grace_seconds` before the
//! runtime drops.
//!
//! Without this, a Ctrl+C / SIGTERM mid-extraction silently cancels the
//! in-flight task: the response already reached the client (no replay), so
//! the half-extracted facts are lost forever — unacceptable for a memory
//! system whose purpose is durability. The supervisor is the minimal fix: an
//! in-flight counter incremented at spawn and decremented on completion, plus
//! a [`Notify`] that wakes the shutdown drain when the last task finishes.
//!
//! After the grace window elapses, remaining tasks are left to the runtime's
//! drop (which cancels them): this is the explicit graceful-degradation
//! boundary documented in `smos.toml`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::Notify;
use tokio::time::Instant;

/// Shared tracker for background extraction tasks.
///
/// Cheap to clone (one `Arc` bump); every clone shares the same counter +
/// notifier, so a task spawned from any clone is visible to the shutdown drain.
#[derive(Clone, Default)]
pub struct ExtractionSupervisor {
    in_flight: Arc<AtomicUsize>,
    done: Arc<Notify>,
}

impl ExtractionSupervisor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Spawn `task` and track it. The task is NOT detached from the
    /// supervisor — the in-flight counter is decremented on completion so
    /// [`drain`](Self::drain) can wait for it.
    ///
    /// The decrement runs inside an [`InFlightGuard`]'s `Drop`, which
    /// fires on normal completion, error propagation, AND panic-unwind —
    /// so a panicking extraction task can never leak its slot. Without
    /// this guard, a panic mid-task would leave the counter at +1 and
    /// the drain would wedge against the grace window forever (the
    /// `Notify` would never fire because `fetch_sub(1)` never ran).
    pub fn spawn<F>(&self, task: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let in_flight = Arc::clone(&self.in_flight);
        let done = Arc::clone(&self.done);
        tokio::spawn(async move {
            // Guard drops first on every exit path (panic, `?`, normal),
            // performing the `fetch_sub` + last-task `notify_one`.
            let _guard = InFlightGuard { in_flight, done };
            task.await;
        });
    }

    /// Wait for every tracked task to finish, or until `grace` elapses —
    /// whichever comes first. Tasks still running after the grace window are
    /// left to the runtime's drop (cancelled): the explicit degradation
    /// boundary.
    pub async fn drain(&self, grace: Duration) {
        let deadline = Instant::now() + grace;
        loop {
            if self.in_flight.load(Ordering::SeqCst) == 0 {
                return;
            }
            let now = Instant::now();
            if now >= deadline {
                tracing::warn!(
                    ?grace,
                    in_flight = self.in_flight.load(Ordering::SeqCst),
                    "extraction drain timed out; cancelling remaining in-flight tasks"
                );
                return;
            }
            // `notified()` consumes a permit if `notify_waiters` already fired;
            // otherwise it waits until the next completion (or the timeout).
            let _ = tokio::time::timeout_at(deadline, self.done.notified()).await;
        }
    }
}

/// RAII decrement for [`ExtractionSupervisor::in_flight`].
///
/// Created inside the spawned task before `task.await`; dropped on every
/// exit path (normal return, `?`-propagation, panic-unwind). Performs the
/// `fetch_sub(1)` + last-task `notify_one()` pair so the drain is woken
/// the moment the counter hits zero — even when the underlying task
/// panicked, which the previous inline `fetch_sub` would have skipped.
struct InFlightGuard {
    in_flight: Arc<AtomicUsize>,
    done: Arc<Notify>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        // `notify_one` stores a permit if the drain is between the
        // `in_flight.load()` check and `notified()`, so the wakeup is not
        // lost when the LAST task completes in that window. This must run
        // on panic too — a leaked slot would wedge the drain against the
        // grace window forever.
        if self.in_flight.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.done.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    #[tokio::test]
    async fn drain_returns_immediately_when_no_tasks_tracked() {
        let supervisor = ExtractionSupervisor::new();
        // No tasks spawned — drain must return instantly.
        let start = Instant::now();
        supervisor.drain(Duration::from_secs(5)).await;
        assert!(start.elapsed() < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn drain_waits_for_in_flight_tasks_to_complete() {
        let supervisor = ExtractionSupervisor::new();
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let c = counter.clone();
            supervisor.spawn(async move {
                tokio::time::sleep(Duration::from_millis(20)).await;
                c.fetch_add(1, AtomicOrdering::SeqCst);
            });
        }
        supervisor.drain(Duration::from_secs(2)).await;
        assert_eq!(
            counter.load(AtomicOrdering::SeqCst),
            3,
            "all tracked tasks completed before drain returned"
        );
    }

    #[tokio::test]
    async fn drain_returns_after_grace_when_tasks_outlive_window() {
        let supervisor = ExtractionSupervisor::new();
        // A task that sleeps longer than the grace window.
        supervisor.spawn(async move {
            tokio::time::sleep(Duration::from_secs(2)).await;
        });
        let start = Instant::now();
        supervisor.drain(Duration::from_millis(100)).await;
        // Drain must respect the grace window (not block 2 s for the task).
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "drain must return after the grace window, not wait for the slow task"
        );
    }

    /// A panicking extraction task MUST NOT wedge the drain. The
    /// `InFlightGuard`'s `Drop` runs on panic-unwind, performs the
    /// `fetch_sub`, and notifies the drain — otherwise a panic mid-task
    /// would leave the counter at +1 forever (the drain would wait the
    /// full grace window on every subsequent shutdown for a task that
    /// already died). This test pins that contract.
    #[tokio::test]
    async fn drain_survives_task_panic_and_returns_promptly() {
        let supervisor = ExtractionSupervisor::new();
        supervisor.spawn(async {
            panic!("simulated extraction panic");
        });
        // Yield once so the panicking task gets polled at least once
        // before the drain begins — without this the current-thread
        // runtime may schedule the panic AFTER the drain has already
        // started waiting on `notified()`, which is fine for production
        // but makes the timing-based assertion below flaky.
        tokio::task::yield_now().await;
        // A short grace — if the panic leaks the slot, this drain would
        // wait the full 500 ms instead of returning as soon as the
        // panicking task unwinds.
        let start = Instant::now();
        supervisor.drain(Duration::from_millis(500)).await;
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(400),
            "drain must NOT wait the full grace window when the only in-flight \
             task panicked — the Drop guard must have decremented the counter \
             (elapsed = {elapsed:?})"
        );
    }

    /// After a panicking task has unwound, the supervisor MUST accept
    /// fresh spawns and drain them normally. A leaked counter from the
    /// panic would make a later drain return early (counter never reaches
    /// zero) or wedge (counter stuck above zero) — both are
    /// unacceptable. This test pins the "panic then continue" contract.
    #[tokio::test]
    async fn supervisor_continues_working_after_a_panic() {
        let supervisor = ExtractionSupervisor::new();
        supervisor.spawn(async {
            panic!("first task panics");
        });
        // Yield once so the panicking task gets scheduled and unwinds
        // before the second spawn.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let counter = Arc::new(AtomicUsize::new(0));
        let tracked = counter.clone();
        supervisor.spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            tracked.fetch_add(1, AtomicOrdering::SeqCst);
        });
        supervisor.drain(Duration::from_secs(2)).await;
        assert_eq!(
            counter.load(AtomicOrdering::SeqCst),
            1,
            "post-panic task must complete and be observed by the drain"
        );
    }
}
