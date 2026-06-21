//! Concrete runtime adapters: timers, task supervisors, background watchers.
//!
//! Slice-5 added [`TokioDelay`] (the production `Delay` impl) and an extraction
//! task supervisor ([`ExtractionSupervisor`]) built on an in-flight counter +
//! `Notify`. The supervisor lets the HTTP server drain in-flight extraction
//! tasks for `shutdown_extraction_grace_seconds` before the runtime drops, so
//! a Ctrl+C does not silently cancel half-finished fact extraction
//! (§12 durability).
//!
//! Slice-7 adds [`SessionWatcher`]: the background sweeper that retires
//! expired / overflowed sessions between requests so the operator does not
//! have to press `smos finalize` for every conversation. The watcher is the
//! graceful-degradation counterpart of the extraction supervisor — it owns
//! the only path that flushes the pending backlog to `Accepted` without an
//! explicit CLI trigger.

pub mod delay;
pub mod extraction_supervisor;
pub mod session_watcher;

pub use delay::TokioDelay;
pub use extraction_supervisor::ExtractionSupervisor;
pub use session_watcher::SessionWatcher;
