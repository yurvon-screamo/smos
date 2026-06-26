//! Windows service process entry point (public surface).
//!
//! When the SCM starts the service it re-launches `smos.exe` with
//! [`SERVICE_RUN_FLAG`] prepended to the `binPath` (see
//! `service::windows_helpers::format_bin_path`). The unified `smos`
//! binary detects that flag in `main()` BEFORE building the tokio runtime
//! and hands control to [`run_as_service`], which blocks the main thread
//! inside the SCM dispatcher for the lifetime of the process. The actual
//! service body (runtime build, status reporting, §12 drain) lives in
//! [`scm`] and runs on the SCM-spawned worker thread — `#[tokio::main]`
//! cannot coexist with the dispatcher, so the runtime is built inside
//! `ServiceMain`, not in `main`.

#![cfg(target_os = "windows")]

mod scm;
mod status;

pub use scm::{SERVICE_RUN_FLAG, run_as_service};
