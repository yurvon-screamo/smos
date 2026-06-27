//! Service log tail entry for `smos service status`.
//!
//! A Windows service runs detached (Session 0, no console), so its
//! `tracing` output goes to a rolling file under the service account's
//! `<smos_home>/logs/`. When the service fails to start (e.g. SCM
//! `WIN32_EXIT_CODE 1066` with `SERVICE_EXIT_CODE 1`), the operator is
//! stuck: `sc query` shows the dead state but never WHY. This module
//! finds the most recent `smos-service.log*` file under each candidate
//! directory (LocalSystem profile by default, the operator's SMOS_HOME
//! if redirected) and prints its tail so the failure reason is visible
//! inline in `smos service status`. The actual file reader +
//! filesystem-mocking tests live in [`reader`].

use std::path::PathBuf;

use reader::{TailOutcome, read_latest_tail};

#[path = "windows_log_reader.rs"]
mod reader;

/// Number of trailing lines to surface in `status`. Enough to show the
/// fatal `error!` plus the init context that preceded it, without
/// drowning the terminal in a multi-MB daily log.
const STATUS_TAIL_LINES: usize = 40;

/// Print the tail of the most recent service log, or a remediation hint
/// when no log exists at any candidate path. LocalSystem is probed first
/// (the default service account), then the operator's SMOS_HOME (covers
/// the env-file redirect written by `smos service install`).
pub(super) fn print_recent_service_log() {
    println!("Recent service log (last {STATUS_TAIL_LINES} lines):");
    for (label, dir) in candidate_log_dirs() {
        match read_latest_tail(&dir, STATUS_TAIL_LINES) {
            TailOutcome::Tail { path, lines } => {
                println!("  source: {} ({label})", path.display());
                println!("  -----");
                for line in &lines {
                    println!("  {line}");
                }
                println!("  -----");
                return;
            }
            TailOutcome::Missing => continue,
            TailOutcome::Unreadable { path, error } => {
                println!(
                    "  found {} ({label}) but could not read it: {error}",
                    path.display()
                );
                println!("    rerun `smos service status` from an elevated shell");
                return;
            }
        }
    }
    println!("  (no service log found at any candidate path)");
    println!("    LocalSystem default: <SystemRoot>\\System32\\config\\systemprofile\\.smos\\logs");
    println!("    or your SMOS_HOME (written next to the binary at install time,");
    println!("    see `smos service install` output for the env file path).");
}

/// Directories to probe, in priority order. The LocalSystem path resolves
/// `%SystemRoot%` (falling back to `C:\Windows`) so it stays correct on
/// non-`C:` Windows installs.
///
/// Known limitation: recency is NOT compared across candidate dirs — the
/// first dir with ANY matching log wins. If the operator redirected via
/// the env file (written next to the binary at install time) AFTER a
/// previous LocalSystem run, a stale LocalSystem log can mask the fresh
/// operator log. Acceptable for an interactive diagnostic; the printed
/// `source:` line lets the operator check the other path by hand.
fn candidate_log_dirs() -> Vec<(&'static str, PathBuf)> {
    vec![
        ("LocalSystem profile", localsystem_smos_logs()),
        ("operator SMOS_HOME", crate::paths::smos_home().join("logs")),
    ]
}

fn localsystem_smos_logs() -> PathBuf {
    let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
    PathBuf::from(system_root)
        .join("System32")
        .join("config")
        .join("systemprofile")
        .join(".smos")
        .join("logs")
}
