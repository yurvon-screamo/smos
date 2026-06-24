//! `smos` binary presence + version check.
//!
//! The doctor is part of the `smos` crate, so the canonical location of the
//! binary under inspection is the running executable itself — probed via
//! [`std::env::current_exe`]. This works for development builds (cargo run
//! from the workspace), `cargo binstall` drops, and any operator-installed
//! binary without the doctor having to know where the operator placed it.

use super::super::types::CheckResult;

/// Probe the running executable and report its version + absolute path.
/// Always returns a row so the report lists the binary check even when
/// `current_exe` cannot resolve the path (a rare edge case under some
/// container runtimes).
pub async fn check_binaries() -> Vec<CheckResult> {
    let exe = std::env::current_exe()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "unknown".to_string());
    let version = env!("CARGO_PKG_VERSION");

    vec![CheckResult::pass(
        "smos binary",
        format!("version: {version}, path: {exe}"),
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn check_binaries_returns_single_pass_row_with_version() {
        let rows = check_binaries().await;
        assert_eq!(rows.len(), 1, "exactly one binary check row expected");
        let row = &rows[0];
        assert_eq!(row.name, "smos binary");
        assert!(row.status.is_pass(), "binary check must pass");
        assert!(
            row.details.contains(env!("CARGO_PKG_VERSION")),
            "details must include the compiled-in version: {}",
            row.details
        );
    }

    #[tokio::test]
    async fn check_binaries_never_returns_fail_even_when_path_unknown() {
        // The function must not return a FAIL row — `current_exe` failing is
        // a recoverable "I cannot tell you where I live" case, not a binary
        // absence. The version is still meaningful even without a path.
        let rows = check_binaries().await;
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].status.is_fail());
    }
}
