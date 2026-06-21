//! Helper functions for [`super`] — `sc.exe` invocation wrappers, path
//! validation, admin detection, and SCM output parsing. Kept in a sibling
//! file so the main `windows.rs` module stays under the 200-line size
//! limit while keeping all SCM-related concerns in one place.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use super::super::paths::ServicePaths;
use super::{DESCRIPTION, FAILURE_ACTIONS, FAILURE_RESET_SECONDS};

/// Run `sc.exe` with the given args and return its stdout on success.
pub(super) fn run_sc(args: &[&str]) -> Result<String> {
    let output = Command::new("sc")
        .args(args)
        .output()
        .with_context(|| format!("failed to spawn sc.exe with args {args:?}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("sc {:?} failed: {}", args, stderr.trim());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Parse the `STATE` line out of `sc.exe query <name>` output.
pub(super) fn extract_state(query_output: &str) -> String {
    query_output
        .lines()
        .find(|l| l.contains("STATE"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "UNKNOWN".to_string())
}

/// `sc.exe query <name>` succeeds iff the service is registered with SCM.
pub(super) fn service_exists(name: &str) -> Result<bool> {
    let output = Command::new("sc")
        .args(["query", name])
        .output()
        .context("failed to spawn sc.exe")?;
    Ok(output.status.success())
}

/// Reject paths that would corrupt the SCM `binPath=` argument string:
/// a `"` breaks quoting, and a trailing `\` escapes the closing quote.
pub(super) fn validate_windows_path(path: &Path) -> Result<()> {
    let s = path.to_string_lossy();
    if s.contains('"') {
        bail!("path contains a quote character which breaks Windows service binPath: {s}");
    }
    if s.ends_with('\\') {
        bail!("path ends with a backslash which escapes the closing quote in binPath: {s}");
    }
    Ok(())
}

/// Build the SCM `binPath=` argument string for `smos serve --config <cfg>`.
///
/// Both the binary and the config path are quoted so paths containing
/// spaces (e.g. `C:\Program Files\smos\smos.exe`) survive SCM's
/// `CommandLineToArgvW` argv splitting. Backslashes inside each quoted
/// segment are doubled to prevent the closing quote from being escaped.
/// Paths containing `"` or ending in `\` are rejected by
/// [`validate_windows_path`] before formatting.
pub(super) fn format_bin_path(binary: &Path, config: &Path) -> Result<String> {
    validate_windows_path(binary)?;
    validate_windows_path(config)?;
    let escaped_binary = binary.to_string_lossy().replace('\\', "\\\\");
    let escaped_config = config.to_string_lossy().replace('\\', "\\\\");
    Ok(format!(
        "\"{escaped_binary}\" serve --config \"{escaped_config}\""
    ))
}

/// Detect admin rights via `whoami /groups` and the High Mandatory Level
/// SID (S-1-16-12288). The SID is locale-independent, unlike the textual
/// "High Mandatory Level" label which only appears on English Windows.
/// This avoids the `net session` heuristic (which depends on the
/// LanmanServer service being running) and stays clear of the `windows`
/// crate's `unsafe` token APIs.
pub(super) fn is_admin() -> Result<bool> {
    let output = Command::new("whoami")
        .args(["/groups"])
        .output()
        .context("failed to spawn whoami")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.contains("S-1-16-12288"))
}

/// `sc description` is best-effort — log a warning on failure rather than
/// aborting the install, since the service itself is already created.
pub(super) fn set_description(paths: &ServicePaths) {
    if let Err(e) = run_sc(&["description", &paths.service_name, DESCRIPTION]) {
        tracing::warn!("failed to set service description: {e}");
    }
}

/// `sc failure` configures restart backoff — best-effort, log on failure.
pub(super) fn set_failure_recovery(paths: &ServicePaths) {
    let reset = FAILURE_RESET_SECONDS.to_string();
    if let Err(e) = run_sc(&[
        "failure",
        &paths.service_name,
        "reset=",
        &reset,
        "actions=",
        FAILURE_ACTIONS,
    ]) {
        tracing::warn!("failed to configure failure recovery: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn format_bin_path_quotes_both_segments_so_spaces_survive() {
        // Regression: an unquoted binary path broke CreateProcess when
        // SMOS was installed under `C:\Program Files\` — SCM split the
        // binPath at the first space and tried to exec a non-existent
        // `C:\Program` binary (CreateProcess error 193).
        let binary = PathBuf::from("C:\\Program Files\\smos\\smos.exe");
        let config = PathBuf::from("C:\\Program Files\\smos\\smos.toml");
        let bin_path = format_bin_path(&binary, &config).expect("format_bin_path");
        assert!(
            bin_path.starts_with("\"C:\\\\Program Files\\\\smos\\\\smos.exe\""),
            "binary segment must be quoted so SCM does not split at the space: {bin_path}"
        );
        assert!(
            bin_path.contains("\"C:\\\\Program Files\\\\smos\\\\smos.toml\""),
            "config segment must be quoted so the value survives argv splitting: {bin_path}"
        );
        assert!(bin_path.contains(" serve --config "));
    }

    #[test]
    fn format_bin_path_yields_scm_parsable_value() {
        // End-to-end shape of the value handed to `sc create binPath=`.
        // `std::process::Command` further escapes it when forwarding to
        // sc.exe; here we just pin that format_bin_path produces the
        // canonical `"binary" serve --config "config"` form with no
        // extra wrapping quotes that would confuse SCM at launch time.
        let binary = PathBuf::from("C:\\smos\\smos.exe");
        let config = PathBuf::from("C:\\smos\\smos.toml");
        let bin_path = format_bin_path(&binary, &config).expect("format_bin_path");
        assert_eq!(
            bin_path,
            "\"C:\\\\smos\\\\smos.exe\" serve --config \"C:\\\\smos\\\\smos.toml\""
        );
        // No spurious outer wrapping.
        assert!(!bin_path.starts_with("\"\""), "double-wrapping breaks SCM");
        assert!(!bin_path.ends_with("\"\""), "double-wrapping breaks SCM");
    }

    #[test]
    fn format_bin_path_rejects_path_with_embedded_quote() {
        let bad = PathBuf::from("C:\\smos\"evil.exe");
        let ok = PathBuf::from("C:\\smos\\smos.toml");
        assert!(format_bin_path(&bad, &ok).is_err());
        assert!(format_bin_path(&ok, &bad).is_err());
    }

    #[test]
    fn format_bin_path_rejects_trailing_backslash() {
        // A trailing `\` would escape the closing quote SCM wraps around
        // binPath, turning it into a literal `\"` that breaks argv parsing.
        let bad = PathBuf::from("C:\\smos\\");
        let ok = PathBuf::from("C:\\smos\\smos.toml");
        assert!(format_bin_path(&bad, &ok).is_err());
        assert!(format_bin_path(&ok, &bad).is_err());
    }
}
