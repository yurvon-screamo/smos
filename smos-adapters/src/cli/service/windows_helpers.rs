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
        bail!(
            "sc {:?} failed: {}",
            args,
            sc_failure_detail(&output.stdout, &output.stderr)
        );
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

/// Build the logical `binPath` value SCM stores for `smos --run-as-service`.
///
/// This is the string SCM persists and later hands verbatim to
/// `CreateProcessW`, so it must already be a valid `CommandLineToArgvW`
/// command line: the binary path is wrapped in quotes so argv splitting
/// at service start keeps the binary as a single token even when it
/// contains spaces. Backslashes are left as-is — they are only
/// significant when immediately preceding a `"`, and
/// [`validate_windows_path`] already rejects paths ending in `\` (which
/// would escape the closing quote).
///
/// `--run-as-service` is the hidden flag [`crate::cli::service_runner`]
/// matches in `main()` to route the SCM-launched process into the SCM
/// dispatcher instead of the clap CLI: SCM owns the main thread for the
/// process lifetime, so a tokio runtime created by `#[tokio::main]`
/// would collide with the runtime built inside `ServiceMain`.
///
/// Both `--config` and `--smos-home` are baked explicitly into binPath.
/// Unlike Linux/macOS user-services (where the service account's home
/// directory resolves naturally), a Windows LocalSystem service has no
/// operator home — `smos_home()` would resolve to
/// `C:\WINDOWS\System32\config\systemprofile\.smos\`, landing logs, DB,
/// and models in the system directory. Baking `--smos-home` at install
/// time (when the operator's profile is available) prevents this; baking
/// `--config` pins the config file path unambiguously.
///
/// This is NOT the form passed to `sc.exe` on the command line: that form
/// needs an extra layer of outer quoting plus inner-quote escaping handled
/// by [`quote_for_argv`].
pub(super) fn format_bin_path(binary: &Path, config: &Path, smos_home: &Path) -> Result<String> {
    validate_windows_path(binary)?;
    validate_windows_path(config)?;
    validate_windows_path(smos_home)?;
    let bin_str = binary.to_string_lossy();
    let config_str = config.to_string_lossy();
    let home_str = smos_home.to_string_lossy();
    Ok(format!(
        "\"{bin_str}\" --run-as-service --config \"{config_str}\" --smos-home \"{home_str}\""
    ))
}

/// Quote `s` as a single argv token using the canonical `CommandLineToArgvW`
/// algorithm: wrap in outer double quotes, double every backslash run that
/// is immediately followed by `"` (or that is trailing, i.e. immediately
/// before the closing quote we emit), and turn each `"` into `\"`.
///
/// The result round-trips through `CommandLineToArgvW` back to `s` for ANY
/// input — including embedded quotes and backslash-before-quote runs — so it
/// is safe to pass arbitrary path strings. It must be handed to
/// [`CommandExt::raw_arg`], NOT `Command::arg`: `arg` would re-wrap the
/// value in another quote layer and double-escape the inner `\"` sequences,
/// producing a token sc.exe cannot parse back.
pub(super) fn quote_for_argv(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    let mut backslashes = 0usize;
    for c in s.chars() {
        match c {
            '\\' => backslashes += 1,
            '"' => {
                // A `"` is preceded by an odd number of backslashes in the
                // encoded form (2n+1: n doubled backslashes + the escaped
                // quote). Flush the preceding run doubled, then `\"`.
                for _ in 0..(backslashes * 2 + 1) {
                    out.push('\\');
                }
                out.push('"');
                backslashes = 0;
            }
            other => {
                // Backslashes not adjacent to a `"` are inert — emit verbatim.
                for _ in 0..backslashes {
                    out.push('\\');
                }
                backslashes = 0;
                out.push(other);
            }
        }
    }
    // Trailing backslashes sit immediately before the closing quote we are
    // about to emit, so they must be doubled too.
    for _ in 0..(backslashes * 2) {
        out.push('\\');
    }
    out.push('"');
    out
}

/// Surface the real reason `sc.exe` failed.
///
/// `sc.exe` is unusual: it writes most failures (e.g.
/// `CreateService FAILED 1073:` / `The specified service already exists`)
/// to **stdout**, leaving stderr empty. Returning an empty error suffix
/// (as the previous `bail!("sc create failed: {}", stderr.trim())` did)
/// hides the SCM error code from the operator. Prefer stderr when present
/// (spawn-level errors go there), otherwise fall back to stdout.
pub(super) fn sc_failure_detail(stdout: &[u8], stderr: &[u8]) -> String {
    let stderr_s = String::from_utf8_lossy(stderr).trim().to_string();
    if !stderr_s.is_empty() {
        return stderr_s;
    }
    String::from_utf8_lossy(stdout).trim().to_string()
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

/// `sc failureflag <svc> 1` enables recovery actions for graceful stops
/// with a non-zero exit code, not only for process crashes. SMOS reports
/// `SERVICE_STOPPED` with `ServiceExitCode::ServiceSpecific(1)` on a
/// startup error (bad config, unreachable SurrealDB, ORT load failure).
/// Without this flag SCM treats that as a clean stop and never fires the
/// restart backoff configured by [`set_failure_recovery`], so the
/// operator's `failure actions=restart/...` is silently dead. See
/// SERVICE_CONFIG_FAILURE_ACTIONS_FLAG in the Win32 docs.
pub(super) fn set_failure_flag(paths: &ServicePaths) {
    if let Err(e) = run_sc(&["failureflag", &paths.service_name, "1"]) {
        tracing::warn!("failed to enable failureflag (restart-on-error): {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_config() -> PathBuf {
        PathBuf::from("C:\\Users\\me\\.smos\\config.toml")
    }

    fn sample_smos_home() -> PathBuf {
        PathBuf::from("C:\\Users\\me\\.smos")
    }

    #[test]
    fn format_bin_path_quotes_binary_so_spaces_survive() {
        // Regression: an unquoted binary path broke CreateProcess when
        // SMOS was installed under `C:\Program Files\` — SCM split the
        // binPath at the first space and tried to exec a non-existent
        // `C:\Program` binary (CreateProcess error 193).
        let binary = PathBuf::from("C:\\Program Files\\smos\\smos.exe");
        let bin_path = format_bin_path(&binary, &sample_config(), &sample_smos_home())
            .expect("format_bin_path");
        assert!(
            bin_path.starts_with("\"C:\\Program Files\\smos\\smos.exe\""),
            "binary segment must be quoted so SCM does not split at the space: {bin_path}"
        );
    }

    #[test]
    fn format_bin_path_bakes_config_and_smos_home() {
        let binary = PathBuf::from("C:\\smos\\smos.exe");
        let bin_path = format_bin_path(&binary, &sample_config(), &sample_smos_home())
            .expect("format_bin_path");
        assert!(
            bin_path.contains("--config \"C:\\Users\\me\\.smos\\config.toml\""),
            "binPath must bake --config: {bin_path}"
        );
        assert!(
            bin_path.contains("--smos-home \"C:\\Users\\me\\.smos\""),
            "binPath must bake --smos-home: {bin_path}"
        );
        assert!(
            bin_path.contains("--run-as-service"),
            "binPath must contain --run-as-service: {bin_path}"
        );
    }

    #[test]
    fn format_bin_path_rejects_binary_with_embedded_quote() {
        let bad_binary = PathBuf::from("C:\\smos\"evil.exe");
        assert!(format_bin_path(&bad_binary, &sample_config(), &sample_smos_home()).is_err());
    }

    #[test]
    fn format_bin_path_rejects_binary_with_trailing_backslash() {
        // A trailing `\` on the binary would escape the closing quote SCM
        // wraps around binPath, turning it into a literal `\"` that
        // breaks argv parsing.
        let bad_binary = PathBuf::from("C:\\smos\\");
        assert!(format_bin_path(&bad_binary, &sample_config(), &sample_smos_home()).is_err());
    }

    #[test]
    fn format_bin_path_rejects_config_with_trailing_backslash() {
        let binary = PathBuf::from("C:\\smos\\smos.exe");
        let bad_config = PathBuf::from("C:\\Users\\me\\.smos\\");
        assert!(format_bin_path(&binary, &bad_config, &sample_smos_home()).is_err());
    }

    #[test]
    fn format_bin_path_rejects_smos_home_with_trailing_backslash() {
        let binary = PathBuf::from("C:\\smos\\smos.exe");
        let bad_home = PathBuf::from("C:\\Users\\me\\");
        assert!(format_bin_path(&binary, &sample_config(), &bad_home).is_err());
    }

    #[test]
    fn quote_for_argv_wraps_and_escapes_inner_quotes() {
        // Regression: passing `format_bin_path`'s output through
        // `Command::arg` (instead of `raw_arg`) made std wrap the value
        // in an extra quote layer and double-escape every inner `"`,
        // yielding a token sc.exe could not parse — `sc create failed:`
        // with no further detail. `quote_for_argv` produces the canonical
        // single-argv form that `raw_arg` forwards verbatim.
        let bin_path = "\"C:\\Program Files\\smos\\smos.exe\" --run-as-service";
        let argv = quote_for_argv(bin_path);
        assert_eq!(
            argv,
            "\"\\\"C:\\Program Files\\smos\\smos.exe\\\" --run-as-service\""
        );
        // Round-trips through CommandLineToArgvW back to the original.
        assert_eq!(parse_argv(&argv), bin_path);
    }

    #[test]
    fn quote_for_argv_leaves_backslashes_alone() {
        // Backslashes not adjacent to `"` are inert under CommandLineToArgvW;
        // `quote_for_argv` must NOT double them (doubling would corrupt
        // `C:\smos\smos.exe` into `C:\\smos\\smos.exe` in the stored binPath).
        let argv = quote_for_argv("C:\\smos\\smos.exe");
        assert_eq!(argv, "\"C:\\smos\\smos.exe\"");
        assert_eq!(parse_argv(&argv), "C:\\smos\\smos.exe");
    }

    #[test]
    fn quote_for_argv_doubles_backslashes_immediately_before_quote() {
        // The canonical CommandLineToArgvW rule: a backslash run adjacent to
        // a `"` must be doubled so the run survives argv splitting. The
        // previous `replace('"', "\\\"")` implementation left such runs
        // single, breaking round-trip for inputs containing `\` + `"`.
        // `validate_windows_path` already bans embedded quotes in service
        // paths, so this case cannot arise from `format_bin_path` today —
        // but `quote_for_argv` is a general helper and must stay correct.
        let argv = quote_for_argv("C:\\path\\\"evil");
        // 3 backslashes before the embedded quote: 2 (doubled run) + 1 (escape).
        assert_eq!(argv, "\"C:\\path\\\\\\\"evil\"");
        assert_eq!(parse_argv(&argv), "C:\\path\\\"evil");
    }

    #[test]
    fn quote_for_argv_doubles_trailing_backslashes() {
        // A trailing `\` sits immediately before the closing quote we emit,
        // so it must be doubled or it would escape our own closing quote.
        let argv = quote_for_argv("C:\\path\\");
        assert_eq!(argv, "\"C:\\path\\\\\"");
        assert_eq!(parse_argv(&argv), "C:\\path\\");
    }

    #[test]
    fn sc_failure_detail_prefers_stderr_when_present() {
        assert_eq!(
            sc_failure_detail(b"ignored stdout\n", b"  spawn error  \n"),
            "spawn error"
        );
    }

    #[test]
    fn sc_failure_detail_falls_back_to_stdout_when_stderr_empty() {
        // sc.exe writes CreateService FAILED / ChangeServiceConfig failures
        // to stdout, leaving stderr empty — without this fallback the
        // operator sees a bare "sc create failed:" with no SCM code.
        assert_eq!(
            sc_failure_detail(
                b"[SC] CreateService FAILED 1073:\r\n\r\nThe specified service already exists.\r\n",
                b""
            ),
            "[SC] CreateService FAILED 1073:\r\n\r\nThe specified service already exists."
        );
    }

    /// Minimal `CommandLineToArgvW` re-implementation for test round-trips.
    /// Not for production use — only enough of the algorithm to validate
    /// that `quote_for_argv` produces values Windows parses back correctly.
    fn parse_argv(cmd_line: &str) -> String {
        let mut out = String::new();
        let mut in_quotes = false;
        let chars: Vec<char> = cmd_line.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            i = match chars[i] {
                '\\' => parse_backslash_run(&chars, i, &mut out, &mut in_quotes),
                '"' => {
                    in_quotes = !in_quotes;
                    i + 1
                }
                c => {
                    out.push(c);
                    i + 1
                }
            };
        }
        out
    }

    /// Consume a run of backslashes starting at `start`. If the run is
    /// followed by `"`, apply the `2n`/`2n+1` backslash-before-quote rule;
    /// otherwise emit the backslashes verbatim. Returns the index after
    /// the consumed run (and the trailing `"` if any).
    fn parse_backslash_run(
        chars: &[char],
        start: usize,
        out: &mut String,
        in_quotes: &mut bool,
    ) -> usize {
        let mut count = 0;
        let mut i = start;
        while i < chars.len() && chars[i] == '\\' {
            count += 1;
            i += 1;
        }
        if i < chars.len() && chars[i] == '"' {
            out.push_str(&"\\".repeat(count / 2));
            if count % 2 == 1 {
                out.push('"');
            } else {
                *in_quotes = !*in_quotes;
            }
            i + 1
        } else {
            out.push_str(&"\\".repeat(count));
            i
        }
    }
}
