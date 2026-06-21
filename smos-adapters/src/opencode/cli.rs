//! opencode CLI subprocess wrapper.
//!
//! Runs the local `opencode` binary as a subprocess so discovery can fall back
//! to it when no HTTP server responds. The CLI reads the local opencode SQLite
//! database directly, so no server is required and the GPU is never involved.
//!
//! Mirrors `smos-poc/scripts/opencode_source.py::_run_opencode_cli`.

use std::time::Duration;

use smos_application::errors::ProviderError;
use tokio::process::Command;

/// Wall-clock cap for a single CLI invocation. The POC uses 120 s; kept here so
/// a wedged CLI surfaces as `ProviderError::Timeout`, not a hang.
const CLI_TIMEOUT: Duration = Duration::from_secs(120);

/// Sentinel exit code used in error messages when the OS reports no exit code
/// (e.g. a process killed by a signal on unix). Picked so the message still
/// reads `exited <N>` instead of `exited ?`. Real non-zero exits always carry
/// their actual code via `ExitStatus::code()`; this constant only fills the
/// `None` branch.
const NO_EXIT_CODE: i32 = -1;

/// Run `opencode <args…>` and return its stdout as UTF-8.
///
/// Errors are mapped to [`ProviderError`] variants so callers can apply the
/// same fail-open / retry policy they use for HTTP transport failures:
///
/// - `opencode` not on `PATH` → `Unavailable` (no retry).
/// - spawn failure (other than NotFound) → `Unavailable`.
/// - non-zero exit → `RequestFailed` (caller decides).
/// - timeout → `Timeout`.
/// - non-UTF-8 stdout → `InvalidResponse`.
pub async fn run_opencode_cli(args: &[&str]) -> Result<String, ProviderError> {
    let mut cmd = Command::new("opencode");
    cmd.args(args);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd.stdin(std::process::Stdio::null());

    let output = match tokio::time::timeout(CLI_TIMEOUT, cmd.output()).await {
        Ok(Ok(o)) => o,
        Ok(Err(e)) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ProviderError::Unavailable(
                "`opencode` CLI not found on PATH; cannot reach session data \
                 without a running server"
                    .into(),
            ));
        }
        Ok(Err(e)) => {
            return Err(ProviderError::Unavailable(format!(
                "failed to spawn `opencode`: {e}"
            )));
        }
        Err(_) => return Err(ProviderError::Timeout(CLI_TIMEOUT)),
    };

    let captured = CapturedOutput::from_process(output);
    interpret_cli_output(captured, args)
}

/// Cross-platform snapshot of a subprocess result used by
/// [`interpret_cli_output`]. Decouples the error-mapping logic from
/// `std::process::ExitStatus` (whose construction is platform-specific) so the
/// mapping is unit-testable on every OS without spawning a real subprocess.
#[derive(Debug, Clone)]
struct CapturedOutput {
    success: bool,
    exit_code: Option<i32>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl CapturedOutput {
    /// Build a `CapturedOutput` from a real `tokio::process::Output`.
    fn from_process(output: std::process::Output) -> Self {
        Self {
            success: output.status.success(),
            exit_code: output.status.code(),
            stdout: output.stdout,
            stderr: output.stderr,
        }
    }
}

/// Map a captured subprocess output to either stdout text or a
/// [`ProviderError`]. Extracted from [`run_opencode_cli`] so the error mapping
/// is unit-testable without spawning the real `opencode` binary.
fn interpret_cli_output(output: CapturedOutput, args: &[&str]) -> Result<String, ProviderError> {
    if !output.success {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let trimmed: String = stderr.chars().take(300).collect();
        return Err(ProviderError::RequestFailed(format!(
            "`opencode {}` exited {}: {}",
            args.join(" "),
            output.exit_code.unwrap_or(NO_EXIT_CODE),
            trimmed.trim()
        )));
    }
    String::from_utf8(output.stdout)
        .map_err(|e| ProviderError::InvalidResponse(format!("opencode stdout not UTF-8: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn success(stdout: &str) -> CapturedOutput {
        CapturedOutput {
            success: true,
            exit_code: Some(0),
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
        }
    }

    fn failure(code: i32, stderr: &str) -> CapturedOutput {
        CapturedOutput {
            success: false,
            exit_code: Some(code),
            stdout: Vec::new(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn interpret_success_returns_stdout() {
        let output = success("{\"id\":\"ses_1\"}");
        let result = interpret_cli_output(output, &["session", "list"]);
        assert_eq!(result.unwrap(), "{\"id\":\"ses_1\"}");
    }

    #[test]
    fn interpret_non_zero_exit_returns_request_failed() {
        let output = failure(2, "unknown flag --foo");
        let result = interpret_cli_output(output, &["session", "list"]);
        match result {
            Err(ProviderError::RequestFailed(msg)) => {
                assert!(msg.contains("exited 2"));
                assert!(msg.contains("unknown flag --foo"));
            }
            other => panic!("expected RequestFailed, got {other:?}"),
        }
    }

    #[test]
    fn interpret_stderr_truncated_to_300_chars() {
        let long = "x".repeat(500);
        let output = failure(1, &long);
        let result = interpret_cli_output(output, &["export", "ses_x"]);
        match result {
            Err(ProviderError::RequestFailed(msg)) => {
                let stderr_part = msg.split("exited 1: ").nth(1).unwrap_or("");
                assert!(
                    stderr_part.chars().count() <= 300,
                    "stderr must be truncated to 300 chars"
                );
            }
            other => panic!("expected RequestFailed, got {other:?}"),
        }
    }

    #[test]
    fn interpret_invalid_utf8_returns_invalid_response() {
        let output = CapturedOutput {
            success: true,
            exit_code: Some(0),
            stdout: vec![0xFF, 0xFE, 0xFD],
            stderr: Vec::new(),
        };
        let result = interpret_cli_output(output, &["export", "ses_x"]);
        assert!(matches!(result, Err(ProviderError::InvalidResponse(_))));
    }

    #[test]
    fn interpret_missing_exit_code_uses_sentinel_no_exit_code() {
        // On some platforms a signal-terminated process has no exit code; the
        // mapping falls back to NO_EXIT_CODE so the error message still
        // mentions a numeric "exited" value rather than "exited ?".
        let output = CapturedOutput {
            success: false,
            exit_code: None,
            stdout: Vec::new(),
            stderr: b"signal".to_vec(),
        };
        let result = interpret_cli_output(output, &["export", "ses_x"]);
        match result {
            Err(ProviderError::RequestFailed(msg)) => {
                assert!(msg.contains("exited -1"));
            }
            other => panic!("expected RequestFailed, got {other:?}"),
        }
    }
}
