//! Markdown report writer dreaming tool.

use std::path::PathBuf;
use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};
use smos_application::ports::Clock;
use time::OffsetDateTime;

use super::ToolError;

/// Write a markdown audit report to disk.
///
/// The tool generates the filename from the current UTC timestamp
/// (`audit-YYYYMMDDTHHMMSSZ.md`) so repeated runs never overwrite each
/// other. The configured `report_dir` is created (with parents) on first
/// use; a failure to create it surfaces as a [`ToolError::Io`] so the LLM
/// can decide whether to keep going without a report or retry.
///
/// `Clock` is injected (rather than calling `OffsetDateTime::now_utc()`
/// directly) so the timestamp is deterministic under test and respects the
/// same time source as every other SMOS subsystem.
pub struct WriteReportTool {
    pub report_dir: PathBuf,
    pub clock: Arc<dyn Clock + Send + Sync>,
}

#[derive(Debug, Deserialize)]
pub struct WriteReportArgs {
    /// Full markdown report body. The tool wraps it in nothing — the LLM
    /// is expected to produce a complete, self-describing document per the
    /// system prompt's "Output format" section.
    pub markdown: String,
}

impl Tool for WriteReportTool {
    const NAME: &'static str = "write_report";
    type Args = WriteReportArgs;
    type Output = Value;
    type Error = ToolError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.into(),
            description: "Write the final markdown audit report to disk under \
                          the configured report_dir. The filename is generated \
                          from the current UTC timestamp. Returns the absolute \
                          path of the written file."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "markdown": {"type": "string", "description": "Full markdown body."}
                },
                "required": ["markdown"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.markdown.trim().is_empty() {
            return Err(ToolError::InvalidInput("markdown must not be empty".into()));
        }
        let now = self.clock.now().as_offset_date_time();
        let path = persist_report(&self.report_dir, &args.markdown, now)?;
        tracing::info!(
            tool = Self::NAME,
            path = %path.display(),
            bytes = args.markdown.len(),
            "audit report written"
        );
        Ok(json!({ "path": path.to_string_lossy() }))
    }
}

/// Pure helper: build the per-run filename from a UTC instant.
///
/// Extracted from the IO path so the filename shape can be unit-tested
/// without touching the filesystem. Returns `audit-YYYYMMDDTHHMMSSZ.md`.
fn report_filename(now: OffsetDateTime) -> String {
    use time::macros::format_description;
    // `format_description!` is a compile-time macro: the format string is
    // validated at compile time, so a typo fails the build instead of
    // producing a malformed filename at runtime. The `expect` message
    // documents the invariant — the format is well-formed and the
    // `OffsetDateTime` is always formattable for this layout, so a panic
    // here is a true bug, not a recoverable failure (the previous silent
    // fallback to "audit-unknown.md" would overwrite earlier reports on
    // repeated runs).
    let fmt = format_description!("[year][month][day]T[hour][minute][second]Z");
    let stamp = now
        .format(fmt)
        .expect("OffsetDateTime::format is infallible for the [year][month]day...Z layout");
    format!("audit-{stamp}.md")
}

/// Pure helper: create the report directory (with parents) and write the
/// markdown file. Returns the file path on success.
fn persist_report(
    dir: &PathBuf,
    markdown: &str,
    now: OffsetDateTime,
) -> Result<PathBuf, ToolError> {
    std::fs::create_dir_all(dir)?;
    let filename = report_filename(now);
    let path = dir.join(filename);
    std::fs::write(&path, markdown.as_bytes())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use time::Month;

    #[test]
    fn report_filename_has_canonical_shape() {
        let now = OffsetDateTime::from_unix_timestamp(1_750_000_000).unwrap();
        let name = report_filename(now);
        assert!(
            name.starts_with("audit-20250615T"),
            "expected audit-YYYYMMDDTHHMMSSZ.md prefix, got {name}"
        );
        assert!(name.ends_with("Z.md"), "name = {name}");
    }

    #[test]
    fn report_filename_year_month_day_match_input() {
        let now = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let name = report_filename(now);
        // 1_700_000_000 = 2023-11-14T22:13:20Z
        assert!(name.contains("20231114"), "name = {name}");
        assert_eq!(now.year(), 2023);
        assert_eq!(now.month(), Month::November);
    }

    #[test]
    fn persist_report_creates_missing_dir_and_writes_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("nested").join("reports");
        let now = OffsetDateTime::from_unix_timestamp(1_750_000_000).unwrap();
        let path = persist_report(&dir, "# title\nbody", now).expect("write");
        assert!(path.is_file(), "expected file at {}", path.display());
        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(body, "# title\nbody");
    }

    #[test]
    fn persist_report_writes_empty_body_when_called_directly() {
        // The empty-input check lives in `call`, but `persist_report` will
        // happily write an empty file if asked directly. That asymmetry is
        // fine — the LLM boundary validates, the helper just writes.
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().to_path_buf();
        let now = OffsetDateTime::from_unix_timestamp(1_750_000_000).unwrap();
        let path = persist_report(&dir, "", now).expect("write empty");
        assert!(path.is_file());
    }
}
