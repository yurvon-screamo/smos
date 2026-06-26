//! Reader for the service log tail surfaced by [`super`].
//!
//! [`read_latest_tail`] is split out from the entry point so the
//! filesystem-mocking tests (mtime ordering, Unreadable-vs-Missing
//! distinction) live next to the code they cover without inflating the
//! entry module past its size budget.

use std::path::{Path, PathBuf};

/// Basename prefix the service tracing appender writes under. Re-exported
/// from `tracing_setup::SERVICE_LOG_BASENAME` so the filter stays in
/// lock-step with the appender: switching the appender (e.g.
/// `rolling::never`) updates this filter too, instead of silently
/// stopping matching files.
const LOG_BASENAME: &str = crate::cli::tracing_setup::SERVICE_LOG_BASENAME;

/// Outcome of [`read_latest_tail`]. `Missing` and `Unreadable` are kept
/// distinct so the entry point can tell the operator "log does not exist"
/// vs "log exists but is locked / owned by another account / non-UTF8" —
/// the latter is itself a useful diagnostic.
#[derive(Debug)]
pub(super) enum TailOutcome {
    Missing,
    Unreadable { path: PathBuf, error: String },
    Tail { path: PathBuf, lines: Vec<String> },
}

/// Read the last `n` lines of the most recently modified `smos-service.log*`
/// file under `dir`. Distinguishes three outcomes:
/// - `Missing` — directory or matching file does not exist (probe the
///   next candidate dir).
/// - `Unreadable` — directory or file exists but IO failed (permission
///   denied on the LocalSystem profile for a non-elevated operator,
///   sharing violation from the live appender, non-UTF8). The path +
///   error string let the operator act (elevate, stop the service,
///   check encoding).
/// - `Tail` — the requested lines, in original order.
pub(super) fn read_latest_tail(dir: &Path, n: usize) -> TailOutcome {
    // read_dir failure must NOT collapse into Missing: the LocalSystem
    // logs dir is ACL'd and a non-elevated operator (UAC-token-split
    // admin) typically gets access-denied here. Silently reporting
    // "no log" in that case would hide the crash reason on the exact
    // primary use case this module exists for.
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return TailOutcome::Missing,
        Err(e) => {
            return TailOutcome::Unreadable {
                path: dir.to_path_buf(),
                error: e.to_string(),
            };
        }
    };
    let Some(latest) = latest_log_file(entries) else {
        return TailOutcome::Missing;
    };
    let content = match std::fs::read_to_string(&latest) {
        Ok(c) => c,
        Err(e) => {
            return TailOutcome::Unreadable {
                path: latest,
                error: e.to_string(),
            };
        }
    };
    let tail: Vec<String> = content
        .lines()
        .rev()
        .take(n)
        .map(String::from)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if tail.is_empty() {
        TailOutcome::Missing
    } else {
        TailOutcome::Tail {
            path: latest,
            lines: tail,
        }
    }
}

/// Most recently modified file among `entries` whose name starts with
/// [`LOG_BASENAME`]. `tracing_appender::rolling::daily` emits
/// `smos-service.log.YYYY-MM-DD` files, so mtime ordering picks today's
/// file even though the basename varies.
fn latest_log_file(entries: std::fs::ReadDir) -> Option<PathBuf> {
    let mut picks: Vec<(PathBuf, std::time::SystemTime)> = entries
        .filter_map(std::result::Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with(LOG_BASENAME) {
                let mtime = entry.metadata().ok()?.modified().ok()?;
                Some((entry.path(), mtime))
            } else {
                None
            }
        })
        .collect();
    picks.sort_by_key(|(_, mtime)| *mtime);
    picks.last().map(|(path, _)| path.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::FileTimes;
    use std::time::SystemTime;

    fn set_mtime(path: &Path, at: SystemTime) {
        // Pin deterministic mtimes so the test does not depend on
        // filesystem tick granularity.
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open");
        f.set_times(FileTimes::new().set_modified(at))
            .expect("set_modified");
    }

    #[test]
    fn latest_log_file_picks_most_recent_by_mtime() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let dir = tmp.path();

        let older = dir.join("smos-service.log.2026-06-25");
        std::fs::write(&older, b"old\n").expect("write older");
        set_mtime(
            &older,
            SystemTime::now() - std::time::Duration::from_secs(120),
        );

        let newer = dir.join("smos-service.log.2026-06-26");
        std::fs::write(&newer, b"new\n").expect("write newer");
        set_mtime(&newer, SystemTime::now());

        let latest =
            latest_log_file(std::fs::read_dir(dir).expect("read_dir")).expect("a log file exists");
        assert_eq!(latest.file_name().unwrap(), "smos-service.log.2026-06-26");
    }

    #[test]
    fn latest_log_file_ignores_unrelated_files() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let dir = tmp.path();
        std::fs::write(dir.join("smos-proxy.log"), b"noise").expect("write unrelated");
        std::fs::write(dir.join("README"), b"noise").expect("write readme");
        let entries = std::fs::read_dir(dir).expect("read_dir");
        assert!(latest_log_file(entries).is_none());
    }

    #[test]
    fn read_latest_tail_returns_last_n_lines_in_order() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let dir = tmp.path();
        let body = (1..=10)
            .map(|i| format!("line-{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join("smos-service.log.2026-06-26"), body).expect("write");

        match read_latest_tail(dir, 3) {
            TailOutcome::Tail { path, lines } => {
                assert_eq!(path.file_name().unwrap(), "smos-service.log.2026-06-26");
                assert_eq!(lines, vec!["line-8", "line-9", "line-10"]);
            }
            other => panic!("expected Tail, got {other:?}"),
        }
    }

    #[test]
    fn read_latest_tail_missing_when_dir_absent() {
        let dir = Path::new("C:\\nonexistent\\smos\\diag\\test\\dir");
        assert!(matches!(read_latest_tail(dir, 5), TailOutcome::Missing));
    }

    #[test]
    fn read_latest_tail_unreadable_when_io_fails() {
        // A directory in place of a file: read_to_string fails, which
        // must surface as Unreadable rather than be swallowed into Missing.
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let dir = tmp.path();
        std::fs::create_dir(dir.join("smos-service.log.2026-06-26")).expect("seed dir-as-file");
        match read_latest_tail(dir, 5) {
            TailOutcome::Unreadable { path, .. } => {
                assert_eq!(path.file_name().unwrap(), "smos-service.log.2026-06-26");
            }
            other => panic!("expected Unreadable, got {other:?}"),
        }
    }
}
