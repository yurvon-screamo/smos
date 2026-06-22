//! Markdown files → `(FactFrontmatter, body)` tuples.
//!
//! Mirror of [`crate::git_sync::export`]: walks `<repo>/facts/**/*.md` in
//! deterministic order (alphabetical by file path) so re-importing the
//! same repo yields the same fact insertion order regardless of the OS's
//! `readdir` ordering.

use std::path::Path;

use walkdir::WalkDir;

use crate::git_sync::export::FACTS_ROOT;
use crate::git_sync::format::{FactFrontmatter, parse_fact_md};

/// Read every fact markdown file under `<repo>/facts/` and return
/// `(frontmatter, body)` pairs ordered by file path.
///
/// Infallible by design: a directory entry that cannot be read, a file
/// that cannot be parsed, or a walkdir error are logged at WARN and
/// skipped so a single corrupt or inaccessible file never aborts the
/// whole import. The caller MUST inspect the `skipped` count on
/// [`ImportReport`] to distinguish a genuinely empty repo from one where
/// every file failed — returning an empty `Vec` without that context
/// would be a silent failure (see project rule: zero tolerance for silent
/// failures).
pub fn read_fact_files(repo_path: &Path) -> ImportReport {
    let facts_dir = repo_path.join(FACTS_ROOT);
    if !facts_dir.is_dir() {
        return ImportReport::default();
    }

    let mut results = Vec::new();
    let mut skipped = 0usize;
    for entry in WalkDir::new(&facts_dir).sort_by_file_name() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "skipping unreadable directory entry during import");
                skipped += 1;
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let path = entry.path();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "skipping unreadable fact file");
                skipped += 1;
                continue;
            }
        };
        match parse_fact_md(&content) {
            Some(parsed) => results.push(parsed),
            None => {
                tracing::warn!(path = %path.display(), "skipping unparseable fact file");
                skipped += 1;
            }
        }
    }
    ImportReport {
        facts: results,
        skipped,
    }
}

/// Outcome of [`read_fact_files`]. Carries both the successfully parsed
/// `(frontmatter, body)` pairs AND the count of files that were skipped
/// due to read / parse failures — the count lets the caller distinguish
/// "repo has no facts yet" from "every fact file was unreadable", which
/// a bare `Vec` would conflate.
#[derive(Debug, Default)]
pub struct ImportReport {
    /// Parsed `(frontmatter, body)` pairs, ordered by file path.
    pub facts: Vec<(FactFrontmatter, String)>,
    /// Number of directory entries / files that were skipped due to a
    /// walkdir error, a read error, or a parse failure. Each skip is
    /// logged at WARN; this counter surfaces the aggregate to the
    /// operator-facing summary.
    pub skipped: usize,
}

impl ImportReport {
    /// `true` when no facts were parsed (regardless of whether any were
    /// skipped). The caller uses this to short-circuit the import loop.
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git_sync::export::write_fact_files;
    use crate::git_sync::test_support::sample_fact;

    #[test]
    fn read_fact_files_round_trips_write_fact_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        let original = vec![
            sample_fact("alpha content", "ns1"),
            sample_fact("beta content", "ns2"),
        ];
        write_fact_files(tmp.path(), &original).expect("write");

        let report = read_fact_files(tmp.path());
        assert_eq!(report.facts.len(), original.len());
        assert_eq!(report.skipped, 0);

        let bodies: Vec<&str> = report.facts.iter().map(|(_, b)| b.as_str()).collect();
        assert!(bodies.contains(&"alpha content"));
        assert!(bodies.contains(&"beta content"));
    }

    #[test]
    fn read_fact_files_returns_empty_when_no_facts_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let report = read_fact_files(tmp.path());
        assert!(report.is_empty());
        assert_eq!(report.skipped, 0);
    }

    #[test]
    fn read_fact_files_skips_non_md_files() {
        let tmp = tempfile::TempDir::new().unwrap();
        let facts_dir = tmp.path().join(FACTS_ROOT).join("ns1");
        std::fs::create_dir_all(&facts_dir).unwrap();
        std::fs::write(facts_dir.join("README.txt"), "not a fact").unwrap();

        let report = read_fact_files(tmp.path());
        assert!(report.is_empty(), "non-md file is not a fact");
        assert_eq!(
            report.skipped, 0,
            ".txt extension is filtered, not counted as skip"
        );
    }

    #[test]
    fn read_fact_files_counts_unparseable_files_as_skipped() {
        let tmp = tempfile::TempDir::new().unwrap();
        let facts_dir = tmp.path().join(FACTS_ROOT).join("ns1");
        std::fs::create_dir_all(&facts_dir).unwrap();
        // A `.md` file whose body is not valid frontmatter+body must be
        // counted in `skipped` (and emit a WARN) — silently dropping it
        // would be a silent failure.
        std::fs::write(facts_dir.join("broken.md"), "no frontmatter at all").unwrap();

        let report = read_fact_files(tmp.path());
        assert!(report.is_empty());
        assert_eq!(
            report.skipped, 1,
            "broken .md must surface in skipped count"
        );
    }
}
