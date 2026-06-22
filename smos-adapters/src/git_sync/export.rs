//! Facts → markdown files.
//!
//! The on-disk layout mirrors the per-namespace storage used in SurrealDB:
//! `<repo>/facts/<memory_key>/<fact_id>.md`. Namespacing by `memory_key`
//! keeps one person's facts out of another person's directory, which makes
//! manual review via `git log -- facts/<memory_key>/` tractable.

use std::path::Path;

use anyhow::{Context, Result};
use smos_domain::Fact;

use crate::git_sync::format::render_fact_md;

/// Root directory for fact files inside a git-sync clone.
pub const FACTS_ROOT: &str = "facts";

/// Write every fact in `facts` to its markdown file under
/// `<repo>/facts/<memory_key>/<fact_id>.md`. Existing files are overwritten
/// so re-exporting after a status change (e.g. `pending → accepted`) lands
/// the new state on disk.
pub fn write_fact_files(repo_path: &Path, facts: &[Fact]) -> Result<()> {
    for fact in facts {
        let dir = repo_path.join(FACTS_ROOT).join(fact.memory_key().as_str());
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let file_path = dir.join(format!("{}.md", fact.id().as_str()));
        let content = render_fact_md(fact)
            .map_err(|e| anyhow::anyhow!("render fact {}: {e}", fact.id().as_str()))?;
        std::fs::write(&file_path, content)
            .with_context(|| format!("write {}", file_path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git_sync::test_support::sample_fact;

    #[test]
    fn write_fact_files_creates_per_namespace_directories() {
        let tmp = tempfile::TempDir::new().unwrap();
        let facts = vec![
            sample_fact("alpha content", "ns1"),
            sample_fact("beta content", "ns2"),
        ];
        write_fact_files(tmp.path(), &facts).expect("write");

        let ns1_dir = tmp.path().join(FACTS_ROOT).join("ns1");
        let ns2_dir = tmp.path().join(FACTS_ROOT).join("ns2");
        assert!(ns1_dir.is_dir());
        assert!(ns2_dir.is_dir());

        let entries: Vec<_> = std::fs::read_dir(&ns1_dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].ends_with(".md"));
    }
}
