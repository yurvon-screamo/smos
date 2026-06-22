//! `smos import git <url>` — read fact markdown files from a git repo and
//! re-hydrate them into SurrealDB.
//!
//! Builds an ad-hoc [`GitConfig`](crate::config::GitConfig) from the
//! supplied URL plus the SMOS home (so the clone lands under `~/.smos/git`
//! instead of polluting CWD), opens the clone via
//! [`GitSyncManager::open_or_clone`], reads every `facts/<key>/<id>.md`,
//! and persists each re-hydrated fact through the `FactRepository::save`
//! port.
//!
//! This is the inverse of [`crate::cli::finalize_runner`] — the same
//! frontmatter layout, the same markdown file shape, the opposite
//! direction.

use anyhow::Result;

use crate::SurrealStore;
use crate::cli::tracing_setup::init_tracing_default;
use crate::config::{GitConfig, SmosConfig};
use crate::git_sync::GitSyncManager;
use crate::paths::smos_home;
use smos_application::ports::FactRepository;

/// Parsed `smos import git <url>` invocation.
pub struct ImportGitArgs {
    pub url: String,
}

/// Entry point invoked by the unified `smos` binary's `ImportGit` subcommand.
pub async fn run_import_git(config_path: &str, args: ImportGitArgs) -> Result<()> {
    init_tracing_default();
    let config = SmosConfig::load(config_path)?;

    let store = SurrealStore::connect(
        &config.surreal.path,
        &config.surreal.namespace,
        &config.surreal.database,
    )
    .await?;
    store.run_migrations().await?;

    let git_config = build_git_config(&args.url);
    let mgr = GitSyncManager::open_or_clone(&git_config)?;
    let report = mgr.import_from_repo()?;
    if report.is_empty() {
        if report.skipped > 0 {
            println!(
                "No facts imported from {} — every fact file failed to parse \
                 ({} skipped; see warnings above).",
                args.url, report.skipped
            );
        } else {
            println!("No fact files found in {}", args.url);
        }
        return Ok(());
    }

    // A `FactRepository::save` failure aborts the import on purpose: a DB
    // error (SurrealDB down, schema mismatch, unique-constraint violation)
    // is almost certainly persistent — every subsequent save would fail the
    // same way, so we surface the first failure rather than logging N
    // copies of it and silently dropping the import. A `to_fact` failure,
    // by contrast, is per-file (a corrupt frontmatter does not predict
    // corruption in sibling files), so it is logged and skipped.
    let mut saved = 0usize;
    let mut skipped = report.skipped;
    for (fm, body) in &report.facts {
        match fm.to_fact(body) {
            Ok(fact) => {
                FactRepository::save(&store, &fact).await?;
                saved += 1;
            }
            Err(e) => {
                tracing::warn!(
                    fact_id = %fm.id,
                    memory_key = %fm.memory_key,
                    error = %e,
                    "skipping unparseable fact file"
                );
                skipped += 1;
            }
        }
    }

    println!("\n=== Git import complete ===");
    println!("Repo:    {}", args.url);
    println!("Clone:   {}", mgr.repo_path().display());
    println!("Found:   {} fact files", report.facts.len());
    println!("Saved:   {}", saved);
    if skipped > 0 {
        println!("Skipped: {} (see warnings above)", skipped);
    }
    Ok(())
}

/// Build an ad-hoc `GitConfig` for the import path. The clone lands under
/// `~/.smos/git/import-<timestamp>` so repeated invocations land in separate
/// directories and the operator's CWD is not polluted. Each invocation
/// produces a fresh clone — no reuse across runs.
fn build_git_config(url: &str) -> GitConfig {
    let timestamp = unix_timestamp_secs();
    let local_path = smos_home()
        .join("git")
        .join(format!("import-{timestamp}"))
        .to_string_lossy()
        .into_owned();
    GitConfig {
        repo_url: url.to_string(),
        branch: "main".into(),
        auto_push: false,
        local_path,
        disable_gpg_sign: true,
    }
}

/// Current unix time in seconds. Used to namespace the import clone
/// directory so concurrent or repeated invocations do not collide on the
/// same path. Second-resolution is sufficient because the import flow is
/// operator-driven (a human invokes `smos import git`); two concurrent
/// invocations within the same second would land in the same directory
/// but `git clone` itself serialises on the `.git/` lockfile, so the
/// second clone fails loudly rather than corrupting the first.
fn unix_timestamp_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_git_config_uses_smos_home_layout() {
        let a = build_git_config("https://x");
        assert_eq!(a.repo_url, "https://x");
        assert_eq!(a.branch, "main");
        assert!(!a.auto_push);
        // The clone MUST live under `<smos_home>/git/import-<timestamp>`
        // (forward slashes on Unix, backslashes on Windows) — assert on
        // the trailing segment so the test is path-separator agnostic.
        let tail = std::path::Path::new(&a.local_path)
            .file_name()
            .and_then(|s| s.to_str())
            .expect("local_path has a file name");
        assert!(
            tail.starts_with("import-"),
            "expected trailing `import-<ts>` segment, got {tail:?} (full path {})",
            a.local_path
        );
        let parent = std::path::Path::new(&a.local_path)
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .expect("local_path has parent");
        assert_eq!(
            parent, "git",
            "expected `git` parent segment, got {parent:?}"
        );
    }

    #[test]
    fn unix_timestamp_secs_is_non_negative() {
        let ts = unix_timestamp_secs();
        assert!(ts >= 0, "unix timestamp must be non-negative, got {ts}");
    }
}
