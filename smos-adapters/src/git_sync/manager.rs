//! [`GitSyncManager`] — open-or-clone, export, commit, push, import.
//!
//! The manager owns a local clone of the configured git repo and surfaces
//! the four operations SMOS needs:
//!
//! - `Self::open_or_clone` — attach to (or create) the local clone.
//! - `Self::export_facts` — write every fact in a slice to its markdown
//!   file under the clone (does NOT commit).
//! - `Self::commit_and_push` — `git add -A && git commit -m <msg>` and,
//!   when `auto_push` is set, `git push`.
//! - `Self::import_from_repo` — read every markdown fact back into
//!   `(FactFrontmatter, body)` tuples for re-hydration via
//!   `crate::git_sync::format::FactFrontmatter::to_fact`.
//!
//! `git` is invoked via `std::process::Command`; the manager does not embed
//! a git library, so the operator MUST have `git` on `PATH`. Each `git`
//! invocation runs through `run_git`, which surfaces non-zero exit codes
//! as `Err(anyhow!(…))` with the captured stderr.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use smos_domain::Fact;

use crate::config::GitConfig;
use crate::git_sync::export::write_fact_files;
use crate::git_sync::import::{ImportReport, read_fact_files};
use crate::paths::expand_tilde;

/// Git-backed memory sync lifecycle.
///
/// Construct with [`Self::open_or_clone`]; the resulting manager owns the
/// resolved `repo_path` and the original config so subsequent operations
/// (`export_facts`, `commit_and_push`, `import_from_repo`) share the same
/// clone.
pub struct GitSyncManager {
    config: GitConfig,
    repo_path: PathBuf,
}

impl GitSyncManager {
    /// Attach to an existing local clone, or clone `config.repo_url` into
    /// `config.local_path`. After this call returns, `repo_path` is a
    /// working git checkout on the configured branch.
    ///
    /// Empty `repo_url` is treated as "no remote": the local path is
    /// created if missing and the manager works offline. This mirrors the
    /// configuration contract (`[git].repo_url = ""` disables push but
    /// keeps the local layout usable for `smos import git <url>` flows).
    pub fn open_or_clone(config: &GitConfig) -> Result<Self> {
        let repo_path = expand_tilde(&config.local_path);

        if repo_path.join(".git").exists() {
            pull_latest(&repo_path, &config.branch)?;
        } else if !config.repo_url.is_empty() {
            clone_repo(&repo_path, &config.repo_url, &config.branch)?;
        } else {
            std::fs::create_dir_all(&repo_path)
                .with_context(|| format!("create {}", repo_path.display()))?;
            init_local_repo(&repo_path, &config.branch, config.disable_gpg_sign)?;
        }

        Ok(Self {
            config: config.clone(),
            repo_path,
        })
    }

    /// Write every fact in `facts` to its markdown file under the clone.
    /// Does NOT commit; pair with [`Self::commit_and_push`] to land the
    /// changes on the configured branch.
    pub fn export_facts(&self, facts: &[Fact]) -> Result<()> {
        write_fact_files(&self.repo_path, facts)
    }

    /// Stage every change, commit with `message`, and (when `auto_push`)
    /// push to the configured remote. A clean working tree (nothing to
    /// commit) is NOT an error: the commit step is skipped silently and
    /// the push step is skipped too (the remote is already in sync).
    pub fn commit_and_push(&self, message: &str) -> Result<()> {
        run_git(&self.repo_path, &["add", "-A"])?;
        if !has_staged_changes(&self.repo_path)? {
            tracing::info!("git sync: nothing to commit; working tree clean");
            return Ok(());
        }
        run_git(&self.repo_path, &["commit", "-m", message])?;
        if self.config.auto_push {
            run_git(&self.repo_path, &["push", "origin", &self.config.branch])?;
        }
        Ok(())
    }

    /// Read every markdown fact back into `(frontmatter, body)` pairs.
    /// Callers reconstruct domain [`Fact`]s via `FactFrontmatter::to_fact`.
    pub fn import_from_repo(&self) -> Result<ImportReport> {
        Ok(read_fact_files(&self.repo_path))
    }

    /// Read-only accessor for tests / observability.
    pub fn repo_path(&self) -> &Path {
        &self.repo_path
    }
}

/// `git -C <repo> pull --ff-only` against the configured branch.
fn pull_latest(repo_path: &Path, branch: &str) -> Result<()> {
    run_git(repo_path, &["fetch", "origin", branch])?;
    run_git(
        repo_path,
        &["merge", "--ff-only", &format!("origin/{branch}")],
    )
    .or_else(|_| run_git(repo_path, &["checkout", branch]))
}

/// Clone `url` into `repo_path`. The parent directory is created if
/// missing so the operator can point `local_path` at a fresh location.
fn clone_repo(repo_path: &Path, url: &str, branch: &str) -> Result<()> {
    if let Some(parent) = repo_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    run_git(
        repo_path.parent().unwrap_or_else(|| Path::new(".")),
        &[
            "clone",
            "--branch",
            branch,
            url,
            &repo_path.to_string_lossy(),
        ],
    )
}

/// `git init` + initial commit for an offline local-only checkout. Used
/// when `repo_url` is empty but the operator still wants the markdown
/// layout on disk. Creates a regular working-tree repo (NOT a bare repo)
/// because `export_facts` writes files into the working tree.
fn init_local_repo(repo_path: &Path, branch: &str, disable_gpg_sign: bool) -> Result<()> {
    run_git(repo_path, &["init", "--initial-branch", branch])?;
    if disable_gpg_sign {
        run_git(repo_path, &["config", "commit.gpgsign", "false"])?;
    }
    Ok(())
}

/// `true` when `git diff --cached --quiet` reports an unclean index.
fn has_staged_changes(repo_path: &Path) -> Result<bool> {
    let output = Command::new("git")
        .current_dir(repo_path)
        .args(["diff", "--cached", "--quiet"])
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", "true")
        .env("SSH_ASKPASS", "")
        .output()
        .with_context(|| "git diff --cached invocation failed")?;
    Ok(!output.status.success())
}

/// Run a `git` command in `dir`, surfacing non-zero exits with the
/// captured stderr. Stdout is intentionally discarded — every caller is
/// sensitive to failure, not to git's chatter.
///
/// Two environment variables are pinned to defeat interactive prompts so
/// a missing SSH key or HTTPS credential cannot hang `smos serve` /
/// `smos finalize`:
///
/// - `GIT_TERMINAL_PROMPT=0` — disables the in-process credential prompt
///   git opens on stdin. Without this, `git clone` / `git push` against
///   an HTTPS remote without cached credentials would block on stdin
///   forever (the SMOS process owns no interactive TTY).
/// - `GIT_ASKPASS=true` (with the empty-string `core.askPass` config) —
///   defeats the ssh-askpass GUI fallback some distributions ship.
fn run_git(dir: &Path, args: &[&str]) -> Result<()> {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_ASKPASS", "true")
        .env("SSH_ASKPASS", "")
        .output()
        .with_context(|| format!("git {} invocation failed", args.join(" ")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git {} failed: {}", args.join(" "), stderr.trim());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    /// Detect whether `git` is on PATH. Tests that exercise actual git
    /// invocations skip when the binary is missing so CI environments
    /// without git do not spuriously fail.
    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn config(local_path: &Path) -> GitConfig {
        GitConfig {
            repo_url: String::new(),
            branch: "main".into(),
            auto_push: false,
            local_path: local_path.to_string_lossy().into_owned(),
            disable_gpg_sign: true,
        }
    }

    #[test]
    fn open_or_clone_creates_empty_local_repo_when_url_missing() {
        if !git_available() {
            return;
        }
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = config(&tmp.path().join("repo"));
        let mgr = GitSyncManager::open_or_clone(&cfg).expect("open_or_clone");
        assert!(mgr.repo_path().join(".git").exists());
    }

    #[test]
    fn export_then_import_round_trips_through_disk() {
        if !git_available() {
            return;
        }
        use smos_domain::MemoryKey;
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = config(&tmp.path().join("repo"));
        let mgr = GitSyncManager::open_or_clone(&cfg).expect("open_or_clone");

        let fact = crate::git_sync::test_support::sample_fact("Rust is memory-safe.", "origa");
        mgr.export_facts(std::slice::from_ref(&fact))
            .expect("export");
        let report = mgr.import_from_repo().expect("import");
        assert_eq!(report.facts.len(), 1);
        assert_eq!(report.skipped, 0);
        let (fm, body) = &report.facts[0];
        assert_eq!(body, "Rust is memory-safe.");
        let rebuilt = fm.to_fact(body).expect("rehydrate");
        assert_eq!(rebuilt.id(), fact.id());
        let _ = MemoryKey::from_raw("origa").unwrap();
    }

    #[test]
    fn commit_and_push_is_noop_on_clean_tree() {
        if !git_available() {
            return;
        }
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = config(&tmp.path().join("repo"));
        let mgr = GitSyncManager::open_or_clone(&cfg).expect("open_or_clone");
        // No file changes since init → commit must be skipped silently.
        mgr.commit_and_push("noop").expect("noop ok");
    }
}
