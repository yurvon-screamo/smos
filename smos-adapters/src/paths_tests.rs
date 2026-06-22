//! Tests for [`super::paths`]. Extracted into its own file so
//! [`super::paths`] stays under the file-size budget.

#![cfg(test)]

use std::path::PathBuf;

use super::paths::{
    SMOS_HOME_SUBDIRS, SmosPaths, ensure_smos_home, expand_tilde, resolve_config_path, smos_home,
};

/// Acquire the workspace-wide env-test lock. See
/// [`crate::test_env_lock`] for why this is required.
fn lock() -> std::sync::MutexGuard<'static, ()> {
    crate::test_env_lock::lock()
}

/// `SMOS_HOME` (when set + non-empty) wins outright. The check is
/// trimmed so an operator who exports `SMOS_HOME="   "` gets the
/// canonical fallback instead of a paths-with-spaces surprise.
#[test]
fn smos_home_prefers_env_override() {
    let _g = lock();
    let prior = std::env::var("SMOS_HOME").ok();
    // SAFETY: `PATHS_TEST_LOCK` is held for the duration of the env
    // mutation + read, and the prior value is restored before return
    // so other tests in the binary see the original state.
    unsafe {
        std::env::set_var("SMOS_HOME", "/tmp/smos-test-home");
    }
    let resolved = smos_home();
    unsafe {
        match prior {
            Some(v) => std::env::set_var("SMOS_HOME", v),
            None => std::env::remove_var("SMOS_HOME"),
        }
    }
    assert_eq!(resolved, PathBuf::from("/tmp/smos-test-home"));
}

#[test]
fn smos_home_ignores_whitespace_only_env_override() {
    let _g = lock();
    let prior = std::env::var("SMOS_HOME").ok();
    // SAFETY: same lock-protected guarantee.
    unsafe {
        std::env::set_var("SMOS_HOME", "   ");
    }
    let resolved = smos_home();
    unsafe {
        match prior {
            Some(v) => std::env::set_var("SMOS_HOME", v),
            None => std::env::remove_var("SMOS_HOME"),
        }
    }
    // A whitespace-only SMOS_HOME must NOT be used; the resolver falls
    // through to the next tier (USERPROFILE/HOME-based ~/.smos).
    assert!(!resolved.as_os_str().is_empty());
    assert_ne!(resolved, PathBuf::from("   "));
}

/// `expand_tilde` is the workhorse behind persona paths. It MUST handle
/// the canonical `~/` prefix, return absolute paths verbatim, and never
/// panic when the user home is unavailable.
#[test]
fn expand_tilde_handles_canonical_cases() {
    // Absolute path passes through unchanged.
    let abs = expand_tilde("/etc/passwd");
    assert_eq!(abs, PathBuf::from("/etc/passwd"));

    // Bare `~` (no trailing slash) also expands to the home dir.
    // We do NOT assert the resolved path exists on disk — CI runners
    // may run under a stripped-down account whose HOME/USERPROFILE
    // points at a non-existent directory. The contract is just
    // "expand_tilde does not panic and returns SOMETHING"; the
    // existence check belongs in the caller (persona loader is
    // fail-soft).
    let _home_only = expand_tilde("~");

    // Empty input is a no-op.
    assert_eq!(expand_tilde(""), PathBuf::from(""));
}

/// `resolve_config_path` mirrors the operator's mental model: CLI
/// override beats CWD beats `~/.smos/config.toml`.
#[test]
fn resolve_config_path_prefers_cli_override() {
    let _g = lock();
    let resolved = resolve_config_path(Some("/custom/path.toml"));
    assert_eq!(resolved, PathBuf::from("/custom/path.toml"));

    // An empty CLI override should NOT win — fall through to the next
    // tier so `--config ""` does not silently break startup.
    let prior = std::env::var("SMOS_HOME").ok();
    // SAFETY: same lock-protected guarantee.
    unsafe {
        std::env::remove_var("SMOS_HOME");
    }
    let fallback = resolve_config_path(Some(""));
    unsafe {
        match prior {
            Some(v) => std::env::set_var("SMOS_HOME", v),
            None => std::env::remove_var("SMOS_HOME"),
        }
    }
    // The fallback path is the canonical ~/.smos/config.toml OR the
    // local ./smos.toml if it happens to exist; either way, it is NOT
    // the empty string the caller passed.
    assert_ne!(fallback, PathBuf::from(""));
}

/// `ensure_smos_home` creates every well-known subdir. We exercise it
/// against `SMOS_HOME` pointed at a fresh tempdir so the user's real
/// `~/.smos` is never touched by the test suite.
#[test]
fn ensure_smos_home_creates_every_well_known_subdir() {
    let _g = lock();
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let prior = std::env::var("SMOS_HOME").ok();
    // SAFETY: same lock-protected guarantee.
    unsafe {
        std::env::set_var("SMOS_HOME", tmp.path());
    }
    let home = ensure_smos_home().expect("ensure");
    unsafe {
        match prior {
            Some(v) => std::env::set_var("SMOS_HOME", v),
            None => std::env::remove_var("SMOS_HOME"),
        }
    }
    assert_eq!(home, tmp.path());
    for subdir in SMOS_HOME_SUBDIRS {
        assert!(
            home.join(subdir).is_dir(),
            "expected ~/.smos/{} to exist after ensure_smos_home",
            subdir
        );
    }
}

/// `SmosPaths::resolve` mirrors `smos_home` and exposes every leaf path
/// the rest of the codebase consumes. Pinned so a refactor that drops a
/// field breaks this test rather than silently breaking a downstream
/// default.
#[test]
fn smos_paths_resolve_exposes_every_well_known_leaf() {
    let _g = lock();
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let prior = std::env::var("SMOS_HOME").ok();
    // SAFETY: same lock-protected guarantee.
    unsafe {
        std::env::set_var("SMOS_HOME", tmp.path());
    }
    let paths = SmosPaths::resolve();
    unsafe {
        match prior {
            Some(v) => std::env::set_var("SMOS_HOME", v),
            None => std::env::remove_var("SMOS_HOME"),
        }
    }
    assert_eq!(paths.home, tmp.path());
    assert_eq!(paths.config, tmp.path().join("config.toml"));
    assert_eq!(paths.db, tmp.path().join("db"));
    assert_eq!(paths.models, tmp.path().join("models"));
    assert_eq!(paths.persons, tmp.path().join("persons"));
    assert_eq!(paths.logs, tmp.path().join("logs"));
    assert_eq!(paths.reports, tmp.path().join("reports"));
}
