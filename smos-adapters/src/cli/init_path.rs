//! `PATH` lookup for `smos init` — locate an executable on `PATH` without
//! spawning a `which` / `where` subprocess.
//!
//! Walks `PATH` directly so the result does not depend on a shell helper
//! being installed (some minimal containers ship without `which`). On
//! Windows the `PATHEXT` extensions are honoured so `llama-server.exe` is
//! found even when the caller passes the bare name.
//!
//! Lives in its own module so [`super::init_runner`] stays focused on setup
//! orchestration and the cross-platform quirks have a single, testable home.

use std::path::{Path, PathBuf};

/// Locate `bin` on `PATH`. Returns the first matching executable, or `None`
/// when nothing matches / `PATH` is unset.
pub(crate) fn find_in_path(bin: &str) -> Option<PathBuf> {
    let path_env = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_env) {
        if let Some(found) = probe_dir_for(&dir, bin) {
            return Some(found);
        }
    }
    None
}

/// Try the bare name, then (on Windows) every `PATHEXT` extension.
fn probe_dir_for(dir: &Path, bin: &str) -> Option<PathBuf> {
    let candidate = dir.join(bin);
    if is_executable(&candidate) {
        return Some(candidate);
    }
    for ext in executable_extensions() {
        let with_ext = dir.join(format!("{bin}.{ext}"));
        if is_executable(&with_ext) {
            return Some(with_ext);
        }
    }
    None
}

/// Extensions to append when resolving a bare name. On Windows this honours
/// `PATHEXT` (falling back to the common trio); everywhere else the bare
/// name already carries its form and the list is empty.
fn executable_extensions() -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("PATHEXT")
            .ok()
            .map(|s| {
                s.split(';')
                    .filter_map(|e| {
                        let e = e.trim_start_matches('.').to_ascii_lowercase();
                        (!e.is_empty()).then_some(e)
                    })
                    .collect()
            })
            .unwrap_or_else(|| {
                ["exe", "bat", "cmd"]
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect()
            })
    }
    #[cfg(not(target_os = "windows"))]
    {
        Vec::new()
    }
}

/// A file that exists, is a regular file, and (on Unix) carries at least one
/// execute bit.
fn is_executable(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acquire the workspace-wide env-test lock — this module's positive
    /// test mutates `PATH`, which is process-global.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    /// A name that cannot exist on any sane filesystem proves the helper
    /// fails closed (returns `None`) rather than panicking.
    #[test]
    fn find_in_path_returns_none_for_impossible_name() {
        let bin = "smos-init-definitely-not-present-9f3a1c";
        assert!(find_in_path(bin).is_none());
    }

    /// Drop a fake executable into a tempdir, prepend it to `PATH`, and
    /// confirm `find_in_path` resolves it. Exercises both the bare-name path
    /// (Unix) and the `PATHEXT` extension path (Windows).
    #[test]
    fn find_in_path_resolves_executable_in_prepended_dir() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let bin_name = "smos-init-fake-bin-4d7e2b";
        let exe_name = if cfg!(target_os = "windows") {
            format!("{bin_name}.exe")
        } else {
            bin_name.to_string()
        };
        let exe_path = tmp.path().join(&exe_name);
        std::fs::write(&exe_path, b"#!/bin/sh\n").expect("write fake exe");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&exe_path, std::fs::Permissions::from_mode(0o755))
                .expect("chmod");
        }

        let prior = std::env::var_os("PATH");
        let mut new_path = std::env::split_paths(&std::ffi::OsString::from(
            tmp.path().to_string_lossy().to_string(),
        ))
        .collect::<Vec<_>>();
        if let Some(p) = &prior {
            new_path.extend(std::env::split_paths(p));
        }
        let joined = std::env::join_paths(new_path).expect("join paths");
        // SAFETY: the env-test lock serialises this mutation + read and the
        // prior value is restored before return.
        unsafe {
            std::env::set_var("PATH", &joined);
        }

        let found = find_in_path(bin_name);
        unsafe {
            match prior {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }

        let found = found.expect("fake executable must be found on PATH");
        assert_eq!(
            found.file_name().and_then(|n| n.to_str()),
            Some(exe_name.as_str())
        );
    }
}
