//! `smos` binary presence + version check.
//!
//! The doctor itself is part of the `smos-adapters` crate, so when the
//! operator runs it from the workspace root, `target/{debug,release}/`
//! contains every binary that the workspace produces. We probe both
//! profiles (release first — operators should smoke against the release
//! build) and pick whichever exists.

use std::path::Path;

use super::super::types::CheckResult;

/// Build the canonical binary name with the platform-appropriate suffix.
/// Windows appends `.exe`; unix does not. The doctor must run on the host
/// that produced the binary, so the host's exe suffix is the right one.
fn binary_name(profile: &str) -> String {
    let base = format!("target/{profile}/smos");
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base
    }
}

/// Probe the workspace `target/` directory for the smos binary and
/// return one [`CheckResult`]. Always returns a row so the report lists
/// the binary check even on a fresh checkout (where the binary has not
/// been built yet).
pub async fn check_binaries() -> Vec<CheckResult> {
    let release_name = binary_name("release");
    let debug_name = binary_name("debug");
    let release = Path::new(&release_name);
    let debug = Path::new(&debug_name);

    let (path_str, profile) = if release.exists() {
        (release.to_string_lossy().to_string(), "release")
    } else if debug.exists() {
        (debug.to_string_lossy().to_string(), "debug")
    } else {
        return vec![
            CheckResult::fail("smos binary", "not found in target/{release,debug}")
                .with_recommendation("run `cargo build --release --bin smos`"),
        ];
    };

    let version = env!("CARGO_PKG_VERSION");
    let details = format!("version: {version}, profile: {profile}, path: {path_str}");
    vec![CheckResult::pass("smos binary", details)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_name_appends_exe_suffix_on_windows() {
        let name = binary_name("release");
        if cfg!(windows) {
            assert!(name.ends_with("smos.exe"));
        } else {
            assert!(name.ends_with("smos"));
            assert!(!name.ends_with(".exe"));
        }
    }

    #[test]
    fn binary_name_includes_profile_segment() {
        let release = binary_name("release");
        let debug = binary_name("debug");
        assert!(release.contains("target/release/"));
        assert!(debug.contains("target/debug/"));
    }
}
