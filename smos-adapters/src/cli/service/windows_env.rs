//! Operator-profile env injection for the Windows service.
//!
//! A LocalSystem service resolves `~/.smos` to
//! `C:\Windows\System32\config\systemprofile\.smos`, NOT the installing
//! operator's profile — so by default the service loads an empty config
//! (`providers must not be empty`) under SCM and crashes on every start
//! attempt. This module writes the operator's `SMOS_HOME` /
//! `USERPROFILE` / `HOMEDRIVE` / `HOMEPATH` into the service's
//! `Environment` registry value; SCM applies `Environment` to the
//! service process before `ServiceMain` runs, so every `smos_home()` /
//! `user_home_dir()` call inside SMOS resolves to the operator's paths.

use std::process::Command;

use super::paths::ServicePaths;

/// Build the `(NAME, VALUE)` pairs that, when injected into the service's
/// `Environment` registry value, make a LocalSystem service resolve
/// `smos_home()` and `user_home_dir()` to the INSTALLING operator's
/// profile instead of `systemprofile`. Returns `None` when the
/// operator's home cannot be resolved (rare: an install shell with no
/// `USERPROFILE` / `HOMEDRIVE+HOMEPATH`); the caller treats that as a
/// hard error and refuses the install rather than installing a service
/// that would crash-loop under the LocalSystem default profile.
pub(super) fn build_env_pairs() -> Option<Vec<(String, String)>> {
    let smos_home = crate::paths::smos_home();
    let user_home = crate::paths::user_home_dir()?;
    let mut pairs: Vec<(String, String)> = vec![
        ("SMOS_HOME".into(), smos_home.to_string_lossy().into_owned()),
        (
            "USERPROFILE".into(),
            user_home.to_string_lossy().into_owned(),
        ),
    ];
    if let Ok(d) = std::env::var("HOMEDRIVE") {
        pairs.push(("HOMEDRIVE".into(), d));
    }
    if let Ok(p) = std::env::var("HOMEPATH") {
        pairs.push(("HOMEPATH".into(), p));
    }
    Some(pairs)
}

/// Encode the env pairs as a single `reg.exe /d` string for
/// `REG_MULTI_SZ`, using `|` as the multi-string separator. `|` is a
/// reserved character in Windows paths (NTFS forbids it in filenames),
/// so it cannot collide with a value.
pub(super) fn encode_env_multi_string(pairs: &[(String, String)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("|")
}

/// Write the operator-profile env vars into the service's `Environment`
/// registry value via `reg.exe add ... /t REG_MULTI_SZ /s "|"`. Returns
/// the injected pairs so the install summary can echo them. On `Err` the
/// caller (install_service) hard-`bail!`s — installing a service that
/// would crash-loop under LocalSystem with an empty config is worse than
/// refusing the install, so this is NOT best-effort.
pub(super) fn set_service_environment(
    paths: &ServicePaths,
) -> Result<Vec<(String, String)>, String> {
    let Some(pairs) = build_env_pairs() else {
        return Err("could not resolve operator home directory".into());
    };
    let data = encode_env_multi_string(&pairs);
    let key = format!(
        r"HKLM\SYSTEM\CurrentControlSet\Services\{}",
        paths.service_name
    );
    let output = Command::new("reg")
        .args([
            "add",
            &key,
            "/v",
            "Environment",
            "/t",
            "REG_MULTI_SZ",
            "/s",
            "|",
            "/d",
            &data,
            "/f",
        ])
        .output()
        .map_err(|e| format!("spawn reg.exe: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let detail = if !stderr.is_empty() { stderr } else { stdout };
        return Err(format!("reg add Environment failed: {detail}"));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    #[test]
    fn build_env_pairs_includes_smos_home_and_userprofile() {
        let _g = lock();
        let pairs = build_env_pairs().expect("env pairs in test shell");
        let names: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            names.contains(&"SMOS_HOME"),
            "SMOS_HOME must always be injected"
        );
        assert!(
            names.contains(&"USERPROFILE"),
            "USERPROFILE must always be injected"
        );
    }

    #[test]
    fn build_env_pairs_smos_home_uses_resolved_path() {
        let _g = lock();
        let prior = std::env::var_os("SMOS_HOME");
        let tmp = tempfile::TempDir::new().expect("tempdir");
        // SAFETY: test_env_lock serialises all tests that touch SMOS_HOME;
        // the env mutation + read happens under the lock, and the prior
        // value is restored before return so a previously-set SMOS_HOME
        // survives the test.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }
        let pairs = build_env_pairs().expect("env pairs");
        let smos_home = pairs
            .iter()
            .find(|(k, _)| k == "SMOS_HOME")
            .map(|(_, v)| v.clone())
            .expect("SMOS_HOME pair present");
        assert_eq!(smos_home, tmp.path().to_string_lossy());
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
    }

    #[test]
    fn encode_env_multi_string_joins_with_pipe_separator() {
        let pairs = vec![("A".into(), "1".into()), ("B".into(), "2".into())];
        assert_eq!(encode_env_multi_string(&pairs), "A=1|B=2");
    }

    #[test]
    fn encode_env_multi_string_uses_pipe_so_paths_cannot_collide() {
        // Windows paths cannot contain `|` (reserved char), so the
        // separator cannot be confused with a path component.
        let pairs = vec![("PATH".into(), r"C:\dir\sub".into())];
        let encoded = encode_env_multi_string(&pairs);
        assert_eq!(encoded, r"PATH=C:\dir\sub");
        assert!(!encoded.contains('|'));
    }

    #[test]
    fn encode_env_multi_string_single_pair_has_no_separator() {
        let pairs = vec![("ONLY".into(), "value".into())];
        assert_eq!(encode_env_multi_string(&pairs), "ONLY=value");
    }
}
