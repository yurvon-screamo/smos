//! Operator-profile env-file handoff between install and the service.
//!
//! The `HKLM\...\\Services\\<svc>\\Environment` registry value turned
//! out unreliable — SCM does not always apply it to a LocalSystem
//! service, so the operator's `SMOS_HOME` never reached `smos_home()`.
//! Instead, install writes a plain `KEY=VALUE` file next to the binary,
//! and `service_main` reads it and `set_var`s the values BEFORE the
//! first `smos_home()` / config load. File IO under the binary directory
//! is reliable across all Windows versions and free of SCM quirks.

use std::path::{Path, PathBuf};

const ENV_FILENAME: &str = "smos-service.env";

/// `(NAME, VALUE)` pairs that route `smos_home()` / `user_home_dir()`
/// to the INSTALLING operator's profile. `None` when the operator's
/// home cannot be resolved (no `USERPROFILE` / `HOMEDRIVE+HOMEPATH` in
/// the install shell); the caller treats that as a hard error.
pub(crate) fn build_env_pairs() -> Option<Vec<(String, String)>> {
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

pub(crate) fn env_file_path(binary_dir: &Path) -> PathBuf {
    binary_dir.join(ENV_FILENAME)
}

/// Write the pairs to `<binary_dir>/smos-service.env` as `KEY=VALUE`
/// lines (one per line). Used by install; read back at service start.
pub(crate) fn write_env_file(
    binary_dir: &Path,
    pairs: &[(String, String)],
) -> std::io::Result<PathBuf> {
    let path = env_file_path(binary_dir);
    let body: String = pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&path, body)?;
    Ok(path)
}

/// Read the env file under `binary_dir`. `None` when the file is absent
/// (older install, custom binary location); the caller falls back to
/// the LocalSystem default env.
pub(crate) fn load_env_file(binary_dir: &Path) -> Option<Vec<(String, String)>> {
    let path = env_file_path(binary_dir);
    let content = std::fs::read_to_string(&path).ok()?;
    Some(parse_env_file(&content))
}

/// Parse `KEY=VALUE` lines. Blank/`#` lines and lines without `=` are
/// skipped. The value is everything after the FIRST `=` (so a value may
/// contain `=`).
fn parse_env_file(content: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(eq_idx) = line.find('=') else {
            continue;
        };
        let key = line[..eq_idx].trim().to_string();
        let value = line[eq_idx + 1..].trim().to_string();
        if !key.is_empty() {
            pairs.push((key, value));
        }
    }
    pairs
}

/// Apply the pairs to the process environment via `std::env::set_var`.
/// Called from `service_main` BEFORE any `smos_home()` / config load.
///
/// # Safety
///
/// `set_var` is `unsafe` since 1.85 because a concurrent `getenv` in
/// another thread could observe a torn read. `service_main` is the
/// first user code the process runs — no other thread is alive yet
/// (runtime + handler come later), so the call is race-free.
pub(crate) unsafe fn apply_env_vars(pairs: &[(String, String)]) {
    for (k, v) in pairs {
        unsafe { std::env::set_var(k, v) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_file_path_appends_fixed_filename() {
        let p = env_file_path(Path::new(r"C:\smos"));
        assert_eq!(p.file_name().unwrap(), ENV_FILENAME);
    }

    #[test]
    fn parse_env_file_reads_key_value_lines() {
        let body = "SMOS_HOME=C:\\Users\\me\\.smos\nUSERPROFILE=C:\\Users\\me\n";
        assert_eq!(
            parse_env_file(body),
            vec![
                ("SMOS_HOME".into(), r"C:\Users\me\.smos".into()),
                ("USERPROFILE".into(), r"C:\Users\me".into()),
            ]
        );
    }

    #[test]
    fn parse_env_file_skips_blank_and_comment_lines() {
        let body = "# a comment\n\nKEY1=val1\n  # indented comment\nKEY2=val2\n";
        assert_eq!(
            parse_env_file(body),
            vec![
                ("KEY1".into(), "val1".into()),
                ("KEY2".into(), "val2".into())
            ]
        );
    }

    #[test]
    fn parse_env_file_value_may_contain_equals() {
        // Split on the FIRST `=` only — a value with `=` must survive.
        assert_eq!(
            parse_env_file("TOKEN=a=b=c\n"),
            vec![("TOKEN".into(), "a=b=c".into())]
        );
    }

    #[test]
    fn parse_env_file_skips_lines_without_equals() {
        assert_eq!(
            parse_env_file("KEY1=val1\nMALFORMED\nKEY2=val2\n"),
            vec![
                ("KEY1".into(), "val1".into()),
                ("KEY2".into(), "val2".into())
            ]
        );
    }

    #[test]
    fn write_env_file_round_trips_through_load() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let pairs = vec![
            ("SMOS_HOME".into(), r"C:\u\.smos".into()),
            ("USERPROFILE".into(), r"C:\u".into()),
        ];
        let written = write_env_file(tmp.path(), &pairs).expect("write");
        assert!(written.is_file());
        assert_eq!(load_env_file(tmp.path()).expect("loaded"), pairs);
    }

    #[test]
    fn load_env_file_returns_none_when_absent() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        assert!(load_env_file(tmp.path()).is_none());
    }

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    #[test]
    fn build_env_pairs_includes_smos_home_and_userprofile() {
        let _g = env_lock();
        let pairs = build_env_pairs().expect("env pairs in test shell");
        let names: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert!(names.contains(&"SMOS_HOME"));
        assert!(names.contains(&"USERPROFILE"));
    }

    #[test]
    fn build_env_pairs_smos_home_uses_resolved_path() {
        let _g = env_lock();
        let prior = std::env::var_os("SMOS_HOME");
        let tmp = tempfile::TempDir::new().expect("tempdir");
        // SAFETY: env_lock serialises tests that touch SMOS_HOME; the
        // prior value is restored before return so a previously-set
        // SMOS_HOME survives the test.
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
}
