//! SMOS home-directory resolution (`~/.smos`).
//!
//! Every SMOS-managed artefact lives under a single home directory so the
//! operator can find logs, models, persona files, RocksDB data, and audit
//! reports in one place. The directory is resolved in priority order:
//!
//! 1. `SMOS_HOME` environment variable (operator override).
//! 2. `~/.smos` (the canonical default — `~` is the OS user home).
//! 3. `./.smos` (last-resort fallback for sandboxed environments where the
//!    user home is unavailable, e.g. some container runtimes).
//!
//! All path helpers in the codebase MUST go through [`smos_home`] (or
//! [`SmosPaths::resolve`]) so the override chain stays consistent — a future
//! operator who sets `SMOS_HOME=/var/lib/smos` expects every adapter to
//! honour it, not just the ones that happened to read the env var directly.
//!
//! # Subdirs
//!
//! [`ensure_smos_home`] materialises `db/`, `models/`, `persons/`, `git/`,
//! `logs/`, `reports/`, `cache/`. See [`SMOS_HOME_SUBDIRS`].

use std::path::{Path, PathBuf};

/// Resolve the SMOS home directory.
///
/// Priority: `SMOS_HOME` env > `~/.smos` > `./.smos`. See the module docs for
/// the rationale behind each tier.
pub fn smos_home() -> PathBuf {
    if let Ok(dir) = std::env::var("SMOS_HOME")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Some(home) = user_home_dir() {
        return home.join(".smos");
    }
    PathBuf::from(".smos")
}

/// Resolve the OS user home directory.
///
/// Returns `None` when neither the platform-canonical env var nor a fallback
/// is set (rare in practice, but possible inside containers or stripped-down
/// service accounts).
pub fn user_home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        if let Some(p) = std::env::var_os("USERPROFILE")
            && !p.is_empty()
        {
            return Some(PathBuf::from(p));
        }
        // Windows also exposes HOMEDRIVE + HOMEPATH (e.g. `C:` + `\Users\me`)
        // for legacy roaming profiles. Combine them when USERPROFILE is unset.
        let drive = std::env::var_os("HOMEDRIVE");
        let path = std::env::var_os("HOMEPATH");
        match (drive, path) {
            (Some(d), Some(p)) if !d.is_empty() || !p.is_empty() => {
                let mut combined = PathBuf::from(d);
                combined.push(p);
                Some(combined)
            }
            _ => None,
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var_os("HOME")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
    }
}

/// Ensure `~/.smos` and every well-known subdirectory exist. Creates any
/// missing component with `create_dir_all`. Returns the resolved home path
/// so callers can chain subsequent file IO off it without re-resolving.
pub fn ensure_smos_home() -> std::io::Result<PathBuf> {
    let home = smos_home();
    for subdir in SMOS_HOME_SUBDIRS {
        std::fs::create_dir_all(home.join(subdir))?;
    }
    Ok(home)
}

/// Well-known subdirectories that [`ensure_smos_home`] materialises.
///
/// Kept as a constant so the [`SmosPaths`] resolver and the `smos init`
/// command share the same source of truth.
pub const SMOS_HOME_SUBDIRS: &[&str] =
    &["db", "models", "persons", "git", "logs", "reports", "cache"];

/// Snapshot of every well-known SMOS path resolved against [`smos_home`].
///
/// Built once at startup (or per `smos init`) and consumed by config
/// defaults, the NLI backend, the dreaming report writer, and the persona
/// loader. Cheap to construct; clone-friendly.
#[derive(Debug, Clone)]
pub struct SmosPaths {
    pub home: PathBuf,
    pub config: PathBuf,
    pub db: PathBuf,
    pub models: PathBuf,
    pub persons: PathBuf,
    pub logs: PathBuf,
    pub reports: PathBuf,
}

impl SmosPaths {
    /// Resolve every well-known path against the current [`smos_home`].
    pub fn resolve() -> Self {
        let home = smos_home();
        Self {
            config: home.join("config.toml"),
            db: home.join("db"),
            models: home.join("models"),
            persons: home.join("persons"),
            logs: home.join("logs"),
            reports: home.join("reports"),
            home,
        }
    }
}

/// Resolve the effective config file path.
///
/// Priority:
/// 1. `cli_override` — explicit `--config <path>` from the CLI wins.
/// 2. `./smos.toml` — a local override in the current working directory
///    (matches the operator habit of dropping a project-local config next
///    to where `smos serve` is launched).
/// 3. `~/.smos/config.toml` — the canonical default installed by
///    `smos init`.
pub fn resolve_config_path(cli_override: Option<&str>) -> PathBuf {
    if let Some(p) = cli_override
        && !p.is_empty()
    {
        return PathBuf::from(p);
    }
    let local = Path::new("smos.toml");
    if local.exists() {
        return local.to_path_buf();
    }
    smos_home().join("config.toml")
}

/// Expand a leading `~/`, `~\`, or bare `~` into the user home directory.
///
/// Returns the input unchanged when:
/// - the path does not start with `~`, or
/// - the user home cannot be resolved (e.g. `HOME`/`USERPROFILE` unset).
///
/// Used by the persona loader so `[persons.X].persona = "~/.smos/persons/bob.md"`
/// works regardless of the OS the operator is running on. The
/// `smos_application::helpers::person_router::expand_tilde` helper mirrors
/// this implementation BEHAVIOUR-FOR-BEHAVIOUR — the two MUST stay in sync
/// so persona paths resolve identically on both sides of the application /
/// adapter boundary.
pub fn expand_tilde(path: &str) -> PathBuf {
    let stripped = path
        .strip_prefix("~/")
        .or_else(|| path.strip_prefix("~\\"))
        .or_else(|| path.strip_prefix("~"));
    if let Some(rest) = stripped
        && let Some(home) = user_home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(path)
}
