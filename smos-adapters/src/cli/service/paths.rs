//! Path resolution for the `smos service` subcommand.
//!
//! Pure and platform-independent: derives absolute paths to the running
//! binary and the config file from `std::env::current_exe()` plus an
//! optional override. No filesystem IO happens here — files are NOT
//! checked for existence up front. The platform installer writes the
//! unit/plist unconditionally and surfaces any problem when systemd /
//! launchd / SCM tries to start the service (the operator sees a
//! non-zero exit in `systemctl status`, `launchctl list`, or the Windows
//! Event Viewer). An upfront existence check would duplicate that
//! feedback path without adding diagnostic value.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::SERVICE_NAME;

const DEFAULT_CONFIG_NAME: &str = "smos.toml";

/// Resolved absolute paths needed to install SMOS as a system service.
///
/// Every field is absolute (binary + working_dir come from `current_exe`;
/// config is either an absolute override or joined to the binary's dir).
#[derive(Debug, Clone)]
pub struct ServicePaths {
    /// Absolute path to the running `smos` executable.
    pub binary: PathBuf,
    /// Absolute path to the config file the service will be started with.
    pub config: PathBuf,
    /// Directory containing the binary — used as the service WorkingDirectory.
    pub working_dir: PathBuf,
    /// Service identifier (default `smos`). Used for unit/plist Label,
    /// `sc.exe` name, and `systemctl` unit name.
    pub service_name: String,
}

/// Resolve the binary, config, and working-dir paths from the running exe.
///
/// `config_override` semantics:
/// - empty string → defaults to `smos.toml` next to the binary (defensive;
///   the global `--config` arg always supplies a value)
/// - absolute path → used verbatim
/// - relative path → joined to the binary's directory (NOT the CWD — the
///   service runs from its own directory, so resolution must match)
pub fn resolve_paths(config_override: &str) -> Result<ServicePaths> {
    let binary = std::env::current_exe().context("failed to get current executable path")?;
    let working_dir = binary
        .parent()
        .context("binary has no parent directory")?
        .to_path_buf();

    let config = resolve_config(&working_dir, config_override);

    Ok(ServicePaths {
        binary,
        config,
        working_dir,
        service_name: SERVICE_NAME.to_string(),
    })
}

fn resolve_config(working_dir: &Path, override_value: &str) -> PathBuf {
    if override_value.is_empty() {
        working_dir.join(DEFAULT_CONFIG_NAME)
    } else if Path::new(override_value).is_absolute() {
        PathBuf::from(override_value)
    } else {
        working_dir.join(override_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_paths_uses_current_exe_parent_as_working_dir() {
        let paths = resolve_paths("smos.toml").expect("resolve_paths");
        let exe = std::env::current_exe().expect("current_exe");
        assert_eq!(paths.binary, exe);
        assert_eq!(paths.working_dir, exe.parent().unwrap());
    }

    #[test]
    fn resolve_paths_relative_config_is_joined_to_working_dir() {
        let paths = resolve_paths("custom.toml").expect("resolve_paths");
        let expected = paths.working_dir.join("custom.toml");
        assert_eq!(paths.config, expected);
    }

    #[test]
    fn resolve_paths_absolute_config_is_used_verbatim() {
        let abs = if cfg!(windows) {
            "C:\\etc\\smos\\smos.toml"
        } else {
            "/etc/smos/smos.toml"
        };
        let paths = resolve_paths(abs).expect("resolve_paths");
        assert_eq!(paths.config, PathBuf::from(abs));
    }

    #[test]
    fn resolve_paths_empty_override_falls_back_to_smos_toml() {
        let paths = resolve_paths("").expect("resolve_paths");
        assert_eq!(paths.config, paths.working_dir.join("smos.toml"));
    }

    #[test]
    fn service_name_defaults_to_smos_constant() {
        let paths = resolve_paths("smos.toml").expect("resolve_paths");
        assert_eq!(paths.service_name, SERVICE_NAME);
        assert_eq!(paths.service_name, "smos");
    }

    #[test]
    fn resolve_config_handles_all_three_branches() {
        let wd = Path::new("/opt/smos");
        assert_eq!(resolve_config(wd, ""), wd.join("smos.toml"));
        assert_eq!(
            resolve_config(wd, "/etc/smos.toml"),
            PathBuf::from("/etc/smos.toml")
        );
        assert_eq!(resolve_config(wd, "custom.toml"), wd.join("custom.toml"));
    }
}
