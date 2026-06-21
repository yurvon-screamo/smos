//! `smos service` — install/uninstall/start/stop/restart/status SMOS as a
//! system or user service across Windows (sc.exe), Linux (systemd), and
//! macOS (launchd).
//!
//! The clap-exposed [`ServiceAction`] is parsed by the `smos` binary; this
//! module owns the dispatch + platform delegation. Path resolution and
//! unit/plist templates live in [`paths`] and [`templates`] and are
//! platform-independent, so they carry full unit-test coverage on every
//! target. The four platform-specific entry points (`install_service`,
//! `uninstall_service`, `control_service`, `status_service`) are gated by
//! `#[cfg(target_os = …)]` and re-exported here as the `platform` alias.

use anyhow::Result;
use clap::Subcommand;

use crate::cli::tracing_setup::init_tracing_default;

pub mod paths;
pub mod templates;

#[cfg(target_os = "windows")]
mod windows;
#[cfg(target_os = "windows")]
use windows as platform;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as platform;

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
use macos as platform;

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
compile_error!("`smos service` only supports Windows, Linux, and macOS");

pub use paths::{ServicePaths, resolve_paths};

/// Default service identifier across all three platforms.
const SERVICE_NAME: &str = "smos";

/// Concrete control operation to perform on an already-installed service.
///
/// Replaces the previous `&str`-typed action: the dispatch is now
/// exhaustive at the type level so the platform backends cannot drift on
/// string conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceControl {
    Start,
    Stop,
    Restart,
}

/// Subcommand payload parsed by clap. The `smos` binary forwards the
/// parsed `ServiceAction` directly to [`run_service`].
#[derive(Subcommand, Debug)]
pub enum ServiceAction {
    /// Install SMOS as a system service (or user service with `--user`).
    Install {
        /// Install as a user-level service (no admin/root required).
        #[arg(long)]
        user: bool,
    },

    /// Uninstall the SMOS service.
    Uninstall {
        /// Uninstall the user-level service (must match the install flag).
        #[arg(long)]
        user: bool,
    },

    /// Start the service.
    Start,

    /// Stop the service.
    Stop,

    /// Restart the service.
    Restart,

    /// Show service status.
    Status,
}

/// Dispatch a parsed [`ServiceAction`] to the platform-specific backend.
///
/// Only `Install` resolves paths (it needs the binary + config locations);
/// the other actions key off the hardcoded [`SERVICE_NAME`] constant
/// because the platform CLI (`sc.exe`/`systemctl`/`launchctl`) finds the
/// service by name. `config_path` is the global `--config` value (the
/// single source of truth — the per-action `--config` flag was removed to
/// avoid ambiguity between global and local overrides).
///
/// Installs the tracing subscriber so the `tracing::warn!` calls emitted
/// by best-effort operations (description/failure-recovery/unload-on-stop)
/// are visible to the operator — without this, silent-failure fixes
/// (C3/C9/C11) would be no-ops at runtime.
pub async fn run_service(action: ServiceAction, config_path: &str) -> Result<()> {
    init_tracing_default();
    match action {
        ServiceAction::Install { user } => {
            let paths = resolve_paths(config_path)?;
            platform::install_service(&paths, user).await
        }
        ServiceAction::Uninstall { user } => platform::uninstall_service(user).await,
        ServiceAction::Start => platform::control_service(ServiceControl::Start).await,
        ServiceAction::Stop => platform::control_service(ServiceControl::Stop).await,
        ServiceAction::Restart => platform::control_service(ServiceControl::Restart).await,
        ServiceAction::Status => platform::status_service().await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_name_constant_matches_spec() {
        assert_eq!(SERVICE_NAME, "smos");
    }
}
