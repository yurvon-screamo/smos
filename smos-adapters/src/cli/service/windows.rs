//! Windows service management via the `sc.exe` Service Control Manager CLI.
//!
//! `sc.exe` is preferred over the `windows-service` crate because:
//! - zero extra native dependencies (smos ships on stock Windows),
//! - the SCM CLI is documented and stable across Windows versions,
//! - failures surface as plain stderr text the operator can paste into a
//!   bug report without a Rust backtrace obscuring the SCM error code.
//!
//! SCM wrappers, path validation, and admin detection live in
//! [`helpers`] so this module contains only the public lifecycle
//! (`install` / `uninstall` / `control` / `status`).

#![cfg(target_os = "windows")]

use std::process::Command;

use anyhow::{Context, Result, bail};

use super::SERVICE_NAME;
use super::ServiceControl;
use super::paths::ServicePaths;

#[path = "windows_helpers.rs"]
mod helpers;
use helpers::{
    extract_state, format_bin_path, is_admin, run_sc, service_exists, set_description,
    set_failure_recovery,
};

const DISPLAY_NAME: &str = "SMOS Semantic Memory OS";
const DESCRIPTION: &str = "SMOS Semantic Memory OS proxy";
/// Reset the failure counter 24h after the last failure.
const FAILURE_RESET_SECONDS: u32 = 86_400;
/// Restart after 5s, then 10s, then 30s for subsequent failures.
const FAILURE_ACTIONS: &str = "restart/5000/restart/10000/restart/30000";
/// SCM `stop` → `start` requires a delay so the SCM state machine can
/// transition through STOPPED before accepting a new START.
const RESTART_SETTLE_DELAY: std::time::Duration = std::time::Duration::from_secs(3);

pub async fn install_service(paths: &ServicePaths, user: bool) -> Result<()> {
    if user {
        bail!("--user is not supported on Windows yet (use Task Scheduler manually)");
    }
    if !is_admin()? {
        bail!("administrator privileges required to install a system service");
    }
    create_service(paths)?;
    set_description(paths);
    set_failure_recovery(paths);
    // Propagate the start failure so the operator sees a real error
    // instead of a misleading "installed and started" summary. Linux and
    // macOS already propagate via `?` in their installers; Windows used
    // to silently warn (and warn was a no-op before tracing was wired up).
    run_sc(&["start", &paths.service_name])?;
    print_install_summary(paths);
    Ok(())
}

pub async fn uninstall_service(user: bool) -> Result<()> {
    if user {
        bail!("--user is not supported on Windows");
    }
    if !service_exists(SERVICE_NAME)? {
        println!("Service '{SERVICE_NAME}' is not installed (nothing to uninstall)");
        return Ok(());
    }
    if let Err(e) = run_sc(&["stop", SERVICE_NAME]) {
        tracing::warn!("failed to stop service before uninstall: {e}");
    }
    let output = Command::new("sc")
        .args(["delete", SERVICE_NAME])
        .output()
        .context("failed to spawn sc.exe")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("sc delete failed: {}", stderr.trim());
    }
    println!("✓ Service '{SERVICE_NAME}' uninstalled");
    Ok(())
}

pub async fn control_service(control: ServiceControl) -> Result<()> {
    if !service_exists(SERVICE_NAME)? {
        bail!("service '{SERVICE_NAME}' is not installed");
    }
    match control {
        ServiceControl::Start => {
            run_sc(&["start", SERVICE_NAME])?;
            println!("✓ Service '{SERVICE_NAME}' started");
        }
        ServiceControl::Stop => {
            run_sc(&["stop", SERVICE_NAME])?;
            println!("✓ Service '{SERVICE_NAME}' stopped");
        }
        ServiceControl::Restart => {
            if let Err(e) = run_sc(&["stop", SERVICE_NAME]) {
                tracing::warn!("failed to stop service during restart: {e}");
            }
            tokio::time::sleep(RESTART_SETTLE_DELAY).await;
            run_sc(&["start", SERVICE_NAME])?;
            println!("✓ Service '{SERVICE_NAME}' restarted");
        }
    }
    Ok(())
}

pub async fn status_service() -> Result<()> {
    if !service_exists(SERVICE_NAME)? {
        println!("Service: {SERVICE_NAME}");
        println!("Status:  NOT INSTALLED");
        return Ok(());
    }
    let stdout = run_sc(&["query", SERVICE_NAME])?;
    let state = extract_state(&stdout);
    println!("Service: {SERVICE_NAME}");
    println!("Status:  {state}");
    println!();
    println!("Raw output:");
    println!("{stdout}");
    Ok(())
}

fn create_service(paths: &ServicePaths) -> Result<()> {
    // format_bin_path already quotes both the binary and config segments
    // (so paths with spaces survive SCM's CommandLineToArgvW parsing) and
    // doubles every internal backslash (so the closing quote of each
    // segment is not escaped). Do NOT wrap the result in additional
    // quotes — `std::process::Command` already escapes the value when
    // forwarding to sc.exe, and double-wrapping produces a binPath the
    // SCM cannot parse back.
    let bin_path_arg = format_bin_path(&paths.binary, &paths.config)?;
    let output = Command::new("sc")
        .args(["create", &paths.service_name])
        .args(["binPath=", &bin_path_arg])
        .args(["DisplayName=", DISPLAY_NAME])
        .args(["start=", "auto"])
        .output()
        .context("failed to spawn sc.exe")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("sc create failed: {}", stderr.trim());
    }
    Ok(())
}

fn print_install_summary(paths: &ServicePaths) {
    println!("✓ Service '{}' installed and started", paths.service_name);
    println!("  Binary: {}", paths.binary.display());
    println!("  Config: {}", paths.config.display());
    println!("  Logs:   Windows Event Viewer > Windows Logs > Application");
}
