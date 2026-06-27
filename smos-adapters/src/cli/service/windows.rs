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

use std::os::windows::process::CommandExt;
use std::process::Command;

use anyhow::{Context, Result, bail};

use super::SERVICE_NAME;
use super::ServiceControl;
use super::paths::ServicePaths;

#[path = "windows_helpers.rs"]
mod helpers;
use helpers::{
    extract_state, format_bin_path, is_admin, quote_for_argv, run_sc, sc_failure_detail,
    service_exists, set_description, set_failure_flag, set_failure_recovery,
};

use super::env_file::{build_env_pairs, write_env_file};
use super::windows_log::print_recent_service_log;
use super::windows_summary::print_install_summary;

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
    set_failure_flag(paths);
    // Write the operator-profile env file next to the binary. The
    // service process loads it at start (service_main →
    // apply_operator_env_file) and `set_var`s every pair BEFORE the
    // first `smos_home()` call, so the LocalSystem service resolves
    // `~/.smos` to the operator's profile. The registry `Environment`
    // value was the original plan but SCM does not reliably apply it to
    // a LocalSystem service across Windows versions, so the file-based
    // handoff is the load-bearing mechanism. Hard error on failure:
    // installing a service that would crash-loop under LocalSystem with
    // an empty config is worse than refusing the install.
    let pairs = build_env_pairs().ok_or_else(|| {
        anyhow::anyhow!(
            "could not resolve the operator home directory; refusing to install a \
             service that would crash-loop under LocalSystem. Set SMOS_HOME and \
             re-run install."
        )
    })?;
    let binary_dir = paths.binary.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "binary path has no parent directory; cannot place smos-service.env: {}",
            paths.binary.display()
        )
    })?;
    if let Err(e) = write_env_file(binary_dir, &pairs) {
        bail!(
            "failed to write {}: {e}; the service would start under LocalSystem with \
             an empty config and crash-loop.",
            super::env_file::env_file_path(binary_dir).display()
        );
    }
    // Propagate the start failure so the operator sees a real error
    // instead of a misleading "installed and started" summary. Linux and
    // macOS already propagate via `?` in their installers; Windows used
    // to silently warn (and warn was a no-op before tracing was wired up).
    run_sc(&["start", &paths.service_name])?;
    print_install_summary(paths, pairs);
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
    run_sc(&["delete", SERVICE_NAME])?;
    // Remove the operator-profile env file written next to the binary
    // at install time. `sc delete` only removes the SCM registry entry;
    // the env file would otherwise survive as an orphan. Best-effort —
    // a leftover file does not break a re-install (write_env_file
    // overwrites) but breaks the leave-no-trace uninstall contract.
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
    {
        let env_file = super::env_file::env_file_path(&dir);
        if env_file.exists()
            && let Err(e) = std::fs::remove_file(&env_file)
        {
            tracing::warn!(error = %e, "failed to remove {}", env_file.display());
        }
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
    println!();
    print_recent_service_log();
    Ok(())
}

fn create_service(paths: &ServicePaths) -> Result<()> {
    if service_exists(&paths.service_name)? {
        bail!(
            "service '{}' already exists; run `smos service uninstall` first",
            paths.service_name
        );
    }
    // `format_bin_path` returns the canonical binPath value SCM will store
    // (binary quoted, no outer wrapping). To forward it as a single argv
    // token to sc.exe we wrap it via `quote_for_argv` (outer quotes +
    // inner `"` escaped as `\"`) and pass it through `raw_arg`.
    //
    // `raw_arg` is critical: `Command::arg` would re-wrap the value in an
    // extra quote layer and double-escape the inner `\"` sequences, so
    // sc.exe receives a token it cannot parse back. That produced
    // `sc create failed:` with no further detail.
    //
    // sc.exe syntax is `binPath= "<value>"` — a space AFTER `binPath=`,
    // then the value in quotes. We emit `binPath=` and the value as
    // separate argv tokens so sc.exe's parameter scanner matches
    // `binPath=`.
    let bin_path_value = format_bin_path(&paths.binary)?;
    let output = Command::new("sc")
        .arg("create")
        .arg(&paths.service_name)
        .arg("binPath=")
        .raw_arg(quote_for_argv(&bin_path_value))
        .args(["DisplayName=", DISPLAY_NAME])
        .args(["start=", "auto"])
        .output()
        .context("failed to spawn sc.exe")?;
    if !output.status.success() {
        let detail = sc_failure_detail(&output.stdout, &output.stderr);

        // Print the diagnostics block BEFORE `bail!` so the operator sees
        // every variable that fed into the failing `sc create` even when
        // the error message itself is truncated by tooling. Each line is
        // an eprintln so it lands on stderr alongside the bail message,
        // not interleaved with stdout progress output.
        eprintln!("--- sc create diagnostics ---");
        eprintln!("smos version: {}", env!("CARGO_PKG_VERSION"));
        eprintln!("generated binPath: {}", bin_path_value);
        eprintln!("binary path: {}", paths.binary.display());
        eprintln!("sc detail: {}", detail);
        eprintln!("------------------------------");

        bail!("sc create failed: {}", detail);
    }
    Ok(())
}
