//! macOS service management via launchd (`launchctl`).
//!
//! Installs SMOS as either a system daemon (`/Library/LaunchDaemons/`) or
//! a user agent (`~/Library/LaunchAgents/`). The plist content is rendered
//! by [`super::templates::render_launchd_plist`].
//!
//! Root detection uses `id -u` (POSIX) rather than `libc::getuid()` to
//! avoid the `unsafe` block — SMOS enforces a strict no-`unsafe` policy.

#![cfg(target_os = "macos")]

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use super::SERVICE_NAME;
use super::ServiceControl;
use super::paths::ServicePaths;
use super::templates::{launchd_label, launchd_log_path, render_launchd_plist};

/// Build the launchd label for the SMOS service. Delegated to
/// [`templates::launchd_label`] so the plist `<key>Label</key>` and the
/// `launchctl list <label>` lookup stay in lockstep.
fn label() -> String {
    launchd_label(SERVICE_NAME)
}

pub async fn install_service(paths: &ServicePaths, user: bool) -> Result<()> {
    let plist_content = render_launchd_plist(paths, user);
    let plist_path = plist_path_for_scope(user)?;
    if let Some(parent) = Path::new(&plist_path).parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;
    }
    fs::write(&plist_path, &plist_content)
        .with_context(|| format!("failed to write plist to {plist_path}"))?;
    run_launchctl(&["load", &plist_path])?;
    print_install_summary(paths, &plist_path, user);
    Ok(())
}

pub async fn uninstall_service(user: bool) -> Result<()> {
    let plist_path = plist_path_for_scope(user)?;
    if !Path::new(&plist_path).exists() {
        println!("Service '{SERVICE_NAME}' is not installed (nothing to uninstall)");
        return Ok(());
    }
    if let Err(e) = run_launchctl(&["unload", &plist_path]) {
        tracing::warn!("failed to unload service before uninstall: {e}");
    }
    fs::remove_file(&plist_path).with_context(|| format!("failed to remove plist {plist_path}"))?;
    println!("✓ Service '{SERVICE_NAME}' uninstalled");
    Ok(())
}

pub async fn control_service(control: ServiceControl) -> Result<()> {
    // launchctl has no start/stop verbs — load/unload are the canonical
    // transitions. Restart = unload + load.
    let user = detect_installed_scope()
        .context("service is not installed; run `smos service install` first")?;
    let plist_path = plist_path_for_scope(user)?;
    match control {
        ServiceControl::Start => {
            run_launchctl(&["load", &plist_path])?;
            println!("✓ Service '{SERVICE_NAME}' started");
        }
        ServiceControl::Stop => {
            run_launchctl(&["unload", &plist_path])?;
            println!("✓ Service '{SERVICE_NAME}' stopped");
        }
        ServiceControl::Restart => {
            if let Err(e) = run_launchctl(&["unload", &plist_path]) {
                tracing::warn!("failed to unload service during restart: {e}");
            }
            run_launchctl(&["load", &plist_path])?;
            println!("✓ Service '{SERVICE_NAME}' restarted");
        }
    }
    Ok(())
}

pub async fn status_service() -> Result<()> {
    let label = label();
    if detect_installed_scope().is_none() {
        println!("Service: {SERVICE_NAME}");
        println!("Status:  NOT INSTALLED");
        return Ok(());
    }
    let output = Command::new("launchctl")
        .args(["list", &label])
        .output()
        .context("failed to spawn launchctl")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("Service: {SERVICE_NAME}");
    println!("Label:   {label}");
    println!();
    if !stdout.is_empty() {
        println!("{stdout}");
    } else if !stderr.is_empty() {
        println!("{stderr}");
    } else {
        println!("(no output — service may not be loaded)");
    }
    Ok(())
}

fn plist_path_for_scope(user: bool) -> Result<String> {
    let label = label();
    match user {
        true => {
            let home = std::env::var("HOME").context("HOME env var not set")?;
            Ok(user_plist_path(&home, &label))
        }
        false => {
            if !is_root()? {
                bail!("root required for system daemon (use --user for a user agent)");
            }
            Ok(system_plist_path(&label))
        }
    }
}

/// Build the absolute path of the user-scope launchd plist from a
/// resolved `$HOME` and label. Extracted from [`plist_path_for_scope`]
/// so the string format is unit-testable without env vars.
fn user_plist_path(home: &str, label: &str) -> String {
    format!("{home}/Library/LaunchAgents/{label}.plist")
}

/// Build the absolute path of the system-scope launchd daemon plist.
fn system_plist_path(label: &str) -> String {
    format!("/Library/LaunchDaemons/{label}.plist")
}

/// Detect which scope the operator installed into by checking for the
/// user-level and system-level plist files. Matches the systemd heuristic
/// in [`super::linux`]. Returns `None` when neither plist exists.
fn detect_installed_scope() -> Option<bool> {
    let label = label();
    if let Ok(home) = std::env::var("HOME") {
        if Path::new(&user_plist_path(&home, &label)).exists() {
            return Some(true);
        }
    }
    if Path::new(&system_plist_path(&label)).exists() {
        return Some(false);
    }
    None
}

fn run_launchctl(args: &[&str]) -> Result<()> {
    let output = Command::new("launchctl")
        .args(args)
        .output()
        .with_context(|| format!("failed to spawn launchctl with args {args:?}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("launchctl {:?} failed: {}", args, stderr.trim());
    }
    Ok(())
}

/// Detect root via `id -u` (POSIX). Returns an error if `id` cannot be
/// spawned so the operator sees "id command not found" instead of a
/// misleading "root required".
fn is_root() -> Result<bool> {
    let output = Command::new("id")
        .arg("-u")
        .output()
        .context("failed to spawn `id` (is it on PATH?)")?;
    Ok(String::from_utf8_lossy(&output.stdout).trim() == "0")
}

fn print_install_summary(paths: &ServicePaths, plist_path: &str, user: bool) {
    let log_path = launchd_log_path(&paths.service_name, user);
    println!("✓ Service '{}' installed and loaded", paths.service_name);
    println!("  Plist: {plist_path}");
    println!("  Logs:  {log_path}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_plist_path_joins_home_with_launch_agents() {
        let path = user_plist_path("/Users/smos", "com.smos.smos");
        assert_eq!(path, "/Users/smos/Library/LaunchAgents/com.smos.smos.plist");
    }

    #[test]
    fn system_plist_path_is_under_library_launchdaemons() {
        let path = system_plist_path("com.smos.smos");
        assert_eq!(path, "/Library/LaunchDaemons/com.smos.smos.plist");
    }
}
