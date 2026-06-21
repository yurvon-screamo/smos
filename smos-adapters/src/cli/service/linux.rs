//! Linux service management via systemd (`systemctl`).
//!
//! Installs SMOS as either a system service (`/etc/systemd/system/`) or
//! a user service (`~/.config/systemd/user/`). The unit file content is
//! rendered by [`super::templates::render_systemd_unit`].
//!
//! Root detection uses `id -u` (POSIX) rather than `libc::getuid()` to
//! avoid the `unsafe` block — SMOS enforces a strict no-`unsafe` policy.

#![cfg(target_os = "linux")]

use std::fs;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use super::SERVICE_NAME;
use super::ServiceControl;
use super::paths::ServicePaths;
use super::templates::render_systemd_unit;

const SYSTEM_UNIT_PATH: &str = "/etc/systemd/system/smos.service";

pub async fn install_service(paths: &ServicePaths, user: bool) -> Result<()> {
    let unit_content = render_systemd_unit(paths, user);
    let (unit_path, scope) = unit_path_for_scope(user)?;
    fs::write(&unit_path, &unit_content)
        .with_context(|| format!("failed to write unit file to {unit_path}"))?;
    run_systemctl(scope, &["daemon-reload"])?;
    run_systemctl(scope, &["enable", &paths.service_name])?;
    run_systemctl(scope, &["start", &paths.service_name])?;
    print_install_summary(paths, &unit_path, scope);
    Ok(())
}

pub async fn uninstall_service(user: bool) -> Result<()> {
    let (unit_path, scope) = unit_path_for_scope(user)?;
    if !Path::new(&unit_path).exists() {
        println!("Service '{SERVICE_NAME}' is not installed (nothing to uninstall)");
        return Ok(());
    }
    if let Err(e) = run_systemctl(scope, &["stop", SERVICE_NAME]) {
        tracing::warn!("failed to stop service before uninstall: {e}");
    }
    if let Err(e) = run_systemctl(scope, &["disable", SERVICE_NAME]) {
        tracing::warn!("failed to disable service before uninstall: {e}");
    }
    fs::remove_file(&unit_path)
        .with_context(|| format!("failed to remove unit file {unit_path}"))?;
    run_systemctl(scope, &["daemon-reload"])?;
    println!("✓ Service '{SERVICE_NAME}' uninstalled");
    Ok(())
}

pub async fn control_service(control: ServiceControl) -> Result<()> {
    let scope = detect_installed_scope()
        .context("service is not installed; run `smos service install` first")?;
    let (verb, past) = match control {
        ServiceControl::Start => ("start", "started"),
        ServiceControl::Stop => ("stop", "stopped"),
        ServiceControl::Restart => ("restart", "restarted"),
    };
    run_systemctl(scope, &[verb, SERVICE_NAME])?;
    println!("✓ Service '{SERVICE_NAME}' {past}");
    Ok(())
}

pub async fn status_service() -> Result<()> {
    let scope = match detect_installed_scope() {
        Some(scope) => scope,
        None => {
            println!("Service: {SERVICE_NAME}");
            println!("Status:  NOT INSTALLED");
            return Ok(());
        }
    };
    // systemctl returns non-zero when the service is stopped — that is
    // information, not an error, so we never bail here.
    let output = Command::new("systemctl")
        .args(scope_args(scope))
        .args(["status", SERVICE_NAME])
        .output()
        .context("failed to spawn systemctl")?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stdout.is_empty() {
        println!("{stdout}");
    }
    if !stderr.is_empty() {
        eprintln!("{stderr}");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Scope {
    System,
    User,
}

fn unit_path_for_scope(user: bool) -> Result<(String, Scope)> {
    match user {
        true => {
            let home = std::env::var("HOME").context("HOME env var not set")?;
            let dir = format!("{home}/.config/systemd/user");
            fs::create_dir_all(&dir)
                .with_context(|| format!("failed to create directory {dir}"))?;
            Ok((user_unit_path(&home), Scope::User))
        }
        false => {
            if !is_root()? {
                bail!("root required for system service (use --user for a user service)");
            }
            Ok((SYSTEM_UNIT_PATH.to_string(), Scope::System))
        }
    }
}

/// Build the absolute path of the user-scope systemd unit file from a
/// resolved `$HOME`. Extracted from [`unit_path_for_scope`] so the
/// string format is unit-testable without depending on env vars.
fn user_unit_path(home: &str) -> String {
    format!("{home}/.config/systemd/user/{SERVICE_NAME}.service")
}

/// Detect which scope the operator installed into, so start/stop/status
/// hit the right `systemctl` instance without an explicit flag.
///
/// Returns `None` when neither the user nor the system unit file is
/// present so callers can surface a uniform "NOT INSTALLED" message
/// instead of letting `systemctl` emit a confusing "Unit not loaded".
fn detect_installed_scope() -> Option<Scope> {
    if let Ok(home) = std::env::var("HOME") {
        if Path::new(&user_unit_path(&home)).exists() {
            return Some(Scope::User);
        }
    }
    if Path::new(SYSTEM_UNIT_PATH).exists() {
        return Some(Scope::System);
    }
    None
}

fn scope_args(scope: Scope) -> &'static [&'static str] {
    match scope {
        Scope::System => &[],
        Scope::User => &["--user"],
    }
}

fn run_systemctl(scope: Scope, args: &[&str]) -> Result<()> {
    let output = Command::new("systemctl")
        .args(scope_args(scope))
        .args(args)
        .output()
        .with_context(|| format!("failed to spawn systemctl with args {args:?}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("systemctl {:?} failed: {}", args, stderr.trim());
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

fn print_install_summary(paths: &ServicePaths, unit_path: &str, scope: Scope) {
    let journalctl = match scope {
        Scope::System => format!("journalctl -u {}", paths.service_name),
        Scope::User => format!("journalctl --user -u {}", paths.service_name),
    };
    println!("✓ Service '{}' installed and started", paths.service_name);
    println!("  Unit: {unit_path}");
    println!("  Logs: {journalctl}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_unit_path_joins_home_with_config_systemd_user() {
        let path = user_unit_path("/home/smos");
        assert_eq!(path, "/home/smos/.config/systemd/user/smos.service");
    }
}
