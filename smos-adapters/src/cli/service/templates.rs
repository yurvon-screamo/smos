//! Service unit/plist file templates for systemd (Linux) and launchd (macOS).
//!
//! Placeholders use the `{name}` syntax (NOT `format!`'s `{}`) so the
//! templates can carry literal braces (XML attribute lists, plist dicts)
//! without escaping. Substitution is a plain `str::replace` chain — no
//! formatter, no escapes, no surprises.
//!
//! Both renderers take a `user: bool` flag because the unit/plist content
//! differs fundamentally between user and system scope:
//! - systemd user units target `default.target` and inherit the operator's
//!   identity (no `User=`/`Group=`); system units target `multi-user.target`
//!   and drop privileges to a dedicated `smos` system account.
//! - launchd user agents log under `~/Library/Logs/`; system daemons log
//!   under `/var/log/` (root-only writable).

use super::paths::ServicePaths;

/// Default service account for system-scope systemd units. Operator is
/// expected to provision this user before installing the system service
/// (e.g. `useradd --system smos`). Override with `SMOS_SERVICE_USER` /
/// `SMOS_SERVICE_GROUP` env vars when the operator wants a different
/// runtime identity.
const DEFAULT_SYSTEM_USER: &str = "smos";

/// Render a systemd unit file for the SMOS service.
///
/// `user = true` → `~/.config/systemd/user/` unit (`WantedBy=default.target`,
/// no `User=`/`Group=` — inherits the installer's identity).
/// `user = false` → `/etc/systemd/system/` unit (`WantedBy=multi-user.target`,
/// drops to the `smos` system account via `User=`/`Group=`).
///
/// Security hardening (`NoNewPrivileges`, `PrivateTmp`) is always on. The
/// unit deliberately does NOT drop network/filesystem access because SMOS
/// reads its data directory and proxies upstream HTTP.
pub fn render_systemd_unit(paths: &ServicePaths, user: bool) -> String {
    let (system_user, system_group) = systemd_system_identity();
    render_systemd_unit_with(paths, user, &system_user, &system_group)
}

/// Same as [`render_systemd_unit`] but accepts the system identity
/// explicitly so tests can exercise both default and override paths
/// without mutating process-wide env vars (forbidden under the no-`unsafe`
/// policy because `env::set_var` is `unsafe` since Rust 2024).
fn render_systemd_unit_with(
    paths: &ServicePaths,
    user: bool,
    system_user: &str,
    system_group: &str,
) -> String {
    let wanted_by = if user {
        "default.target"
    } else {
        "multi-user.target"
    };
    let user_group = if user {
        String::new()
    } else {
        format!("User={system_user}\nGroup={system_group}\n")
    };
    SYSTEMD_TEMPLATE
        .replace("{binary}", &paths.binary.to_string_lossy())
        .replace("{config}", &paths.config.to_string_lossy())
        .replace("{work_dir}", &paths.working_dir.to_string_lossy())
        .replace("{service_name}", &paths.service_name)
        .replace("{wanted_by}", wanted_by)
        .replace("{user_group}", &user_group)
}

/// Render a launchd plist for the SMOS service.
///
/// `user = true` → `~/Library/LaunchAgents/` agent (logs to
/// `~/Library/Logs/<service>.log`, falls back to `/tmp/` if `$HOME` is
/// unset). `user = false` → `/Library/LaunchDaemons/` daemon (logs to
/// `/var/log/<service>.log`). `RunAtLoad=true` starts on load;
/// `KeepAlive` with `SuccessfulExit=false` respawns only on non-zero exit
/// so a clean `smos shutdown` does NOT trigger a restart loop.
pub fn render_launchd_plist(paths: &ServicePaths, user: bool) -> String {
    let home = std::env::var("HOME").ok();
    let log_path = launchd_log_path_with(&paths.service_name, user, home.as_deref());
    let label = launchd_label(&paths.service_name);
    PLIST_TEMPLATE
        .replace("{binary}", &paths.binary.to_string_lossy())
        .replace("{config}", &paths.config.to_string_lossy())
        .replace("{work_dir}", &paths.working_dir.to_string_lossy())
        .replace("{label}", &label)
        .replace("{log_path}", &log_path)
}

/// Build the reverse-DNS launchd label from the service identifier.
///
/// Exposed publicly so the macOS backend (`macos.rs`) and the plist
/// template stay in lockstep — `launchctl list <label>` only finds the
/// agent if the `<key>Label</key>` in the plist matches.
pub fn launchd_label(service_name: &str) -> String {
    format!("com.smos.{service_name}")
}

/// Resolve the launchd log path for the running service.
///
/// Public wrapper around [`launchd_log_path_with`] that reads `HOME`
/// from the process environment. Reused by both `render_launchd_plist`
/// and the macOS backend's `print_install_summary` so the path the
/// plist points at and the path the install summary advertises never
/// drift apart.
pub fn launchd_log_path(service_name: &str, user: bool) -> String {
    let home = std::env::var("HOME").ok();
    launchd_log_path_with(service_name, user, home.as_deref())
}

/// Resolve the launchd log path for the given scope and home directory.
///
/// Split out from [`render_launchd_plist`] so tests can cover the
/// `/var/log` vs `~/Library/Logs` vs `/tmp` fallback matrix without
/// touching process env vars.
fn launchd_log_path_with(service_name: &str, user: bool, home: Option<&str>) -> String {
    match (user, home) {
        (true, Some(h)) => format!("{h}/Library/Logs/{service_name}.log"),
        (true, None) => format!("/tmp/{service_name}.log"),
        (false, _) => format!("/var/log/{service_name}.log"),
    }
}

/// Read the SMOS_SERVICE_USER / SMOS_SERVICE_GROUP overrides (or fall
/// back to the dedicated `smos` account). Group defaults to the same
/// value as User when only the user override is set.
fn systemd_system_identity() -> (String, String) {
    let user =
        std::env::var("SMOS_SERVICE_USER").unwrap_or_else(|_| DEFAULT_SYSTEM_USER.to_string());
    let group = std::env::var("SMOS_SERVICE_GROUP").unwrap_or_else(|_| user.clone());
    (user, group)
}

const SYSTEMD_TEMPLATE: &str = "\
[Unit]
Description=SMOS Semantic Memory OS proxy
After=network.target

[Service]
Type=simple
ExecStart={binary} serve --config {config}
WorkingDirectory={work_dir}
{user_group}Restart=on-failure
RestartSec=5

StandardOutput=journal
StandardError=journal
SyslogIdentifier={service_name}

NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy={wanted_by}
";

const PLIST_TEMPLATE: &str = "\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
<plist version=\"1.0\">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary}</string>
        <string>serve</string>
        <string>--config</string>
        <string>{config}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
</dict>
</plist>
";

#[cfg(test)]
#[path = "templates_tests.rs"]
mod tests;
