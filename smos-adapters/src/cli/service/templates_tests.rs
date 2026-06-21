//! Unit tests for [`super::templates`]. Kept in a sibling file so the
//! template module stays under the 200-line size limit.

use super::*;
use std::path::PathBuf;

fn sample_paths() -> ServicePaths {
    ServicePaths {
        binary: PathBuf::from("/opt/smos/smos"),
        config: PathBuf::from("/opt/smos/smos.toml"),
        working_dir: PathBuf::from("/opt/smos"),
        service_name: "smos".to_string(),
    }
}

const PLACEHOLDERS_SYSTEMD: &[&str] = &[
    "{binary}",
    "{config}",
    "{work_dir}",
    "{service_name}",
    "{wanted_by}",
    "{user_group}",
];

const PLACEHOLDERS_PLIST: &[&str] = &[
    "{binary}",
    "{config}",
    "{work_dir}",
    "{label}",
    "{log_path}",
];

#[test]
fn systemd_system_unit_has_no_leftover_placeholders() {
    let unit = render_systemd_unit_with(&sample_paths(), false, "smos", "smos");
    for placeholder in PLACEHOLDERS_SYSTEMD {
        assert!(
            !unit.contains(placeholder),
            "leftover {placeholder}:\n{unit}"
        );
    }
}

#[test]
fn systemd_user_unit_has_no_leftover_placeholders() {
    let unit = render_systemd_unit_with(&sample_paths(), true, "smos", "smos");
    for placeholder in PLACEHOLDERS_SYSTEMD {
        assert!(
            !unit.contains(placeholder),
            "leftover {placeholder}:\n{unit}"
        );
    }
}

#[test]
fn systemd_system_unit_targets_multi_user_with_default_user_group() {
    let unit = render_systemd_unit_with(&sample_paths(), false, "smos", "smos");
    assert!(unit.contains("ExecStart=/opt/smos/smos serve --config /opt/smos/smos.toml"));
    assert!(unit.contains("WorkingDirectory=/opt/smos"));
    assert!(unit.contains("SyslogIdentifier=smos"));
    assert!(unit.contains("WantedBy=multi-user.target"));
    assert!(!unit.contains("WantedBy=default.target"));
    assert!(unit.contains("User=smos"));
    assert!(unit.contains("Group=smos"));
}

#[test]
fn systemd_system_unit_honors_custom_user_group() {
    let unit = render_systemd_unit_with(&sample_paths(), false, "customuser", "customgroup");
    assert!(unit.contains("User=customuser"));
    assert!(unit.contains("Group=customgroup"));
}

#[test]
fn systemd_user_unit_targets_default_without_user_group() {
    let unit = render_systemd_unit_with(&sample_paths(), true, "smos", "smos");
    assert!(unit.contains("WantedBy=default.target"));
    assert!(!unit.contains("WantedBy=multi-user.target"));
    assert!(!unit.contains("User="));
    assert!(!unit.contains("Group="));
}

#[test]
fn systemd_unit_keeps_security_hardening_directives() {
    let unit = render_systemd_unit_with(&sample_paths(), false, "smos", "smos");
    assert!(unit.contains("NoNewPrivileges=true"));
    assert!(unit.contains("PrivateTmp=true"));
    assert!(unit.contains("Restart=on-failure"));
    assert!(unit.contains("RestartSec=5"));
}

#[test]
fn plist_system_scope_has_no_leftover_placeholders() {
    let plist = render_launchd_plist(&sample_paths(), false);
    for placeholder in PLACEHOLDERS_PLIST {
        assert!(
            !plist.contains(placeholder),
            "leftover {placeholder}:\n{plist}"
        );
    }
}

#[test]
fn plist_user_scope_has_no_leftover_placeholders() {
    let plist = render_launchd_plist(&sample_paths(), true);
    for placeholder in PLACEHOLDERS_PLIST {
        assert!(
            !plist.contains(placeholder),
            "leftover {placeholder}:\n{plist}"
        );
    }
}

#[test]
fn plist_contains_program_arguments_in_order() {
    let plist = render_launchd_plist(&sample_paths(), false);
    let bin_pos = plist
        .find("<string>/opt/smos/smos</string>")
        .expect("plist must contain the binary ProgramArgument");
    let serve_pos = plist
        .find("<string>serve</string>")
        .expect("plist must contain the `serve` ProgramArgument");
    let config_flag_pos = plist
        .find("<string>--config</string>")
        .expect("plist must contain the `--config` ProgramArgument");
    let config_val_pos = plist
        .find("<string>/opt/smos/smos.toml</string>")
        .expect("plist must contain the config value ProgramArgument");
    assert!(bin_pos < serve_pos);
    assert!(serve_pos < config_flag_pos);
    assert!(config_flag_pos < config_val_pos);
}

#[test]
fn plist_has_keepalive_and_runatload() {
    let plist = render_launchd_plist(&sample_paths(), false);
    assert!(plist.contains("<key>RunAtLoad</key>\n    <true/>"));
    assert!(plist.contains("<key>KeepAlive</key>"));
}

#[test]
fn launchd_log_path_system_scope_always_uses_var_log() {
    assert_eq!(
        launchd_log_path_with("smos", false, Some("/Users/test")),
        "/var/log/smos.log"
    );
    assert_eq!(
        launchd_log_path_with("smos", false, None),
        "/var/log/smos.log"
    );
}

#[test]
fn launchd_log_path_user_scope_uses_home_library_logs() {
    assert_eq!(
        launchd_log_path_with("smos", true, Some("/Users/test")),
        "/Users/test/Library/Logs/smos.log"
    );
}

#[test]
fn launchd_log_path_user_scope_falls_back_to_tmp_without_home() {
    // launchd cannot write to /var/log as a regular user — if $HOME
    // is also unset, /tmp is the only universally-writable fallback.
    assert_eq!(launchd_log_path_with("smos", true, None), "/tmp/smos.log");
}

#[test]
fn launchd_log_path_user_scope_never_uses_var_log() {
    // The H1 regression: a user-scope plist pointing at /var/log
    // would crash-loop because launchd runs the agent as the user
    // without write access to /var/log.
    assert_ne!(
        launchd_log_path_with("smos", true, Some("/Users/test")),
        "/var/log/smos.log"
    );
    assert_ne!(
        launchd_log_path_with("smos", true, None),
        "/var/log/smos.log"
    );
}

#[test]
fn launchd_label_uses_reverse_dns_convention() {
    assert_eq!(launchd_label("smos"), "com.smos.smos");
    assert_eq!(launchd_label("custom"), "com.smos.custom");
}

#[test]
fn plist_label_matches_launchctl_lookup_target() {
    // The macos backend queries `launchctl list <label>`; the plist must
    // declare the same `<key>Label</key>` or the lookup silently misses
    // the agent. This pins the two surfaces together.
    let plist = render_launchd_plist(&sample_paths(), false);
    let expected_label = launchd_label(&sample_paths().service_name);
    assert!(
        plist.contains(&format!("<string>{expected_label}</string>")),
        "plist must declare Label={expected_label}:\n{plist}"
    );
}
