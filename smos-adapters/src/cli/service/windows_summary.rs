//! `smos service install` human-readable summary.
//!
//! Split out of [`super::windows`] so the lifecycle module stays under
//! the size budget. The summary echoes what was installed (binary path)
//! and the operator-profile env vars written into the service registry,
//! plus the two operator-facing gotchas (`smos init` before start; logs
//! location). Injection failure is a hard error upstream (install
//! `bail!`s before reaching this summary), so the summary always has a
//! populated `injected_env`.

use super::paths::ServicePaths;

/// Print the install outcome. `injected_env` carries the
/// `(NAME, VALUE)` pairs written into the service's `Environment`
/// registry value (see [`super::windows_env`]); the summary echoes them
/// so the operator can verify the profile shift.
pub(super) fn print_install_summary(paths: &ServicePaths, injected_env: Vec<(String, String)>) {
    println!("✓ Service '{}' installed and started", paths.service_name);
    println!("  Binary: {}", paths.binary.display());
    println!();
    println!(
        "  Operator profile injected (registry ...\\Services\\{}\\Environment):",
        paths.service_name
    );
    for (name, value) in &injected_env {
        println!("    {name} = {value}");
    }
    println!("  The service resolves ~/.smos (config, db, models, logs) to YOUR");
    println!("  profile via SMOS_HOME — not LocalSystem's systemprofile.");
    println!("  Note: the global `--config` flag is IGNORED on Windows install;");
    println!("  override the config location via SMOS_HOME (or edit the registry");
    println!("  Environment value above).");
    println!("  Run `smos init` first so model files exist before start (avoids");
    println!("  a 643 MB DeBERTa/GGUF download from Session 0).");
    println!("  Logs: <smos_home>/logs/ — `smos service status` prints the tail.");
}
