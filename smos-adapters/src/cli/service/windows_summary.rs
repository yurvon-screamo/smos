//! `smos service install` human-readable summary.
//!
//! Split out of [`super::windows`] so the lifecycle module stays under
//! the size budget. The summary echoes what was installed (binary path)
//! and the operator-profile env pairs written to the env file next to
//! the binary, plus the operator-facing gotchas (`smos init` before
//! start; logs location). Write failure is a hard error upstream
//! (install `bail!`s before reaching this summary), so the summary
//! always has a populated `env_pairs`.

use std::path::PathBuf;

use super::env_file::env_file_path;
use super::paths::ServicePaths;

/// Print the install outcome. `env_pairs` carries the `(NAME, VALUE)`
/// pairs written to `<binary_dir>/smos-service.env` (see
/// [`super::env_file`]); the summary echoes them so the operator can
/// verify the profile shift.
pub(super) fn print_install_summary(paths: &ServicePaths, env_pairs: Vec<(String, String)>) {
    let env_file: PathBuf = paths
        .binary
        .parent()
        .map(env_file_path)
        .unwrap_or_else(|| env_file_path(&paths.binary));
    println!("✓ Service '{}' installed and started", paths.service_name);
    println!("  Binary: {}", paths.binary.display());
    println!();
    println!("  Operator profile written to:");
    println!("    {}", env_file.display());
    println!("  Pairs the service adopts at start:");
    for (name, value) in &env_pairs {
        println!("    {name} = {value}");
    }
    println!("  The service resolves ~/.smos (config, db, models, logs) to YOUR");
    println!("  profile via SMOS_HOME — not LocalSystem's systemprofile.");
    println!("  Note: the global `--config` flag is IGNORED on Windows install;");
    println!(
        "  override the config location by editing {} (or set SMOS_HOME",
        env_file.display()
    );
    println!("  in your shell and re-running install).");
    println!("  Run `smos init` first so model files exist before start (avoids");
    println!("  a 643 MB DeBERTa/GGUF download from Session 0).");
    println!("  Logs: <smos_home>/logs/ — `smos service status` prints the tail.");
}
