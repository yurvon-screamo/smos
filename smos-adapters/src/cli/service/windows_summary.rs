//! `smos service install` human-readable summary.
//!
//! Split out of [`super::windows`] so the lifecycle module stays under
//! the size budget. The summary echoes what was installed (binary path)
//! and the baked `--config` / `--smos-home` paths, plus the operator-
//! facing gotchas (`smos init` before start; logs location).

use super::paths::ServicePaths;

/// Print the install outcome.
pub(super) fn print_install_summary(paths: &ServicePaths) {
    let smos_home = crate::paths::smos_home();
    println!("✓ Service '{}' installed and started", paths.service_name);
    println!("  Binary:   {}", paths.binary.display());
    println!(
        "  Config:   {} (baked into binPath via --config)",
        paths.config.display()
    );
    println!(
        "  SMOS home: {} (baked into binPath via --smos-home)",
        smos_home.display()
    );
    println!();
    println!("  The service resolves ~/.smos (config, db, models, logs) to the");
    println!("  operator profile captured at install time — NOT LocalSystem's");
    println!("  systemprofile. Both paths are explicit in the service binPath.");
    println!("  Run `smos init` first so model files exist before start (avoids");
    println!("  a 643 MB DeBERTa/GGUF download from Session 0).");
    println!(
        "  Logs: {}/logs/ — `smos service status` prints the tail.",
        smos_home.display()
    );
}
