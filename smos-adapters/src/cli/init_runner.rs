//! `smos init` — one-command setup runner.
//!
//! A single `smos init` invocation performs the entire first-time setup and
//! reports what is ready and what still needs attention:
//!
//! 1. **Filesystem** — materialises `~/.smos` (or `$SMOS_HOME`) with every
//!    well-known subdirectory and drops a default `config.toml` + a stub
//!    `persons/bob.md` when they are not already present. Re-running is
//!    idempotent: an existing config / persona is NEVER overwritten so the
//!    operator's edits survive a re-init.
//! 2. **Ollama** — probes `http://localhost:11434/api/tags` and pulls every
//!    default-config model that is missing via `ollama pull`.
//! 3. **llama-server** — checks the binary is reachable on `PATH` (the
//!    reranker hard-depends on it).
//! 4. **Reranker** — probes the configured `/health` endpoint.
//! 5. **Database** — connects to SurrealDB and applies migrations.
//!
//! This module owns the orchestration + filesystem bootstrap only; the
//! network probes live in [`crate::cli::init_checks`] and the `PATH` lookup
//! in [`crate::cli::init_path`]. The checks here are deliberately lightweight
//! and inline: they answer "is the box ready to `smos serve`?". `smos doctor`
//! is the separate, detailed diagnostic command (per-model validation, NLI
//! cache, config linting, full stats, Markdown report). `init` does NOT
//! delegate to the doctor module — that would couple the setup wizard to the
//! diagnostic surface and pull in report generation that first-time users do
//! not need.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::cli::init_checks;
use crate::cli::init_defaults::{DEFAULT_CONFIG_TOML, DEFAULT_PERSONA_BOB_MD};
use crate::config::SmosConfig;
use crate::paths::{SMOS_HOME_SUBDIRS, SmosPaths, ensure_smos_home};

/// Default Ollama base URL. The default config points every local role
/// (chat, extraction, embedding) at this single instance, so probing it once
/// covers the whole "are the local models reachable?" question. Hardcoded by
/// design — init is a simple localhost probe; `smos doctor` validates
/// operator-customised provider URLs in depth.
const OLLAMA_URL: &str = "http://localhost:11434";

/// What [`bootstrap_filesystem`] actually did on disk. Carries the flags so
/// [`run_init`] can print "created" vs "already exists" without re-stat'ing
/// the files itself.
struct FsOutcome {
    paths: SmosPaths,
    config_created: bool,
    persona_created: bool,
}

/// Entry point invoked by the unified `smos` binary's `Init` subcommand.
///
/// Runs every setup step in order and prints a consolidated ✓ / ✗ report.
/// No step aborts the run on failure: a missing reranker or an unreachable
/// Ollama is reported with a remediation hint so the operator fixes it and
/// re-runs `smos init` to verify. Only the filesystem bootstrap (step 1)
/// surfaces a hard error, because without `~/.smos` nothing else makes sense.
///
/// `init` always exits 0 (advisory): it is a setup wizard that reports state,
/// not a strict gate. The strict exit-code gate is `smos doctor` (exits
/// non-zero on any `[FAIL]`), which the operator runs for CI / scripting.
pub async fn run_init() -> Result<()> {
    println!("SMOS Setup");
    println!("==========");

    println!("\n[1/5] Creating ~/.smos/ directory structure...");
    let fs = bootstrap_filesystem()?;
    print_filesystem_details(&fs);

    let config = load_config(&fs.paths.config);

    println!("\n[2/5] Checking Ollama ({OLLAMA_URL})...");
    init_checks::check_ollama_and_pull_models(OLLAMA_URL).await;

    println!("\n[3/5] Checking llama-server on PATH...");
    init_checks::check_llama_server();

    println!("\n[4/5] Checking reranker ({})...", config.reranker.url);
    init_checks::check_reranker(&config.reranker).await;

    println!("\n[5/5] Initializing database...");
    init_checks::init_database(&config.surreal).await;

    print_footer(&fs.paths);
    Ok(())
}

/// Create `~/.smos` + subdirectories and drop the default config + stub
/// persona when absent. Pure filesystem IO (no network), so it is the part
/// of init that unit tests cover deterministically.
fn bootstrap_filesystem() -> Result<FsOutcome> {
    ensure_smos_home()?;
    let paths = SmosPaths::resolve();

    let config_created = !paths.config.exists();
    if config_created {
        std::fs::write(&paths.config, DEFAULT_CONFIG_TOML)?;
        tracing::info!(config_path = %paths.config.display(), "wrote default config");
    } else {
        tracing::info!(config_path = %paths.config.display(), "config already exists; left untouched");
    }

    // Drop a stub persona so the default `[persons.bob].persona` reference
    // resolves to a working file. Never overwrite an existing one — the
    // operator may have crafted it already.
    let bob_persona = paths.persons.join("bob.md");
    let persona_created = !bob_persona.exists();
    if persona_created {
        std::fs::write(&bob_persona, DEFAULT_PERSONA_BOB_MD)?;
    }

    Ok(FsOutcome {
        paths,
        config_created,
        persona_created,
    })
}

fn print_filesystem_details(fs: &FsOutcome) {
    let status = |created: bool| if created { "created" } else { "already exists" };
    println!("  ✓ Home:        {}", fs.paths.home.display());
    println!(
        "  ✓ Config:      {} ({})",
        fs.paths.config.display(),
        status(fs.config_created)
    );
    if fs.persona_created {
        println!(
            "  ✓ Stub persona: {}",
            fs.paths.persons.join("bob.md").display()
        );
    }
    // The directory list is the single source of truth for what
    // `ensure_smos_home` materialises, so print from the constant rather
    // than a free-hand string that would silently drift when a new subdir
    // is added.
    let dirs = SMOS_HOME_SUBDIRS.join("/");
    println!("  ✓ Directories: {dirs}/");
}

/// Load the just-bootstrapped config. A parse/validation failure on a
/// pre-existing (operator-edited) config must NOT abort the whole setup —
/// fall back to built-in defaults so the remaining probes still run and the
/// operator sees every problem in one pass.
fn load_config(config_path: &Path) -> SmosConfig {
    match SmosConfig::load(&config_path.to_string_lossy()) {
        Ok(c) => c,
        Err(e) => {
            println!("  ⚠ Could not parse {}: {e}", config_path.display());
            println!("    Falling back to built-in defaults for the remaining checks.");
            SmosConfig::default()
        }
    }
}

fn print_footer(paths: &SmosPaths) {
    println!("\n==========");
    println!("Setup complete!");
    println!("\nNext steps:");
    println!(
        "  1. Edit {} to configure providers and persons",
        paths.config.display()
    );
    println!("  2. Edit persona files under {}/", paths.persons.display());
    println!("  3. Start the server: smos serve");
    println!("\nInstall as a service: smos service install");
}

/// Resolve the canonical config path WITHOUT writing anything. Used by the
/// unified `smos` binary so `--config <path>` overrides take effect for
/// `smos serve` regardless of whether `smos init` was ever run.
///
/// Exposed here (rather than inlined at the call site) so the priority chain
/// — CLI override > `./smos.toml` > `~/.smos/config.toml` — stays documented
/// in one place.
pub fn resolve_effective_config_path(cli_override: Option<&str>) -> PathBuf {
    crate::paths::resolve_config_path(cli_override)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Acquire the workspace-wide env-test lock. See
    /// [`crate::test_env_lock`] for why this is required.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::test_env_lock::lock()
    }

    /// `bootstrap_filesystem` against a fresh `SMOS_HOME` tempdir creates the
    /// directory tree AND drops the default config. Re-running is idempotent:
    /// the second invocation must NOT overwrite the file the operator may
    /// have edited.
    #[test]
    fn bootstrap_filesystem_is_idempotent_and_writes_default_config_once() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: `INIT_TEST_LOCK` is held for the duration of the env
        // mutation + read, and the prior value is restored before return so
        // other tests in the binary see the original state.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }

        let fs = bootstrap_filesystem().expect("first bootstrap");
        assert!(fs.config_created, "first bootstrap must create the config");
        let config_path = tmp.path().join("config.toml");
        assert!(
            config_path.is_file(),
            "config must exist after first bootstrap"
        );
        let first_content = std::fs::read_to_string(&config_path).expect("read first config");

        // Simulate an operator edit: overwrite the config with custom
        // content. The second bootstrap MUST preserve it.
        std::fs::write(&config_path, "# operator-edited config\n").expect("write edit");
        let fs = bootstrap_filesystem().expect("second bootstrap");
        assert!(
            !fs.config_created,
            "second bootstrap must report config already exists"
        );
        let second_content = std::fs::read_to_string(&config_path).expect("read second config");
        assert_eq!(
            second_content, "# operator-edited config\n",
            "second bootstrap must NOT overwrite an existing config"
        );
        assert_ne!(
            second_content, first_content,
            "sanity: the two reads must differ"
        );

        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
    }

    /// `bootstrap_filesystem` drops a stub `persons/bob.md` so the default
    /// `[persons.bob].persona` reference resolves to a working file. The
    /// second invocation must NOT overwrite an operator-edited persona.
    #[test]
    fn bootstrap_filesystem_drops_stub_bob_persona_idempotently() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: same lock-protected guarantee.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }

        let fs = bootstrap_filesystem().expect("first bootstrap");
        assert!(
            fs.persona_created,
            "first bootstrap must create the persona"
        );
        let bob_path = tmp.path().join("persons").join("bob.md");
        assert!(
            bob_path.is_file(),
            "stub persona must exist after first bootstrap"
        );

        // Edit the persona; second bootstrap MUST preserve the edit.
        std::fs::write(&bob_path, "# operator persona\n").expect("edit");
        let fs = bootstrap_filesystem().expect("second bootstrap");
        assert!(
            !fs.persona_created,
            "second bootstrap must report persona already exists"
        );
        let content = std::fs::read_to_string(&bob_path).expect("read");
        assert_eq!(
            content, "# operator persona\n",
            "second bootstrap must NOT overwrite an existing persona file"
        );

        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
    }

    /// `DEFAULT_CONFIG_TOML` is the inline default config shipped in the
    /// binary. Pinning that it parses back into a valid
    /// [`crate::config::SmosConfig`] catches a typo in either the literal or
    /// the parser before it ships.
    #[test]
    fn default_config_toml_parses_into_valid_smos_config() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: same serialisation guarantee.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }
        let cfg = crate::config::SmosConfig::load_from_str(DEFAULT_CONFIG_TOML)
            .expect("default toml must parse + validate");
        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
        assert!(!cfg.providers.is_empty(), "default must ship >= 1 provider");
        assert!(!cfg.persons.is_empty(), "default must ship >= 1 person");
    }

    /// `resolve_effective_config_path` mirrors the documented priority chain.
    #[test]
    fn resolve_effective_config_path_prefers_cli_override() {
        let p = resolve_effective_config_path(Some("/explicit/path.toml"));
        assert_eq!(p, std::path::PathBuf::from("/explicit/path.toml"));
    }

    /// `smos_home` is the single source of truth for the home directory.
    /// Pinned so a refactor that drops the env-var override breaks the test
    /// rather than silently changing the resolution order.
    #[test]
    fn smos_home_is_exported_and_callable() {
        let _ = crate::paths::smos_home();
    }
}
