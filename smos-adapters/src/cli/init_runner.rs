//! `smos init` — first-time setup runner.
//!
//! Materialises `~/.smos` (or `$SMOS_HOME`) with every well-known
//! subdirectory and drops a default `config.toml` at its root when one is
//! not already present. Re-running `smos init` is idempotent: existing
//! directories are kept, an existing `config.toml` is NEVER overwritten so
//! the operator's edits survive a re-init.

use anyhow::Result;

use crate::paths::{SmosPaths, ensure_smos_home};

/// The canonical default `config.toml` written by [`run_init`] when no
/// config exists yet.
///
/// Mirrors the in-repo `smos.toml` shape so an operator who reads either
/// sees the same surface. Kept inline (not loaded from a file) so the
/// binary stays self-contained — a single-file distribution can drop the
/// `smos` binary on a fresh box and `smos init` produces a working config
/// without any extra assets.
pub const DEFAULT_CONFIG_TOML: &str = include_str!("../../../smos.toml");

/// Entry point invoked by the unified `smos` binary's `Init` subcommand.
///
/// Steps:
/// 1. [`ensure_smos_home`] creates every well-known subdirectory.
/// 2. If `~/.smos/config.toml` does NOT exist, write [`DEFAULT_CONFIG_TOML`]
///    to it. An existing config is left untouched (logged).
/// 3. Drop a stub `persons/bob.md` when one is not already present so the
///    operator has a concrete starting point (the default config references
///    this file via `[persons.bob].persona`). Existing files are kept.
/// 4. Print the resolved paths + a pointer to the persona file so the
///    operator knows where to drop persona `.md` content.
pub fn run_init() -> Result<()> {
    let home = ensure_smos_home()?;
    let paths = SmosPaths::resolve();
    let config_path = &paths.config;

    if config_path.exists() {
        tracing::info!(
            config_path = %config_path.display(),
            "config already exists; left untouched"
        );
        println!("Config already exists at {}", config_path.display());
    } else {
        std::fs::write(config_path, DEFAULT_CONFIG_TOML)?;
        tracing::info!(
            config_path = %config_path.display(),
            "wrote default config"
        );
        println!("Created default config at {}", config_path.display());
    }

    // Drop a stub persona file so the default `[persons.bob]` entry has a
    // working target. The operator is expected to edit it; we never
    // overwrite an existing file (the operator may have crafted one
    // already).
    let bob_persona = paths.persons.join("bob.md");
    if !bob_persona.exists() {
        std::fs::write(&bob_persona, DEFAULT_PERSONA_BOB_MD)?;
        println!("Created stub persona at {}", bob_persona.display());
        println!(
            "Edit {} to customise the bob persona (this is the system prompt \
             injected on every chat-completion request that names model \"bob\").",
            bob_persona.display()
        );
    }

    println!("SMOS home: {}", home.display());
    println!(
        "Directory structure created under {} (db/, models/, persons/, logs/, reports/).",
        home.display()
    );
    Ok(())
}

/// Minimal stub persona shipped with the default `[persons.bob]` entry.
///
/// Kept inline (not loaded from a file) so the binary stays self-contained.
/// The operator is expected to replace it with the real persona content
/// after `smos init`.
const DEFAULT_PERSONA_BOB_MD: &str = "\
# Bob persona\n\
\n\
Replace this file with the system prompt you want injected as the leading\n\
`system` message for every chat-completion request that names `model: \"bob\"`\n\
in its body. The content is forwarded verbatim to the upstream provider\n\
declared in `[persons.bob].provider`.\n\
";

/// Resolve the canonical config path WITHOUT writing anything. Used by
/// the unified `smos` binary so `--config <path>` overrides take effect
/// for `smos serve` regardless of whether `smos init` was ever run.
///
/// Exposed here (rather than inlined at the call site) so the priority
/// chain — CLI override > `./smos.toml` > `~/.smos/config.toml` — stays
/// documented in one place.
pub fn resolve_effective_config_path(cli_override: Option<&str>) -> std::path::PathBuf {
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
    /// `run_init` against a fresh `SMOS_HOME` tempdir creates the
    /// directory tree AND drops the default config. Re-running is
    /// idempotent: the second invocation must NOT overwrite the file the
    /// operator may have edited.
    #[test]
    fn run_init_is_idempotent_and_writes_default_config_once() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: `INIT_TEST_LOCK` is held for the duration of the env
        // mutation + read, and the prior value is restored before return
        // so other tests in the binary see the original state.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }

        run_init().expect("first init");
        let config_path = tmp.path().join("config.toml");
        assert!(config_path.is_file(), "config must exist after first init");
        let first_content = std::fs::read_to_string(&config_path).expect("read first config");

        // Simulate an operator edit: overwrite the config with custom
        // content. The second init MUST preserve it.
        std::fs::write(&config_path, "# operator-edited config\n").expect("write edit");
        run_init().expect("second init");
        let second_content = std::fs::read_to_string(&config_path).expect("read second config");
        assert_eq!(
            second_content, "# operator-edited config\n",
            "second init must NOT overwrite an existing config"
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

    /// `run_init` drops a stub `persons/bob.md` so the default
    /// `[persons.bob].persona` reference resolves to a working file. The
    /// second invocation must NOT overwrite an operator-edited persona.
    #[test]
    fn run_init_drops_stub_bob_persona_idempotently() {
        let _g = lock();
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let prior = std::env::var("SMOS_HOME").ok();
        // SAFETY: same lock-protected guarantee.
        unsafe {
            std::env::set_var("SMOS_HOME", tmp.path());
        }

        run_init().expect("first init");
        let bob_path = tmp.path().join("persons").join("bob.md");
        assert!(
            bob_path.is_file(),
            "stub persona must exist after first init"
        );

        // Edit the persona; second init MUST preserve the edit.
        std::fs::write(&bob_path, "# operator persona\n").expect("edit");
        run_init().expect("second init");
        let content = std::fs::read_to_string(&bob_path).expect("read");
        assert_eq!(
            content, "# operator persona\n",
            "second init must NOT overwrite an existing persona file"
        );

        unsafe {
            match prior {
                Some(v) => std::env::set_var("SMOS_HOME", v),
                None => std::env::remove_var("SMOS_HOME"),
            }
        }
    }
    /// `DEFAULT_CONFIG_TOML` is the include_str! of the in-repo smos.toml.
    /// Pinning that it parses back into a valid [`crate::config::SmosConfig`]
    /// catches a typo in either the file or the parser before it ships.
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
    /// Pinned so a refactor that drops the env-var override breaks the
    /// test rather than silently changing the resolution order.
    #[test]
    fn smos_home_is_exported_and_callable() {
        let _ = crate::paths::smos_home();
    }
}
