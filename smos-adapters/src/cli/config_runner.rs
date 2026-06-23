//! `smos config` — read-only inspection of the resolved configuration.
//!
//! Loads the config through the same resolution chain as every other
//! subcommand (`--config` → `./smos.toml` → `~/.smos/config.toml`) and
//! prints either the full TOML or a focused summary. Nothing here mutates
//! the config on disk — `smos` deliberately has no `config add/remove`:
//! editing the TOML by hand is the canonical workflow, and a read-only CLI
//! avoids the two-sources-of-truth problem.

use std::collections::HashMap;

use anyhow::{Context, Result};
use clap::Subcommand;

use crate::cli::tracing_setup::init_tracing_default;
use crate::config::SmosConfig;

/// Subcommand payload parsed by clap. The `smos` binary forwards the parsed
/// [`ConfigAction`] directly to [`run_config`].
#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Print the full resolved configuration as TOML (defaults merged in).
    Show,
    /// List configured providers (name → URL).
    Providers,
    /// List configured persons / agents (name → provider / model).
    Persons,
}

/// Entry point: load config, print the requested view.
pub async fn run_config(config_path: &str, action: ConfigAction) -> Result<()> {
    init_tracing_default();

    let config = SmosConfig::load(config_path)
        .with_context(|| format!("failed to load config from {config_path}"))?;

    match action {
        ConfigAction::Show => print_full_toml(&config)?,
        ConfigAction::Providers => print_providers(&config.providers),
        ConfigAction::Persons => print_persons(&config.persons),
    }
    Ok(())
}

fn print_full_toml(config: &SmosConfig) -> Result<()> {
    let toml = toml::to_string_pretty(config).context("failed to serialise config to TOML")?;
    println!("{toml}");
    Ok(())
}

fn print_providers(providers: &[crate::config::ProviderConfig]) {
    if providers.is_empty() {
        println!("(no [[providers]] configured)");
        return;
    }
    let name_width = providers.iter().map(|p| p.name.len()).max().unwrap_or(0);
    for p in providers {
        println!("{:<width$}  →  {}", p.name, p.url, width = name_width);
    }
}

fn print_persons(persons: &HashMap<String, crate::config::PersonConfig>) {
    if persons.is_empty() {
        println!("(no [persons.*] configured)");
        return;
    }
    let mut names: Vec<&String> = persons.keys().collect();
    names.sort();
    let name_width = names.iter().map(|n| n.len()).max().unwrap_or(0);
    for name in names {
        let p = &persons[name];
        println!(
            "{:<width$}  →  {} / {}",
            name,
            p.provider,
            p.model,
            width = name_width
        );
    }
}
