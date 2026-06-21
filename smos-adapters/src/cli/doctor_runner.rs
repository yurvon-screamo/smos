//! `smos doctor` — environment validation + stats + Markdown report.

use std::process::ExitCode;

use anyhow::{Context, Result};

use crate::cli::tracing_setup::init_tracing_default;
use crate::config::SmosConfig;
use crate::doctor::terminal::ColorMode;
use crate::doctor::{
    DoctorFlags, render_markdown, render_terminal, run_full_check, run_stats_only,
};

const DEFAULT_REPORT_PATH: &str = "smoke_report.md";

/// Parsed `smos doctor` invocation. The `smos` binary's clap parser
/// constructs this struct so the runner does not depend on clap.
pub struct DoctorArgs {
    pub stats: bool,
    pub report: Option<Option<String>>,
    pub skip_ollama: bool,
    pub color: String,
}

/// Entry point: load config, run checks (full or stats-only), render
/// terminal output, optionally write a Markdown report. Returns the
/// process exit code so `smos doctor` exits non-zero on any FAIL row.
pub async fn run_doctor(config_path: &str, args: DoctorArgs) -> Result<ExitCode> {
    init_tracing_default();

    let config = SmosConfig::load(config_path)
        .with_context(|| format!("failed to load config from {config_path}"))?;

    let report = if args.stats {
        run_stats_only(&config, config_path).await
    } else {
        let flags = DoctorFlags {
            skip_ollama: args.skip_ollama,
        };
        run_full_check(&config, &flags, config_path).await
    };

    let color = parse_color_mode(&args.color);
    let use_color = color.resolve(stdout_is_tty());

    let terminal = render_terminal(&report, use_color);
    println!("{terminal}");

    if let Some(report_path) = args.report {
        let path = report_path.unwrap_or_else(|| DEFAULT_REPORT_PATH.to_string());
        let markdown = render_markdown(&report);
        std::fs::write(&path, markdown)
            .with_context(|| format!("failed to write markdown report to {path}"))?;
        println!("\nMarkdown report written to {path}");
    }

    if report.summary().is_success() {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

/// Parse the `--color` flag value into a [`ColorMode`]. Unknown values
/// fall back to `Auto` so a typo never crashes the doctor — the operator
/// still sees plain-text output, which is strictly better than no output.
pub(crate) fn parse_color_mode(raw: &str) -> ColorMode {
    match raw.trim().to_ascii_lowercase().as_str() {
        "always" | "on" | "yes" | "1" => ColorMode::Always,
        "never" | "off" | "no" | "0" => ColorMode::Never,
        _ => ColorMode::Auto,
    }
}

/// True when stdout is a TTY. Mirrors `std::io::IsTerminal::is_terminal`
/// (stabilised in Rust 1.70) so the doctor does not need an external crate
/// (`atty`, `is-terminal`) for the check.
fn stdout_is_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_color_mode_known_aliases() {
        assert!(matches!(parse_color_mode("always"), ColorMode::Always));
        assert!(matches!(parse_color_mode("ON"), ColorMode::Always));
        assert!(matches!(parse_color_mode("never"), ColorMode::Never));
        assert!(matches!(parse_color_mode("off"), ColorMode::Never));
        assert!(matches!(parse_color_mode("auto"), ColorMode::Auto));
    }

    #[test]
    fn parse_color_mode_unknown_falls_back_to_auto() {
        assert!(matches!(parse_color_mode("rainbow"), ColorMode::Auto));
        assert!(matches!(parse_color_mode(""), ColorMode::Auto));
    }

    #[test]
    fn default_constants_match_spec() {
        assert_eq!(DEFAULT_REPORT_PATH, "smoke_report.md");
    }
}
