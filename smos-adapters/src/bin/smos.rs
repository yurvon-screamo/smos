//! `smos` — unified SMOS binary.
//!
//! Dispatches via clap to the appropriate runner in `smos_adapters::cli`.
//! Every subcommand converts the parsed clap structs into the runner-specific
//! `*Args` structs so the runners stay clap-free and individually testable.
//!
//! ## Subcommands
//!
//! - `smos serve` — HTTP proxy server (proxy + watcher + native NLI).
//! - `smos import` — import an opencode session transcript.
//! - `smos doctor` — environment validation, stats, Markdown report.
//! - `smos finalize` — manual single-session drain trigger.
//! - `smos service` — install/uninstall/start/stop/restart/status SMOS as
//!   a system or user service (Windows: sc.exe, Linux: systemd, macOS:
//!   launchd).

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use smos_adapters::cli::{
    AuditArgs, AuditProvider, DoctorArgs, ImportArgs, ServiceAction, run_audit_cli, run_doctor,
    run_finalize, run_import, run_server, run_service,
};

const DEFAULT_CONFIG_PATH: &str = "smos.toml";

#[derive(Parser, Debug)]
#[command(
    name = "smos",
    version,
    about = "SMOS — Semantic Memory OS",
    long_about = "Unified SMOS binary. Subcommands: serve, import, doctor, finalize, service."
)]
struct Cli {
    /// Path to the config file. Defaults to `smos.toml` in the CWD.
    #[arg(long, global = true, default_value = DEFAULT_CONFIG_PATH)]
    config: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Start the HTTP proxy server (proxy + watcher + native NLI).
    Serve,

    /// Import an opencode session transcript into SMOS memory.
    Import {
        /// opencode session id (e.g. `ses_abc123`). Required unless `--list`
        /// or `--from-file` is given.
        #[arg(required_unless_present_any = ["list", "from_file"])]
        session_id: Option<String>,

        /// Import from a local opencode-export JSON file instead of discovery.
        #[arg(long, conflicts_with = "session_id")]
        from_file: Option<String>,

        /// Memory namespace (project key). Defaults to the shared namespace.
        #[arg(long, default_value = "shared")]
        memory_key: String,

        /// Override the opencode server port (skips auto-discovery).
        #[arg(long)]
        port: Option<u16>,

        /// Restrict the import to turns emitted by these agents (repeatable).
        #[arg(long = "agent")]
        agents: Vec<String>,

        /// Take only the first N turns after `--offset` (smoke testing).
        #[arg(long)]
        limit: Option<usize>,

        /// Skip the first N turns before applying `--limit`.
        #[arg(long, default_value = "0")]
        offset: usize,

        /// Parse the transcript and print the turns, do NOT call models or save.
        #[arg(long)]
        dry_run: bool,

        /// List discovered sessions and exit.
        #[arg(long)]
        list: bool,
    },

    /// Environment validation, stats, and Markdown report generator.
    Doctor {
        /// SurrealDB stats only. Skips Ollama + binary checks.
        #[arg(long)]
        stats: bool,

        /// Write a Markdown report to <path>. Default `smoke_report.md`.
        /// Always runs after the terminal output regardless of `--stats`.
        /// Pass the flag without a value to use the default path.
        #[arg(long)]
        report: Option<Option<String>>,

        /// Skip the Ollama + reranker checks entirely.
        #[arg(long)]
        skip_ollama: bool,

        /// Force color on (`always`), off (`never`), or auto-detect (`auto`).
        #[arg(long, default_value = "auto")]
        color: String,
    },

    /// Trigger a manual single-session finalize (NLI drain).
    Finalize {
        /// Session id to finalize (e.g. `sess_<12 hex chars>`).
        session_id: String,

        /// Memory namespace (project key). When omitted, the runner falls
        /// back to the cross-namespace discovery scan (slower but works
        /// when the operator does not know the namespace off-hand).
        #[arg(long)]
        memory_key: Option<String>,
    },

    /// Manage SMOS as a system service (install/uninstall/start/stop/status).
    Service {
        #[command(subcommand)]
        action: ServiceAction,
    },

    /// Run the SMOS Dreaming Agent audit once in the foreground.
    Audit {
        /// Override the configured LLM provider (`cloud` | `local`).
        #[arg(long)]
        provider: Option<String>,

        /// Dry-run: validate provider configuration and bail out before
        /// loading the NLI / embedder models.
        #[arg(long)]
        dry_run: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve => {
            run_server(&cli.config).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Import {
            session_id,
            from_file,
            memory_key,
            port,
            agents,
            limit,
            offset,
            dry_run,
            list,
        } => {
            let args = ImportArgs {
                session_id,
                from_file,
                memory_key,
                port,
                agents,
                limit,
                offset,
                dry_run,
                list,
            };
            run_import(&cli.config, args).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Doctor {
            stats,
            report,
            skip_ollama,
            color,
        } => {
            let args = DoctorArgs {
                stats,
                report,
                skip_ollama,
                color,
            };
            run_doctor(&cli.config, args).await
        }
        Command::Finalize {
            session_id,
            memory_key,
        } => {
            run_finalize(&cli.config, &session_id, memory_key.as_deref()).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Service { action } => {
            run_service(action, &cli.config).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Audit { provider, dry_run } => {
            let provider = match provider.as_deref() {
                Some(s) => Some(AuditProvider::parse(s)?),
                None => None,
            };
            let args = AuditArgs { provider, dry_run };
            run_audit_cli(&cli.config, args).await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}
