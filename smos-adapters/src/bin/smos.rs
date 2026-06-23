//! `smos` — unified SMOS binary.
//!
//! Dispatches via clap to the appropriate runner in `smos::cli`.
//! Every subcommand converts the parsed clap structs into the runner-specific
//! `*Args` structs so the runners stay clap-free and individually testable.
//!
//! ## Subcommands
//!
//! - `smos init` — one-command setup: materialise `~/.smos`, write the
//!   default `config.toml`, probe Ollama / llama-server / reranker, run DB
//!   migrations, and report what still needs attention.
//! - `smos serve` — HTTP proxy server (proxy + watcher + native NLI).
//! - `smos import` — import an opencode session transcript.
//! - `smos import-dir` — bulk import facts from a directory tree
//!   (`*.md`, `*.txt`, `*.json`, `*.jsonl`, `*.yaml`, `*.yml`, `*.toml`).
//! - `smos doctor` — environment validation, stats, Markdown report.
//! - `smos finalize` — manual single-session drain trigger.
//! - `smos service` — install/uninstall/start/stop/restart/status SMOS as
//!   a system or user service (Windows: sc.exe, Linux: systemd, macOS:
//!   launchd).

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use smos::cli::{
    AuditArgs, AuditProvider, DoctorArgs, ImportArgs, ImportDirArgs, ImportGitArgs, ServiceAction,
    run_audit_cli, run_dir_import, run_doctor, run_finalize, run_import, run_import_git, run_init,
    run_server, run_service,
};

#[derive(Parser, Debug)]
#[command(
    name = "smos",
    version,
    about = "SMOS — Semantic Memory OS",
    long_about = "Unified SMOS binary. Subcommands: init, serve, import, doctor, finalize, service."
)]
struct Cli {
    /// Path to the config file. When omitted, the proxy resolves
    /// `./smos.toml` (CWD) first, then `~/.smos/config.toml`.
    #[arg(long, global = true)]
    config: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Initialize SMOS home directory (~/.smos) with default config.
    Init,

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

    /// Import facts from a git repo previously written by SMOS git-sync.
    ///
    /// Clones (or opens) `url` into a temporary local path, reads every
    /// `facts/<memory_key>/<id>.md` file, and re-hydrates the facts into
    /// SurrealDB. The clone is left on disk so a subsequent invocation
    /// can re-use it (incremental pull).
    ImportGit {
        /// Git repository URL. Private repos use the system's SSH credentials.
        url: String,
    },

    /// Bulk import facts from a directory tree.
    ///
    /// Scans `path` recursively for `*.md`, `*.txt`, `*.json`, `*.jsonl`,
    /// `*.yaml`, `*.yml`, `*.toml` (hidden directories like `.git` are
    /// skipped), feeds each file's content through the extraction
    /// pipeline (Qwen → facts), and optionally triggers a single
    /// finalize drain (NLI) at the end.
    ImportDir {
        /// Directory to scan for documents.
        path: String,

        /// Memory namespace (project key). Defaults to the shared namespace.
        #[arg(long, default_value = "shared")]
        memory_key: String,

        /// Limit number of files to process (smoke testing).
        #[arg(long)]
        limit: Option<usize>,

        /// Skip the NLI finalize drain after the import.
        #[arg(long)]
        no_finalize: bool,
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

    // Resolve the effective config path once. `resolve_config_path` already
    // implements the full fallback chain (`--config` > `./smos.toml` >
    // `~/.smos/config.toml`), so we just hand it the CLI override (if any)
    // and stringify the result. `smos init` ignores this path because it
    // bootstraps `~/.smos/config.toml` itself.
    let config_path: String = match cli.config.as_deref() {
        Some(p) if !p.is_empty() => p.to_string(),
        _ => smos::cli::resolve_effective_config_path(cli.config.as_deref())
            .to_string_lossy()
            .into_owned(),
    };

    match cli.command {
        Command::Init => {
            run_init().await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Serve => {
            run_server(&config_path).await?;
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
            run_import(&config_path, args).await?;
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
            run_doctor(&config_path, args).await
        }
        Command::Finalize {
            session_id,
            memory_key,
        } => {
            run_finalize(&config_path, &session_id, memory_key.as_deref()).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::ImportGit { url } => {
            let args = ImportGitArgs { url };
            run_import_git(&config_path, args).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::ImportDir {
            path,
            memory_key,
            limit,
            no_finalize,
        } => {
            let args = ImportDirArgs {
                path,
                memory_key,
                limit,
                no_finalize,
            };
            run_dir_import(&config_path, args).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Service { action } => {
            run_service(action, &config_path).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Audit { provider, dry_run } => {
            let provider = match provider.as_deref() {
                Some(s) => Some(AuditProvider::parse(s)?),
                None => None,
            };
            let args = AuditArgs { provider, dry_run };
            run_audit_cli(&config_path, args).await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}
