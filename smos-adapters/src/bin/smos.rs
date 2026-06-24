//! `smos` — unified SMOS binary.
//!
//! Dispatches via clap to the appropriate runner in `smos::cli`.
//! Every subcommand converts the parsed clap structs into the runner-specific
//! `*Args` structs so the runners stay clap-free and individually testable.
//!
//! ## Subcommands
//!
//! - `smos init` — one-command setup: materialise `~/.smos`, write the
//!   default `config.toml`, download the required GGUF models into
//!   `~/.smos/models/`, probe `llama-server` (PATH + `/health` on the
//!   three configured ports), run DB migrations, and report what still
//!   needs attention.
//! - `smos serve` — HTTP proxy server (proxy + watcher + native NLI).
//! - `smos import [<subcommand>]` — import data into SMOS memory. With no
//!   subcommand the opencode-transcript path runs (the historical
//!   `smos import <session_id>` form); `import directory`, `import git`,
//!   and `import raw` cover the other flavours.
//! - `smos import-dir <path>` / `smos import-git <url>` — deprecated
//!   aliases for the new subcommand form, kept for backward compatibility.
//! - `smos doctor` — environment validation, stats, Markdown report.
//! - `smos finalize` — manual single-session drain trigger.
//! - `smos service` — install/uninstall/start/stop/restart/status SMOS as
//!   a system or user service (Windows: sc.exe, Linux: systemd, macOS:
//!   launchd).
//! - `smos config` — read-only inspection of the resolved configuration
//!   (`show` full TOML, `providers`, `persons`).

use std::process::ExitCode;

use clap::{Parser, Subcommand};

use smos::cli::{
    AuditArgs, AuditProvider, ConfigAction, DoctorArgs, ImportArgs, ImportDirArgs, ImportGitArgs,
    RawImportArgs, ServiceAction, run_audit_cli, run_config, run_dir_import, run_doctor,
    run_finalize, run_import, run_import_git, run_init, run_raw_import, run_server, run_service,
};

#[derive(Parser, Debug)]
#[command(
    name = "smos",
    version,
    about = "SMOS — Semantic Memory OS",
    long_about = "Unified SMOS binary. Subcommands: init, serve, import, doctor, finalize, service, config."
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
    Init {
        /// Skip the live launch-and-verify step. The readiness probe still
        /// runs, but SMOS will not spawn any `llama-server` process — useful
        /// on hosts where `llama-server` is started by hand or is not yet
        /// on `PATH`.
        #[arg(long)]
        no_launch: bool,
    },

    /// Start the HTTP proxy server (proxy + watcher + native NLI).
    Serve,

    /// Import data into SMOS memory.
    ///
    /// With no subcommand the opencode-import path runs (the historical
    /// `smos import <session_id>` shape). Use `smos import <subcommand>`
    /// for the directory / git / raw flavours.
    Import {
        #[command(subcommand)]
        subcommand: Option<ImportSub>,

        /// Backward-compat carrier for the bare `smos import <session_id>`
        /// form. Parsed only when no subcommand is given.
        #[command(flatten)]
        opencode_args: OpencodeArgs,
    },

    /// Deprecated alias for `smos import directory <path>`. Kept so existing
    /// operator scripts and shell history keep working; new invocations
    /// should use the subcommand form.
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

    /// Deprecated alias for `smos import git <url>`. Kept so existing
    /// operator scripts and shell history keep working; new invocations
    /// should use the subcommand form.
    ImportGit {
        /// Git repository URL. Private repos use the system's SSH credentials.
        url: String,
    },

    /// Environment validation, stats, and Markdown report generator.
    Doctor {
        /// SurrealDB stats only. Skips the llama-server + binary checks.
        #[arg(long)]
        stats: bool,

        /// Write a Markdown report to <path>. Default `smoke_report.md`.
        /// Always runs after the terminal output regardless of `--stats`.
        /// Pass the flag without a value to use the default path.
        #[arg(long)]
        report: Option<Option<String>>,

        /// Skip the llama-server + reranker checks entirely.
        ///
        /// `--skip-ollama` is accepted as an alias for operator scripts and
        /// shell history written before the flag was renamed.
        #[arg(long = "skip-llama", alias = "skip-ollama")]
        skip_llama: bool,

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

    /// Inspect the resolved SMOS configuration (read-only).
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
}

/// `smos import <subcommand>` selector. Each variant matches one of the
/// historical top-level `import-*` commands so existing muscle memory
/// transfers (`import-dir` → `import directory`, `import-git` →
/// `import git`) while a new `raw` flavour is added for arbitrary text.
#[derive(Subcommand, Debug)]
enum ImportSub {
    /// Import an opencode session transcript (`smos import opencode …`).
    Opencode(OpencodeArgs),

    /// Bulk import facts from a directory tree (`smos import directory …`).
    Directory {
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

    /// Re-hydrate facts from a git-synced memory repo
    /// (`smos import git <url>`).
    Git {
        /// Git repository URL. Private repos use the system's SSH credentials.
        url: String,
    },

    /// Extract facts from arbitrary free-form text
    /// (`smos import raw "<text>"`).
    Raw {
        /// Text to extract facts from. Read from stdin when omitted (and
        /// `--stdin` is passed).
        text: Option<String>,

        /// Read the input text from stdin instead of the positional arg.
        #[arg(long)]
        stdin: bool,

        /// Memory namespace (project key). Defaults to the shared namespace.
        #[arg(long, default_value = "shared")]
        memory_key: String,
    },
}

/// Flat args for the opencode import flavour. Reused by the bare
/// `smos import <session_id>` form (no subcommand) and the explicit
/// `smos import opencode …` form so the two paths stay DRY.
#[derive(clap::Args, Debug)]
struct OpencodeArgs {
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
        Command::Init { no_launch } => {
            run_init(no_launch).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Serve => {
            run_server(&config_path).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Import {
            subcommand,
            opencode_args,
        } => {
            match subcommand {
                None => {
                    let args = opencode_args_to_import_args(opencode_args);
                    run_import(&config_path, args).await?;
                }
                Some(ImportSub::Opencode(oa)) => {
                    let args = opencode_args_to_import_args(oa);
                    run_import(&config_path, args).await?;
                }
                Some(ImportSub::Directory {
                    path,
                    memory_key,
                    limit,
                    no_finalize,
                }) => {
                    let args = ImportDirArgs {
                        path,
                        memory_key,
                        limit,
                        no_finalize,
                    };
                    run_dir_import(&config_path, args).await?;
                }
                Some(ImportSub::Git { url }) => {
                    let args = ImportGitArgs { url };
                    run_import_git(&config_path, args).await?;
                }
                Some(ImportSub::Raw {
                    text,
                    stdin,
                    memory_key,
                }) => {
                    let body = read_raw_text(text, stdin)?;
                    let args = RawImportArgs {
                        text: body,
                        memory_key,
                    };
                    run_raw_import(&config_path, args).await?;
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Doctor {
            stats,
            report,
            skip_llama,
            color,
        } => {
            let args = DoctorArgs {
                stats,
                report,
                skip_llama,
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
        Command::Config { action } => {
            run_config(&config_path, action).await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Convert the clap-parsed [`OpencodeArgs`] into the runner-side
/// [`ImportArgs`] struct. Keeps the runner clap-free.
fn opencode_args_to_import_args(oa: OpencodeArgs) -> ImportArgs {
    ImportArgs {
        session_id: oa.session_id,
        from_file: oa.from_file,
        memory_key: oa.memory_key,
        port: oa.port,
        agents: oa.agents,
        limit: oa.limit,
        offset: oa.offset,
        dry_run: oa.dry_run,
        list: oa.list,
    }
}

/// Resolve the raw-import text body. When `--stdin` is given the body is
/// read from stdin (positional `text` must be `None`); otherwise the
/// positional argument is used verbatim. An empty body is rejected so the
/// extraction pipeline does not run on a no-op input.
fn read_raw_text(text: Option<String>, stdin: bool) -> anyhow::Result<String> {
    let body = if stdin {
        use std::io::Read;
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    } else {
        text.ok_or_else(|| {
            anyhow::anyhow!(
                "no input text: pass a positional argument or use --stdin to read from stdin"
            )
        })?
    };
    if body.trim().is_empty() {
        anyhow::bail!("raw import text is empty; nothing to extract");
    }
    Ok(body)
}
