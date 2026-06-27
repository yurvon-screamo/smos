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
    OutputFormat, RawImportArgs, SearchArgs, ServiceAction, run_audit_cli, run_config,
    run_dir_import, run_doctor, run_finalize, run_import, run_import_git, run_init, run_raw_import,
    run_search, run_server, run_service,
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

    /// Read-only retrieval: print reranked accepted facts for a query as JSON.
    ///
    /// Requires a reachable embedding endpoint (`[embedding]`) and reranker
    /// (`[reranker]`), and that the DB already holds Accepted facts (run
    /// `smos import raw` then `smos finalize` to populate it). Strictly
    /// sequential per database path: SurrealDB/RocksDB takes a single-writer
    /// lock at connect time.
    Search {
        /// Query text. Read from stdin when `--stdin` is passed instead.
        query: Option<String>,

        /// Read the query from stdin instead of the positional argument.
        #[arg(long)]
        stdin: bool,

        /// Memory namespace (project key) to search within. Required.
        #[arg(long)]
        person: String,

        /// Override `retrieval.top_k_final` (the rerank depth).
        #[arg(long)]
        top_k: Option<usize>,

        /// Output format. Only `json` today; reserved for future formats.
        #[arg(long, default_value = "json")]
        format: String,
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
    /// opencode session id (e.g. `ses_abc123`). Declared optional at the
    /// clap level on purpose: a `required`-style attribute here is checked
    /// even when the sibling `subcommand` of [`Command::Import`] is present,
    /// which would falsely reject `import raw`, `import directory`, and
    /// `import git`. The "required unless --list / --from-file" contract is
    /// enforced at runtime by [`validate_opencode_args`].
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

fn main() -> anyhow::Result<ExitCode> {
    // Service short-circuit: when the SCM launches `smos.exe` with
    // `--run-as-service` (the binPath injected at install time), hand
    // control to the SCM dispatcher BEFORE the tokio runtime is built.
    // `service_dispatcher::start` blocks the calling thread for the whole
    // service lifetime, so the runtime has to be created INSIDE
    // `ServiceMain` on the SCM worker thread — a tokio runtime already
    // living on the main thread (e.g. from `#[tokio::main]`) would
    // collide with that nested runtime, hence the fully synchronous
    // `main` and the early service return.
    #[cfg(target_os = "windows")]
    if is_service_launch() {
        return run_service_process();
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(run_cli())
}

#[cfg(target_os = "windows")]
fn is_service_launch() -> bool {
    std::env::args()
        .nth(1)
        .as_deref()
        .map(|a| a == smos::cli::service_runner::SERVICE_RUN_FLAG)
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn run_service_process() -> anyhow::Result<ExitCode> {
    smos::cli::service_runner::run_as_service()?;
    Ok(ExitCode::SUCCESS)
}

/// Build the tokio runtime for the CLI path and drive the clap dispatch.
/// Kept separate from [`main`] so the service branch can avoid paying
/// for a runtime it cannot use (the dispatcher owns the main thread).
async fn run_cli() -> anyhow::Result<ExitCode> {
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
                    validate_opencode_args(&opencode_args)?;
                    let args = opencode_args_to_import_args(opencode_args);
                    run_import(&config_path, args).await?;
                }
                Some(ImportSub::Opencode(oa)) => {
                    validate_opencode_args(&oa)?;
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
        Command::Search {
            query,
            stdin,
            person,
            top_k,
            format,
        } => {
            let body = read_search_query(query, stdin)?;
            let fmt = OutputFormat::parse(&format)?;
            let args = SearchArgs {
                query: body,
                memory_key: person,
                top_k,
                format: fmt,
            };
            run_search(&config_path, args).await?;
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

/// Enforce the historical "session_id required unless `--list` /
/// `--from-file`" contract for the opencode import flavours at runtime.
///
/// This rule used to live on the `session_id` field as a clap
/// `required_unless_present_any`, but clap evaluates that attribute even when
/// a sibling `subcommand` of [`Command::Import`] is present — so `smos import
/// raw`, `import directory`, and `import git` were all falsely rejected with
/// a missing `<SESSION_ID>` error. Moving the check here leaves the
/// subcommand shapes free of the phantom positional while preserving the
/// bare `smos import <session_id>` and explicit `smos import opencode
/// <session_id>` contracts. The subcommand flavours (raw / directory / git)
/// never reach this function.
///
/// Mirrors the defence-in-depth guard in
/// `smos::cli::import_runner::resolve_transcript`; the two layers
/// intentionally carry different messages — this one for a clap-equivalent
/// CLI UX at the dispatch site, the runner's for direct (clap-free) callers
/// of `run_import`.
fn validate_opencode_args(oa: &OpencodeArgs) -> anyhow::Result<()> {
    if oa.session_id.is_none() && oa.from_file.is_none() && !oa.list {
        anyhow::bail!(
            "the following required arguments were not provided:\n  <SESSION_ID>\n\
             \nUsage: smos import <SESSION_ID>\n\
             \nPass a session id, or use --list / --from-file, or pick a subcommand \
             (raw | directory | git | opencode)."
        );
    }
    Ok(())
}

/// Resolve the `smos search` query body. When `--stdin` is given the body is
/// read from stdin (positional `query` must be `None`); otherwise the
/// positional argument is used verbatim. Unlike [`read_raw_text`], an empty
/// body is permitted — the search use case returns an empty array for it, so
/// a blank query is a valid `[]`-producing invocation rather than an error.
fn read_search_query(query: Option<String>, stdin: bool) -> anyhow::Result<String> {
    let body = if stdin {
        use std::io::Read;
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    } else {
        query.ok_or_else(|| {
            anyhow::anyhow!(
                "no query: pass a positional argument or use --stdin to read from stdin"
            )
        })?
    };
    // Trim only trailing newlines so multi-line queries keep their internal
    // shape; the use case applies its own `trim()` for the min-topic-chars gate.
    Ok(body.trim_end_matches(['\n', '\r']).to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> Cli {
        let mut full = vec!["smos"];
        full.extend_from_slice(args);
        Cli::try_parse_from(full).expect("CLI should parse")
    }

    fn parse_err(args: &[&str]) -> String {
        let mut full = vec!["smos"];
        full.extend_from_slice(args);
        match Cli::try_parse_from(full) {
            Ok(_) => panic!("expected parse error for {:?}", args),
            Err(e) => e.to_string(),
        }
    }

    fn into_bare_opencode(cli: Cli) -> (Option<ImportSub>, OpencodeArgs) {
        match cli.command {
            Command::Import {
                subcommand,
                opencode_args,
            } => (subcommand, opencode_args),
            other => panic!("expected Command::Import, got {other:?}"),
        }
    }

    #[test]
    fn import_raw_positional_parses_without_session_id() {
        // Regression: the flatten'd OpencodeArgs.session_id used to carry a
        // clap `required_unless_present_any`, which clap checked even with a
        // sibling subcommand present — so this rejected with a missing
        // <SESSION_ID> error.
        let cli = parse(&["import", "raw", "I like Python."]);
        match into_bare_opencode(cli) {
            (
                Some(ImportSub::Raw {
                    text,
                    stdin,
                    memory_key,
                }),
                _,
            ) => {
                assert_eq!(text.as_deref(), Some("I like Python."));
                assert!(!stdin);
                assert_eq!(memory_key, "shared");
            }
            (other, _) => panic!("expected Raw subcommand, got {other:?}"),
        }
    }

    #[test]
    fn import_raw_stdin_parses_without_session_id() {
        // The exact shape the BEAM adapter invokes for every chunk.
        let cli = parse(&["import", "raw", "--stdin", "--memory-key", "diagtest"]);
        match into_bare_opencode(cli) {
            (
                Some(ImportSub::Raw {
                    text,
                    stdin,
                    memory_key,
                }),
                _,
            ) => {
                assert!(text.is_none());
                assert!(stdin);
                assert_eq!(memory_key, "diagtest");
            }
            (other, _) => panic!("expected Raw subcommand, got {other:?}"),
        }
    }

    #[test]
    fn import_directory_parses_without_session_id() {
        let cli = parse(&["import", "directory", "/tmp/docs"]);
        assert!(matches!(
            into_bare_opencode(cli),
            (Some(ImportSub::Directory { .. }), _)
        ));
    }

    #[test]
    fn import_git_parses_without_session_id() {
        let cli = parse(&["import", "git", "https://example.invalid/x.git"]);
        assert!(matches!(
            into_bare_opencode(cli),
            (Some(ImportSub::Git { .. }), _)
        ));
    }

    #[test]
    fn bare_import_without_session_id_fails_runtime_validation() {
        // `smos import` with no subcommand and no session_id still must be
        // rejected — now at runtime by validate_opencode_args instead of by
        // clap, preserving the historical contract. The message keeps the
        // clap-style body so operators recognise it; the exit code moves
        // from clap's 2 to anyhow's 1 (accepted per task spec).
        let (subcommand, oa) = into_bare_opencode(parse(&["import"]));
        assert!(subcommand.is_none());
        let msg = validate_opencode_args(&oa).unwrap_err().to_string();
        assert!(
            msg.contains("SESSION_ID"),
            "error should mention SESSION_ID: {msg}"
        );
        assert!(
            msg.contains("required arguments were not provided"),
            "error should keep the clap-style body: {msg}"
        );
    }

    #[test]
    fn explicit_opencode_without_session_id_fails_runtime_validation() {
        // The explicit `smos import opencode` flavour shares OpencodeArgs,
        // so the same runtime contract applies.
        let cli = parse(&["import", "opencode"]);
        match cli.command {
            Command::Import {
                subcommand: Some(ImportSub::Opencode(oa)),
                ..
            } => {
                let msg = validate_opencode_args(&oa).unwrap_err().to_string();
                assert!(msg.contains("SESSION_ID"), "{msg}");
                assert!(
                    msg.contains("required arguments were not provided"),
                    "{msg}"
                );
            }
            other => panic!("expected explicit Import opencode, got {other:?}"),
        }
    }

    #[test]
    fn explicit_opencode_with_session_id_passes_validation() {
        // Positive mirror of the explicit-opencode failure case: locks the
        // dispatch success path for `Some(ImportSub::Opencode(oa))`.
        let cli = parse(&["import", "opencode", "ses_abc"]);
        match cli.command {
            Command::Import {
                subcommand: Some(ImportSub::Opencode(oa)),
                ..
            } => {
                assert_eq!(oa.session_id.as_deref(), Some("ses_abc"));
                validate_opencode_args(&oa).expect("explicit opencode + session_id => ok");
            }
            other => panic!("expected explicit Import opencode, got {other:?}"),
        }
    }

    #[test]
    fn bare_import_with_session_id_passes_validation() {
        let (subcommand, oa) = into_bare_opencode(parse(&["import", "ses_abc"]));
        assert!(subcommand.is_none());
        assert_eq!(oa.session_id.as_deref(), Some("ses_abc"));
        validate_opencode_args(&oa).expect("session id present => ok");
    }

    #[test]
    fn bare_import_with_list_passes_validation() {
        let (subcommand, oa) = into_bare_opencode(parse(&["import", "--list"]));
        assert!(subcommand.is_none());
        assert!(oa.list);
        validate_opencode_args(&oa).expect("--list => ok");
    }

    #[test]
    fn bare_import_with_from_file_passes_validation() {
        let (subcommand, oa) = into_bare_opencode(parse(&["import", "--from-file", "x.json"]));
        assert!(subcommand.is_none());
        assert_eq!(oa.from_file.as_deref(), Some("x.json"));
        validate_opencode_args(&oa).expect("--from-file => ok");
    }

    #[test]
    fn import_help_still_lists_subcommands_and_session_id() {
        // The Usage block of `import --help` must still surface the four
        // subcommands and the positional SESSION_ID (now optional, but
        // present) so operator muscle memory keeps working.
        let help = parse_err(&["import", "--help"]);
        for token in ["raw", "directory", "git", "opencode", "SESSION_ID"] {
            assert!(
                help.contains(token),
                "import --help must mention {token:?}:\n{help}"
            );
        }
    }
}
