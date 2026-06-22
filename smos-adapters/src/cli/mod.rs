//! `cli` — subcommand runners shared by the unified `smos` binary.
//!
//! Each runner is the body of one subcommand exposed as a callable async
//! function so the single `smos` binary can dispatch to it via clap. The
//! runner does NOT parse CLI args itself — the `smos` binary's clap
//! parser converts `Cli` into the runner-specific `*Args` struct so the
//! runner stays clap-free and the surface stays testable.
//!
//! Layout:
//! - [`tracing_setup`] — install the tracing subscriber (shared by every
//!   subcommand).
//! - [`shutdown`] — Ctrl+C / SIGTERM future (server-only).
//! - [`init_runner`] — `smos init` (first-time `~/.smos` setup).
//! - [`server_runner`] — `smos serve` (proxy server).
//! - [`llama_runner`] — `llama-server` auto-launch helper used by
//!   [`server_runner`].
//! - [`finalize_runner`] — `smos finalize` (single-session drain trigger).
//! - [`import_runner`] — `smos import` (opencode transcript importer) +
//!   [`import_helpers`] (pure helpers + unit tests).
//! - [`doctor_runner`] — `smos doctor` (environment validation + report).
//! - [`service`] — `smos service` (cross-platform service management via
//!   sc.exe / systemd / launchd).

pub mod audit_runner;
pub mod doctor_runner;
pub mod finalize_runner;
pub mod git_import_runner;
pub mod import_helpers;
pub mod import_runner;
pub mod init_runner;
pub mod llama_runner;
pub mod server_runner;
pub mod service;
pub mod shutdown;
pub mod tracing_setup;

pub use audit_runner::{AuditArgs, AuditProvider, run_audit_cli};
pub use doctor_runner::{DoctorArgs, run_doctor};
pub use finalize_runner::run_finalize;
pub use git_import_runner::{ImportGitArgs, run_import_git};
pub use import_runner::{ImportArgs, run_import};
pub use init_runner::{DEFAULT_CONFIG_TOML, resolve_effective_config_path, run_init};
pub use server_runner::run_server;
pub use service::{ServiceAction, run_service};
