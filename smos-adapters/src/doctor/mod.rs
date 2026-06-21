//! `doctor` — environment validation + Markdown report generation for SMOS.
//!
//! This module hosts the **pure, unit-tested** helpers used by the
//! `smos doctor` subcommand:
//!
//! - [`types`] — `CheckResult`, `CheckStatus`, `DoctorReport`, `StatsSnapshot`
//! - [`models`] — Ollama model availability matching (case-insensitive,
//!   tag-suffix-tolerant)
//! - [`aggregation`] — summary line + recommendation aggregation
//! - [`markdown`] — Markdown report renderer
//! - [`terminal`] — ANSI-aware terminal renderer
//! - [`checks`] — IO entry points (Ollama, SurrealDB, binary)
//!
//! # Layering
//!
//! The runner in `src/cli/doctor_runner.rs` is intentionally thin: it
//! converts the parsed `smos doctor` args into a `DoctorArgs` struct, calls
//! [`checks::run_full_check`] or [`checks::run_stats_only`], and delegates
//! rendering to [`terminal`] / [`markdown`]. Every shape and every pure
//! helper is reachable from `tests/doctor_unit.rs` so the doctor logic is
//! covered by automated tests, not just manual smoke runs.
//!
//! # Smoke-test scope
//!
//! The IO checks here touch real external systems (Ollama, SurrealDB). They
//! are NEVER invoked from the test suite — only the pure helpers are
//! exercised. Operators invoke the IO paths through the `smos doctor`
//! subcommand as part of the manual smoke checklist.

pub mod aggregation;
pub mod checks;
pub mod markdown;
pub mod models;
pub mod terminal;
pub mod types;

pub use aggregation::{collect_recommendations, summary_line};
pub use checks::{DoctorFlags, run_full_check, run_stats_only};
pub use markdown::render_markdown;
pub use models::{ExpectedModel, match_expected_models};
pub use terminal::{ColorMode, format_check, format_stats, render_terminal};
pub use types::{CheckResult, CheckStatus, DoctorReport, ReportSummary, StatsSnapshot, aggregate};
