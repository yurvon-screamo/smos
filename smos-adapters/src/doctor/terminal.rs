//! Terminal output for the doctor binary.
//!
//! Pure functions over the report shape — the IO (writing to stdout) lives
//! in the binary. ANSI colour codes are emitted only when the caller passes
//! `ColorMode::Auto` AND stdout is detected as a TTY (the binary performs
//! that detection); otherwise the output is plain text so logs and pipes
//! stay grep-friendly.

use super::aggregation::summary_line;
use super::types::{CheckResult, DoctorReport, StatsSnapshot};

/// Whether to emit ANSI colour escapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Always,
    Never,
    Auto,
}

impl ColorMode {
    /// Resolve `Auto` against a runtime TTY flag. `Always` / `Never` are
    /// passed through unchanged so callers can force the behaviour from
    /// the CLI (`--color=always`, `--color=never`).
    pub fn resolve(self, is_tty: bool) -> bool {
        match self {
            ColorMode::Always => true,
            ColorMode::Never => false,
            ColorMode::Auto => is_tty,
        }
    }
}

/// Top-level entry: render the full terminal block.
pub fn render_terminal(report: &DoctorReport, use_color: bool) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("SMOS Doctor — Environment Check\n");
    out.push_str("================================\n");
    out.push_str(&format!("[{}] Starting checks...\n\n", report.generated_at));

    for c in &report.checks {
        out.push_str(&format_check(c, use_color));
        out.push('\n');
    }

    if let Some(stats) = &report.stats {
        out.push_str(&format_stats(stats));
        out.push('\n');
    }

    let summary = report.summary();
    out.push_str("================================\n");
    out.push_str(&summary_line(summary));
    out.push('\n');
    out
}

/// Format one check as a terminal block. Three lines:
/// 1. `[STATUS] name` with the status colour-coded.
/// 2. (optional) indented details, one line per `\n`-separated segment.
/// 3. (optional) indented recommendation.
pub fn format_check(c: &CheckResult, use_color: bool) -> String {
    let label = c.status.as_label();
    let coloured_label = if use_color {
        colour_label(label, c)
    } else {
        label.to_string()
    };
    let mut out = String::with_capacity(128);
    out.push_str(&format!("[{}] {}\n", coloured_label, c.name));
    if !c.details.is_empty() {
        for line in c.details.split('\n') {
            if line.is_empty() {
                continue;
            }
            out.push_str(&format!("       {line}\n"));
        }
    }
    if let Some(rec) = &c.recommendation {
        out.push_str(&format!("       Recommendation: {rec}\n"));
    }
    out
}

/// Format the stats block as a fixed-shape terminal section.
pub fn format_stats(stats: &StatsSnapshot) -> String {
    let mut out = String::with_capacity(256);
    out.push_str("[STATS] SurrealDB snapshot\n");
    out.push_str(&format!(
        "       facts: {} (accepted: {}, pending: {}, rejected: {})\n",
        stats.total_facts, stats.accepted, stats.pending, stats.rejected,
    ));
    out.push_str(&format!(
        "       sessions: {} (active: {}, ended: {})\n",
        stats.total_sessions, stats.active_sessions, stats.ended_sessions,
    ));
    out
}

fn colour_label(label: &str, c: &CheckResult) -> String {
    match c.status {
        super::types::CheckStatus::Pass => format!("\x1b[32m{label}\x1b[0m"),
        super::types::CheckStatus::Warn => format!("\x1b[33m{label}\x1b[0m"),
        super::types::CheckStatus::Fail => format!("\x1b[31m{label}\x1b[0m"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DoctorReport {
        let mut r = DoctorReport::new("2026-06-18T13:45:01Z", "smos.toml");
        r.push(CheckResult::pass("smos binary", "version: 0.1.0"));
        r.push(
            CheckResult::warn(
                "Reranker (REQUIRED for production-quality enrichment)",
                "url unreachable",
            )
            .with_recommendation(
                "reranker REQUIRED for production-quality enrichment — start the \
                 llama.cpp reranker server; without it the enrich pipeline runs \
                 in degraded mode (vector-order-only ranking)",
            ),
        );
        r.push(
            CheckResult::fail("granite4.1:3b", "missing")
                .with_recommendation("ollama pull granite4.1:3b"),
        );
        r.stats = Some(StatsSnapshot {
            total_facts: 5,
            accepted: 3,
            pending: 2,
            rejected: 0,
            total_sessions: 1,
            active_sessions: 0,
            ended_sessions: 1,
        });
        r
    }

    #[test]
    fn color_mode_resolve_passes_through_always_and_never() {
        assert!(ColorMode::Always.resolve(false));
        assert!(ColorMode::Always.resolve(true));
        assert!(!ColorMode::Never.resolve(true));
        assert!(!ColorMode::Never.resolve(false));
    }

    #[test]
    fn color_mode_resolve_auto_follows_tty_flag() {
        assert!(ColorMode::Auto.resolve(true));
        assert!(!ColorMode::Auto.resolve(false));
    }

    #[test]
    fn render_without_color_emits_plain_labels() {
        let out = render_terminal(&sample(), false);
        assert!(out.contains("[PASS] smos binary"));
        assert!(out.contains("[WARN] Reranker (REQUIRED for production-quality enrichment)"));
        assert!(out.contains("[FAIL] granite4.1:3b"));
        // No ANSI escapes when color is off.
        assert!(!out.contains("\x1b["));
    }

    #[test]
    fn render_with_color_wraps_label_in_escape_sequence() {
        let out = render_terminal(&sample(), true);
        // Green for PASS, yellow for WARN, red for FAIL.
        assert!(out.contains("\x1b[32mPASS\x1b[0m"));
        assert!(out.contains("\x1b[33mWARN\x1b[0m"));
        assert!(out.contains("\x1b[31mFAIL\x1b[0m"));
    }

    #[test]
    fn render_emits_recommendation_line_indented() {
        let out = render_terminal(&sample(), false);
        assert!(out.contains("       Recommendation: reranker REQUIRED"));
        assert!(out.contains("       Recommendation: ollama pull granite4.1:3b"));
    }

    #[test]
    fn render_stats_block_includes_all_counters() {
        let out = render_terminal(&sample(), false);
        assert!(out.contains("facts: 5 (accepted: 3, pending: 2, rejected: 0)"));
        assert!(out.contains("sessions: 1 (active: 0, ended: 1)"));
    }

    #[test]
    fn summary_line_present_at_bottom() {
        let out = render_terminal(&sample(), false);
        assert!(out.contains("Result: 1/3 PASS, 1 WARN, 1 FAIL"));
    }

    #[test]
    fn format_check_skips_blank_detail_lines() {
        let c = CheckResult::pass("multi", "first\n\nsecond");
        let out = format_check(&c, false);
        assert!(out.contains("first"));
        assert!(out.contains("second"));
        // The empty middle line should not produce a stray "       \n".
        assert!(!out.contains("       \n"));
    }

    #[test]
    fn header_contains_timestamp() {
        let out = render_terminal(&sample(), false);
        assert!(out.contains("[2026-06-18T13:45:01Z] Starting checks..."));
    }
}
