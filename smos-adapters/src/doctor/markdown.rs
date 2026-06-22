//! Markdown report generator for the doctor.
//!
//! Pure function over [`DoctorReport`] — no IO. The output is a Markdown
//! document bounded by ~200 lines (per spec) that captures the same
//! information as the terminal output but in a shareable form. The shape is:
//!
//! 1. Header with timestamp + config path
//! 2. Summary table (one row per check)
//! 3. Stats section (when a SurrealDB snapshot is attached)
//! 4. Recommendations list (only failing/warning checks)
//!
//! Section ordering is fixed so downstream tooling (grep, diff) can rely on
//! stable anchors.

use super::aggregation::{collect_recommendations, summary_line};
use super::types::{CheckResult, DoctorReport, StatsSnapshot};

/// Top-level entry: render the full Markdown document.
pub fn render_markdown(report: &DoctorReport) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("# SMOS Doctor Report\n\n");
    out.push_str(&format!("**Generated:** {}\n\n", report.generated_at));
    out.push_str(&format!("**Config:** {}\n\n", report.config_path));
    out.push_str("## Summary\n\n");
    out.push_str(&render_summary_table(&report.checks));
    let summary = report.summary();
    out.push_str(&format!("\n**{}**\n\n", summary_line(summary)));
    if let Some(stats) = &report.stats {
        out.push_str("## Stats\n\n");
        out.push_str(&render_stats(stats));
        out.push('\n');
    }
    out.push_str("## Details\n\n");
    out.push_str(&render_details(&report.checks));
    let recs = collect_recommendations(&report.checks);
    if !recs.is_empty() {
        out.push_str("## Recommendations\n\n");
        for r in &recs {
            out.push_str(r);
            out.push('\n');
        }
    }
    out
}

/// Render the per-check summary table. The Markdown is intentionally narrow
/// (2 columns) so the document renders well on GitHub and in static-site
/// generators that wrap tables.
fn render_summary_table(checks: &[CheckResult]) -> String {
    let mut out = String::with_capacity(checks.len() * 32 + 64);
    out.push_str("| Check | Status |\n");
    out.push_str("|-------|--------|\n");
    for c in checks {
        out.push_str(&format!("| {} | {} |\n", c.name, c.status.as_label()));
    }
    out
}

/// Render the per-check detail block. Each check gets a `### name (STATUS)`
/// heading followed by bullet lines for the details + recommendation.
fn render_details(checks: &[CheckResult]) -> String {
    let mut out = String::with_capacity(checks.len() * 128);
    for c in checks {
        out.push_str(&format!("### {} ({})\n\n", c.name, c.status.as_label()));
        if !c.details.is_empty() {
            for line in c.details.split("\n") {
                out.push_str(&format!("- {line}\n"));
            }
        }
        if let Some(rec) = &c.recommendation {
            out.push_str(&format!("- Recommendation: {rec}\n"));
        }
        out.push('\n');
    }
    out
}

/// Render the SurrealDB stats snapshot as a fixed bullet list.
pub fn render_stats(stats: &StatsSnapshot) -> String {
    let mut out = String::with_capacity(256);
    out.push_str(&format!("- Total facts: {}\n", stats.total_facts));
    out.push_str(&format!("  - Accepted: {}\n", stats.accepted));
    out.push_str(&format!("  - Pending: {}\n", stats.pending));
    out.push_str(&format!("  - Rejected: {}\n", stats.rejected));
    out.push_str(&format!("- Sessions: {}\n", stats.total_sessions));
    out.push_str(&format!("  - Active: {}\n", stats.active_sessions));
    out.push_str(&format!("  - Ended: {}\n", stats.ended_sessions));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_report() -> DoctorReport {
        let mut r = DoctorReport::new("2026-06-18T13:45:01Z", "smos.toml");
        r.push(CheckResult::pass("smos binary", "version: 0.1.0"));
        r.push(CheckResult::pass(
            "Ollama connectivity",
            "url: http://localhost:11434\nmodels: 12",
        ));
        r.push(
            CheckResult::fail("Reranker", "url: http://localhost:8181 unreachable")
                .with_recommendation(
                    "start the llama.cpp reranker server; every chat-completion \
                     request fails with HTTP 503 while it is down",
                ),
        );
        r.push(
            CheckResult::fail("granite4.1:3b", "model missing")
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
    fn markdown_header_contains_timestamp_and_config() {
        let md = render_markdown(&sample_report());
        assert!(md.contains("# SMOS Doctor Report"));
        assert!(md.contains("**Generated:** 2026-06-18T13:45:01Z"));
        assert!(md.contains("**Config:** smos.toml"));
    }

    #[test]
    fn markdown_summary_table_lists_every_check() {
        let md = render_markdown(&sample_report());
        assert!(md.contains("| smos binary | PASS |"));
        // Reranker is a hard dependency → FAIL when unreachable.
        assert!(md.contains("| Reranker | FAIL |"));
        assert!(md.contains("| granite4.1:3b | FAIL |"));
    }

    #[test]
    fn markdown_summary_line_renders_totals() {
        let md = render_markdown(&sample_report());
        // 2 PASS + 2 FAIL, 0 WARN after the reranker was promoted to FAIL.
        assert!(md.contains("Result: 2/4 PASS, 0 WARN, 2 FAIL"));
    }

    #[test]
    fn markdown_stats_section_present_when_snapshot_attached() {
        let md = render_markdown(&sample_report());
        assert!(md.contains("## Stats"));
        assert!(md.contains("Total facts: 5"));
        assert!(md.contains("Accepted: 3"));
        assert!(md.contains("Sessions: 1"));
        assert!(md.contains("Active: 0"));
    }

    #[test]
    fn markdown_recommendations_section_only_non_pass() {
        let md = render_markdown(&sample_report());
        assert!(md.contains("## Recommendations"));
        assert!(md.contains("Reranker: start the llama.cpp reranker server"));
        assert!(md.contains("granite4.1:3b: ollama pull granite4.1:3b"));
        // Passing checks should not be echoed as recommendations.
        assert!(!md.contains("Recommendations\n- smos binary"));
    }

    #[test]
    fn markdown_details_split_each_detail_line_as_bullet() {
        let md = render_markdown(&sample_report());
        // "Ollama connectivity" details contained a newline → two bullets.
        assert!(md.contains("- url: http://localhost:11434"));
        assert!(md.contains("- models: 12"));
    }

    #[test]
    fn markdown_omits_stats_section_when_no_snapshot() {
        let mut r = DoctorReport::new("t", "c");
        r.push(CheckResult::pass("ok", ""));
        let md = render_markdown(&r);
        assert!(!md.contains("## Stats"));
    }

    #[test]
    fn render_stats_matches_expected_layout() {
        let stats = StatsSnapshot {
            total_facts: 10,
            accepted: 4,
            pending: 5,
            rejected: 1,
            total_sessions: 2,
            active_sessions: 1,
            ended_sessions: 1,
        };
        let s = render_stats(&stats);
        assert!(s.contains("Total facts: 10"));
        assert!(s.contains("Accepted: 4"));
        assert!(s.contains("Pending: 5"));
        assert!(s.contains("Rejected: 1"));
        assert!(s.contains("Sessions: 2"));
        assert!(s.contains("Active: 1"));
        assert!(s.contains("Ended: 1"));
    }

    #[test]
    fn markdown_total_lines_stays_under_two_hundred() {
        let md = render_markdown(&sample_report());
        // Spec target ≤200 lines even for a fully-populated report. The
        // doctor never produces more than ~20 checks in practice; the bound
        // is generous on purpose so adding a future section is safe.
        assert!(
            md.lines().count() <= 200,
            "markdown report should stay compact, got {} lines",
            md.lines().count()
        );
    }

    #[test]
    fn empty_report_still_emits_header_and_summary() {
        let r = DoctorReport::new("2026-01-01T00:00:00Z", "missing.toml");
        let md = render_markdown(&r);
        assert!(md.contains("# SMOS Doctor Report"));
        assert!(md.contains("Result: 0/0 PASS"));
    }

    #[test]
    fn details_section_omits_block_when_details_empty() {
        let mut r = DoctorReport::new("t", "c");
        r.push(CheckResult::pass("ok", ""));
        let md = render_markdown(&r);
        // Heading is always emitted, but no bullet under it.
        assert!(md.contains("### ok (PASS)"));
        // No stray empty bullet (`- ` followed by newline).
        assert!(!md.contains("- \n"));
    }
}
