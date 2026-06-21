//! Pure helpers around the `ReportSummary` aggregate used by the terminal
//! and Markdown renderers. Kept separate from [`super::types`] because the
//! helpers here compute derived views (success line, badge) rather than
//! defining new shapes.

use super::types::{CheckResult, ReportSummary};

/// Format the trailing summary line shown after the per-check block in the
/// terminal. Mirrors the spec format `Result: 7/8 PASS, 1 WARN, 0 FAIL`.
pub fn summary_line(summary: ReportSummary) -> String {
    let total = summary.total();
    format!(
        "Result: {}/{} PASS, {} WARN, {} FAIL",
        summary.pass, total, summary.warn, summary.fail,
    )
}

/// Build the recommendation list for the Markdown report. Only checks that
/// did not pass contribute; passing checks are intentionally silent so the
/// section stays compact.
///
/// When a failing/warning check carries no `recommendation`, the check name
/// is still emitted so the operator sees the full delta — silent omissions
/// here would let broken infra disappear from the report.
pub fn collect_recommendations(results: &[CheckResult]) -> Vec<String> {
    results
        .iter()
        .filter(|r| !r.status.is_pass())
        .map(|r| match r.recommendation.as_ref() {
            Some(hint) if !hint.trim().is_empty() => format!("- {}: {}", r.name, hint),
            _ => format!("- {}", r.name),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::types::CheckStatus;
    use super::*;

    #[test]
    fn summary_line_includes_all_three_counters() {
        let line = summary_line(ReportSummary {
            pass: 5,
            warn: 2,
            fail: 1,
        });
        assert!(line.contains("5/8 PASS"));
        assert!(line.contains("2 WARN"));
        assert!(line.contains("1 FAIL"));
    }

    #[test]
    fn summary_line_zero_failures_still_renders_total() {
        let line = summary_line(ReportSummary {
            pass: 7,
            warn: 1,
            fail: 0,
        });
        assert!(line.contains("7/8 PASS"));
        assert!(line.contains("1 WARN"));
        assert!(line.contains("0 FAIL"));
    }

    #[test]
    fn collect_recommendations_skips_passing_checks() {
        let results = vec![
            CheckResult::pass("ok", ""),
            CheckResult::fail("bad", "").with_recommendation("pull model"),
            CheckResult::warn("optional", "").with_recommendation("start reranker"),
        ];
        let recs = collect_recommendations(&results);
        assert_eq!(recs.len(), 2);
        assert!(recs[0].contains("bad"));
        assert!(recs[0].contains("pull model"));
        assert!(recs[1].contains("optional"));
        assert!(recs[1].contains("start reranker"));
    }

    #[test]
    fn collect_recommendations_handles_missing_hint() {
        let results = vec![CheckResult::fail("bad", "")];
        let recs = collect_recommendations(&results);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0], "- bad");
    }

    #[test]
    fn check_status_predicates_are_mutually_exclusive() {
        assert!(CheckStatus::Pass.is_pass());
        assert!(!CheckStatus::Pass.is_warn());
        assert!(!CheckStatus::Pass.is_fail());
        assert!(CheckStatus::Warn.is_warn());
        assert!(CheckStatus::Fail.is_fail());
    }
}
