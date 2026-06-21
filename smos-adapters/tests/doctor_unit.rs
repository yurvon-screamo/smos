//! Integration tests for the `smos_adapters::doctor` public API.
//!
//! These tests cover ONLY pure helpers — model matching, formatting,
//! aggregation, and end-to-end renderers. The IO entry points
//! (Ollama / SurrealDB probes) are exercised manually during the
//! smoke test (`docs/SMOKE_TEST.md`); automating them here would couple
//! the suite to live external systems and violate the "tests stay fast"
//! contract of the workspace.

use smos_adapters::doctor::terminal::ColorMode;
use smos_adapters::doctor::{
    CheckResult, CheckStatus, DoctorFlags, DoctorReport, ExpectedModel, StatsSnapshot, aggregate,
    collect_recommendations, match_expected_models, render_markdown, render_terminal, summary_line,
};

// ---------------------------------------------------------------------------
// Model matching
// ---------------------------------------------------------------------------

#[test]
fn model_match_recognises_exact_id() {
    let expected = vec![
        ExpectedModel::new("upstream", "granite4.1:3b"),
        ExpectedModel::new("extraction", "qwen3.5:2b"),
    ];
    let available = vec!["granite4.1:3b".to_string(), "qwen3.5:2b".to_string()];
    let out = match_expected_models(&expected, &available);
    assert!(out.iter().all(|(_, hit)| *hit));
}

#[test]
fn model_match_flags_missing_models_individually() {
    let expected = vec![
        ExpectedModel::new("upstream", "granite4.1:3b"),
        ExpectedModel::new("extraction", "qwen3.5:2b"),
    ];
    let available = vec!["granite4.1:3b".to_string()];
    let out = match_expected_models(&expected, &available);
    assert!(out[0].1, "upstream model must match");
    assert!(!out[1].1, "extraction model must be flagged missing");
}

#[test]
fn model_match_handles_huggingface_repo_variants() {
    // Same publisher + repo name, different quantisation tag → match.
    let expected = vec![ExpectedModel::new(
        "embedding",
        "hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:latest",
    )];
    let available =
        vec!["hf.co/jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0".to_string()];
    let out = match_expected_models(&expected, &available);
    assert!(out[0].1);
}

// ---------------------------------------------------------------------------
// Aggregation + summary
// ---------------------------------------------------------------------------

#[test]
fn aggregate_classifies_each_status_correctly() {
    let results = vec![
        CheckResult::pass("a", "x"),
        CheckResult::pass("b", "x"),
        CheckResult::warn("c", "x"),
        CheckResult::fail("d", "x"),
    ];
    let s = aggregate(&results);
    assert_eq!(s.pass, 2);
    assert_eq!(s.warn, 1);
    assert_eq!(s.fail, 1);
}

#[test]
fn summary_line_format_matches_smoke_test_spec() {
    let s = smos_adapters::doctor::ReportSummary {
        pass: 7,
        warn: 1,
        fail: 0,
    };
    let line = summary_line(s);
    assert!(line.contains("7/8 PASS"));
    assert!(line.contains("1 WARN"));
    assert!(line.contains("0 FAIL"));
}

#[test]
fn collect_recommendations_skips_pass_and_drops_empty_hint() {
    let results = vec![
        CheckResult::pass("ok", ""),
        CheckResult::fail("bad", "").with_recommendation("pull model"),
        CheckResult::warn("optional", "").with_recommendation("start reranker"),
        CheckResult::fail("silent", ""),
    ];
    let recs = collect_recommendations(&results);
    assert_eq!(recs.len(), 3);
    assert!(
        recs.iter()
            .any(|r| r.contains("bad") && r.contains("pull model"))
    );
    assert!(recs.iter().any(|r| r.contains("optional")));
    assert!(recs.iter().any(|r| r.contains("silent")));
}

// ---------------------------------------------------------------------------
// Markdown renderer
// ---------------------------------------------------------------------------

#[test]
fn markdown_includes_header_summary_stats_and_recommendations() {
    let mut report = DoctorReport::new("2026-06-18T13:45:01Z", "smos.toml");
    report.push(CheckResult::pass("smos binary", "version: 0.1.0"));
    report.push(
        CheckResult::fail("granite4.1:3b", "missing")
            .with_recommendation("ollama pull granite4.1:3b"),
    );
    report.stats = Some(StatsSnapshot {
        total_facts: 5,
        accepted: 3,
        pending: 2,
        rejected: 0,
        total_sessions: 1,
        active_sessions: 0,
        ended_sessions: 1,
    });

    let md = render_markdown(&report);
    assert!(md.contains("# SMOS Doctor Report"));
    assert!(md.contains("**Generated:** 2026-06-18T13:45:01Z"));
    assert!(md.contains("**Config:** smos.toml"));
    assert!(md.contains("| smos binary | PASS |"));
    assert!(md.contains("| granite4.1:3b | FAIL |"));
    assert!(md.contains("Result: 1/2 PASS, 0 WARN, 1 FAIL"));
    assert!(md.contains("## Stats"));
    assert!(md.contains("Total facts: 5"));
    assert!(md.contains("## Recommendations"));
    assert!(md.contains("granite4.1:3b: ollama pull granite4.1:3b"));
}

#[test]
fn markdown_total_lines_stays_under_two_hundred() {
    let mut report = DoctorReport::new("2026-06-18T13:45:01Z", "smos.toml");
    // A pathological 30-check report must still stay compact.
    for i in 0..30 {
        report.push(CheckResult::pass(
            format!("check-{i}"),
            format!("detail-{i}"),
        ));
    }
    let md = render_markdown(&report);
    assert!(
        md.lines().count() <= 200,
        "got {} lines",
        md.lines().count()
    );
}

#[test]
fn markdown_omits_recommendations_section_when_everything_passes() {
    let mut report = DoctorReport::new("2026-06-18T13:45:01Z", "smos.toml");
    report.push(CheckResult::pass("ok", ""));
    let md = render_markdown(&report);
    assert!(!md.contains("## Recommendations"));
}

// ---------------------------------------------------------------------------
// Terminal renderer
// ---------------------------------------------------------------------------

#[test]
fn terminal_without_color_omits_ansi_escapes() {
    let mut report = DoctorReport::new("2026-06-18T13:45:01Z", "smos.toml");
    report.push(CheckResult::pass("ok", "v0.1.0"));
    report.push(CheckResult::warn("reranker", "down"));
    let out = render_terminal(&report, false);
    assert!(out.contains("[PASS] ok"));
    assert!(out.contains("[WARN] reranker"));
    assert!(!out.contains("\x1b["));
}

#[test]
fn terminal_with_color_wraps_labels_with_correct_codes() {
    let mut report = DoctorReport::new("2026-06-18T13:45:01Z", "smos.toml");
    report.push(CheckResult::pass("ok", ""));
    report.push(CheckResult::warn("opt", ""));
    report.push(CheckResult::fail("bad", ""));
    let out = render_terminal(&report, true);
    assert!(out.contains("\x1b[32mPASS\x1b[0m"));
    assert!(out.contains("\x1b[33mWARN\x1b[0m"));
    assert!(out.contains("\x1b[31mFAIL\x1b[0m"));
}

#[test]
fn terminal_stats_block_lists_fact_breakdown() {
    let mut report = DoctorReport::new("2026-06-18T13:45:01Z", "smos.toml");
    report.stats = Some(StatsSnapshot {
        total_facts: 10,
        accepted: 5,
        pending: 3,
        rejected: 2,
        total_sessions: 2,
        active_sessions: 1,
        ended_sessions: 1,
    });
    let out = render_terminal(&report, false);
    assert!(out.contains("facts: 10 (accepted: 5, pending: 3, rejected: 2)"));
    assert!(out.contains("sessions: 2 (active: 1, ended: 1)"));
}

// ---------------------------------------------------------------------------
// ColorMode resolution
// ---------------------------------------------------------------------------

#[test]
fn color_mode_auto_follows_tty_flag() {
    assert!(ColorMode::Auto.resolve(true));
    assert!(!ColorMode::Auto.resolve(false));
}

#[test]
fn color_mode_always_is_unaffected_by_tty() {
    assert!(ColorMode::Always.resolve(false));
    assert!(ColorMode::Always.resolve(true));
}

// ---------------------------------------------------------------------------
// Flags + summary semantics
// ---------------------------------------------------------------------------

#[test]
fn doctor_flags_default_matches_smoke_test_spec() {
    let flags = DoctorFlags::default();
    assert!(!flags.skip_ollama);
}

#[test]
fn report_summary_is_success_only_when_no_failures() {
    let mut report = DoctorReport::new("t", "c");
    report.push(CheckResult::pass("a", ""));
    report.push(CheckResult::warn("b", ""));
    assert!(report.summary().is_success());

    report.push(CheckResult::fail("c", ""));
    assert!(!report.summary().is_success());
}

#[test]
fn check_status_predicates_are_mutually_exclusive() {
    assert!(
        CheckStatus::Pass.is_pass() && !CheckStatus::Pass.is_warn() && !CheckStatus::Pass.is_fail()
    );
    assert!(
        !CheckStatus::Warn.is_pass() && CheckStatus::Warn.is_warn() && !CheckStatus::Warn.is_fail()
    );
    assert!(
        !CheckStatus::Fail.is_pass() && !CheckStatus::Fail.is_warn() && CheckStatus::Fail.is_fail()
    );
}
