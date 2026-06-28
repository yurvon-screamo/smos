//! Extraction noise filter — strip SMOS-internal artifacts before extraction
//! (§4, response_pipeline).
//!
//! Four classes of noise are removed so the extractor never turns control
//! metadata or model internals into a "fact":
//!
//! 1. Session markers `<!-- smos:sess_xxx -->` appended to responses.
//! 2. `<smos-memory session="...">…</smos-memory>` blocks (DOTALL — the block
//!    can span many lines).
//! 3. Bare `sess_<token>` identifiers the upstream may have echoed back without
//!    their marker wrapper (the extractor would otherwise lift them into a
//!    "fact" like "the session id is sess_...").
//! 4. Inline `<think>…</think>` reasoning blocks. Reasoning models (DeepSeek-R1,
//!    Qwen-QwQ, GLM-thinking via OpenRouter) emit their chain-of-thought inline
//!    in `content`. The reasoning can be an order of magnitude longer than the
//!    answer and carries no extractable fact, so it is pure token waste for the
//!    extractor. Both the closed form and an unclosed `<think>…` (streaming
//!    cut-off / dropped `</think>`) are stripped.
//!
//! Known limitation: a stray `</think>` with no opening `<think>` is not
//! matched by either think pattern. Reasoning models always emit `<think>`
//! first, so this does not arise in practice, but it is documented for
//! completeness.
//!
//! The bare-id filter must avoid mid-word tokens like `obsess_token` or
//! `disse_data`. Rust's `regex` crate does not support lookbehind, so we use a
//! capture group that records the leading non-word char and re-emit it during
//! substitution to preserve surrounding text.

use regex::Regex;
use std::sync::LazyLock;

/// Markers + memory blocks. Plain alternation — no lookbehind needed.
/// `(?s:...)` makes `.` match newlines so the memory block can span multiple
/// lines (mirrors the POC's `re.DOTALL` flag).
static MARKERS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<!--\s*smos:\S+?\s*-->|(?s)<smos-memory[^>]*>.*?</smos-memory>")
        .expect("markers regex literal")
});

/// Closed `<think>…</think>` reasoning block (non-greedy, DOTALL so the body
/// can span multiple lines). Runs BEFORE [`THINK_OPEN_RE`] so a properly
/// closed block is removed before the unclosed pass — otherwise the greedy
/// open pattern would eat from the first `<think>` to end-of-string.
static THINK_CLOSED_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>.*?</think>").expect("think-closed regex literal"));

/// Unclosed `<think>…` (no `</think>` — streaming cut-off / dropped tag).
/// Greedy to end-of-string: a partial reasoning trail has no fact boundary.
static THINK_OPEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<think>.*$").expect("think-open regex literal"));

/// Bare session id prefixed by either start-of-text or a non-word character.
/// The prefix is captured so substitution can preserve it (otherwise we'd eat
/// the space before a bare id).
static BARE_SESS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(^|[^A-Za-z0-9_])sess_[A-Za-z0-9_]+").expect("bare sess regex literal")
});

/// Return `content` with all SMOS-internal noise stripped, trimmed.
pub fn clean(content: &str) -> String {
    let without_markers = MARKERS_RE.replace_all(content, "");
    let without_think_closed = THINK_CLOSED_RE.replace_all(&without_markers, "");
    let without_think_open = THINK_OPEN_RE.replace_all(&without_think_closed, "");
    let without_bare = BARE_SESS_RE.replace_all(&without_think_open, "${1}");
    without_bare.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_session_marker_comment() {
        let input = "hello\n<!-- smos:sess_abcdef012345 -->";
        assert_eq!(clean(input), "hello");
    }

    #[test]
    fn strips_multiline_smos_memory_block() {
        let input = "before\n<smos-memory session=\"sess_x\">\n[fact_1] doc\n</smos-memory>\nafter";
        let out = clean(input);
        assert!(out.contains("before"));
        assert!(out.contains("after"));
        assert!(!out.contains("smos-memory"));
        assert!(!out.contains("fact_1"));
    }

    #[test]
    fn strips_smos_memory_block_with_attributes() {
        let input = "<smos-memory session=\"sess_y\" extra=\"value\">body</smos-memory>tail";
        let out = clean(input);
        assert_eq!(out, "tail");
    }

    #[test]
    fn strips_bare_session_id_preserving_surrounding_text() {
        let input = "the session id is sess_abcdef012345 here";
        assert_eq!(clean(input), "the session id is  here");
    }

    #[test]
    fn preserves_session_id_embedded_in_a_word() {
        let input = "obsess_token must survive";
        assert_eq!(clean(input), "obsess_token must survive");
    }

    #[test]
    fn preserves_normal_content_without_noise() {
        let input = "Just a regular fact about Rust and cargo.";
        assert_eq!(clean(input), input);
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert_eq!(clean(""), "");
        assert_eq!(clean("   "), "");
    }

    #[test]
    fn strips_bare_id_at_start_of_text() {
        let input = "sess_aabbccddeeff is the id";
        assert_eq!(clean(input), "is the id");
    }

    #[test]
    fn strips_multiple_distinct_noise_patterns_in_one_pass() {
        let input = "marker <!-- smos:sess_1 --> bare sess_aabbccddeeff block <smos-memory session=\"s\">x</smos-memory>";
        let out = clean(input);
        assert_eq!(out, "marker  bare  block");
    }

    #[test]
    fn strips_closed_think_block_at_start() {
        let input = "<think>let me reason about this</think>The answer is 42.";
        assert_eq!(clean(input), "The answer is 42.");
    }

    #[test]
    fn strips_closed_think_block_in_middle() {
        let input = "Before.<think>internal deliberation</think>After.";
        assert_eq!(clean(input), "Before.After.");
    }

    #[test]
    fn strips_closed_think_block_at_end() {
        let input = "Real fact here.<think>and some trailing rumination</think>";
        assert_eq!(clean(input), "Real fact here.");
    }

    #[test]
    fn strips_multiline_think_block_body() {
        // The think block sits on its own line between two newlines; removing
        // it leaves one blank line (internal whitespace is preserved — only
        // leading/trailing is trimmed by `clean`).
        let input = "fact\n<think>line one\nline two\nline three</think>\nmore fact";
        assert_eq!(clean(input), "fact\n\nmore fact");
    }

    #[test]
    fn strips_unclosed_think_to_end_of_string() {
        let input = "answer<think>reasoning that never got a closing tag";
        assert_eq!(clean(input), "answer");
    }

    #[test]
    fn strips_adjacent_closed_think_blocks() {
        let input = "<think>A</think><think>B</think>final";
        assert_eq!(clean(input), "final");
    }

    #[test]
    fn closed_think_stripped_before_unclosed_pass() {
        // A closed block followed by a genuinely unclosed one: the closed pass
        // removes the first, leaving only the unclosed trail for the open pass.
        let input = "<think>closed reasoning</think>fact<think>unclosed trail";
        assert_eq!(clean(input), "fact");
    }

    #[test]
    fn normal_text_without_think_is_unchanged() {
        let input = "The cache uses TTL=60 to avoid stale entries.";
        assert_eq!(clean(input), input);
    }

    #[test]
    fn strips_think_combined_with_markers_and_bare_id() {
        let input = "<think>noise</think>real fact <!-- smos:sess_1 --> sess_aabb <smos-memory session=\"s\">x</smos-memory>";
        let out = clean(input);
        assert_eq!(out, "real fact");
    }
}
