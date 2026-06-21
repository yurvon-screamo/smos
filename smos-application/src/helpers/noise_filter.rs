//! Extraction noise filter — strip SMOS-internal artifacts before extraction
//! (§4, response_pipeline).
//!
//! Three classes of noise are removed so the extractor never turns SMOS control
//! metadata into a "fact":
//!
//! 1. Session markers `<!-- smos:sess_xxx -->` appended to responses.
//! 2. `<smos-memory session="...">…</smos-memory>` blocks (DOTALL — the block
//!    can span many lines).
//! 3. Bare `sess_<token>` identifiers the upstream may have echoed back without
//!    their marker wrapper (the extractor would otherwise lift them into a
//!    "fact" like "the session id is sess_...").
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

/// Bare session id prefixed by either start-of-text or a non-word character.
/// The prefix is captured so substitution can preserve it (otherwise we'd eat
/// the space before a bare id).
static BARE_SESS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(^|[^A-Za-z0-9_])sess_[A-Za-z0-9_]+").expect("bare sess regex literal")
});

/// Return `content` with all SMOS-internal noise stripped, trimmed.
pub fn clean(content: &str) -> String {
    let without_markers = MARKERS_RE.replace_all(content, "");
    let without_bare = BARE_SESS_RE.replace_all(&without_markers, "${1}");
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
}
