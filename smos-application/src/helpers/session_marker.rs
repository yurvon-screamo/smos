//! Session marker — `<!-- smos:sess_xxx -->` detection (§3 step 2, §4 step 2).
//!
//! Only the *detection* (recovery of a previously-stored session id from the
//! conversation history) lives here — it uses regex + JSON traversal, which are
//! not domain concerns. The complementary *generation* of a new marker from a
//! `SessionId` lives on the value object itself as [`SessionId::to_marker`].
//!
//! The marker round-trips the session id through the conversation: SMOS appends
//! it to assistant responses, the client stores it in history, and SMOS scans
//! the trailing 20 messages (of any role) to recover the id on the next
//! request. Marker loss (history compaction, truncation) is acceptable — it
//! just means a new session starts and facts are re-injected.

use regex::Regex;
use serde_json::Value;
use smos_domain::SessionId;
use std::sync::LazyLock;

use crate::helpers::openai_content::flatten_text;
use crate::types::EnrichmentMessages;

/// Match the marker and capture the inner session id token.
///
/// The pattern mirrors the POC: any non-whitespace run, framed by the
/// `<!-- smos: ... -->` comment. Materialisation via [`SessionId::from_raw`]
/// rejects malformed captures so a stray comment never breaks detection.
static MARKER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<!--\s*smos:(\S+?)\s*-->").expect("smos marker regex literal"));

/// Number of trailing messages scanned for a marker (§3 step 2). Applies to
/// both `detect_in_text` (string slice) and `detect_from_messages` (OpenAI).
const WINDOW: usize = 20;

/// Scan a single text blob for the most recent marker; return its session id.
pub fn detect_in_text(text: &str) -> Option<SessionId> {
    // Last capture wins — the freshest marker is the active session.
    let caps = MARKER_RE.captures_iter(text).last()?;
    let captured = caps.get(1)?.as_str();
    SessionId::from_raw(captured).ok()
}

/// Scan the trailing [`WINDOW`] string messages for the freshest marker.
///
/// Iteration runs newest-to-oldest so the first hit wins; an older marker
/// buried further back in history is never returned when a newer one exists.
pub fn detect(messages: &[String]) -> Option<SessionId> {
    let start = messages.len().saturating_sub(WINDOW);
    for msg in messages[start..].iter().rev() {
        if let Some(id) = detect_in_text(msg) {
            return Some(id);
        }
    }
    None
}

/// Scan OpenAI-shaped `messages` (`[{"role":..., "content":...}]`) of any role
/// for a marker.
///
/// `content` may be a plain string or a multipart list of `{"type":"text",
/// "text": ...}` parts; both shapes are flattened into text before scanning
/// via the shared [`flatten_text`] helper. Iteration runs newest-to-oldest over
/// the trailing [`WINDOW`] messages so the first hit wins — identical semantics
/// to [`detect`]. This is the Rust port's entry point for the request pipeline
/// (§3 step 2): the proxy hands in the raw `serde_json::Value` messages array
/// straight off the wire.
pub fn detect_from_messages(messages: &[Value]) -> Option<SessionId> {
    let start = messages.len().saturating_sub(WINDOW);
    for msg in messages[start..].iter().rev() {
        if let Some(content) = msg.get("content")
            && let Some(id) = detect_in_text(&flatten_text(content))
        {
            return Some(id);
        }
    }
    None
}

/// Typed-message counterpart of [`detect_from_messages`].
///
/// Operates on the [`EnrichmentMessages`] array the request pipeline uses
/// internally (built via `enrichment_messages_from_json`); same trailing
/// [`WINDOW`] scan, same newest-to-oldest iteration, same first-hit-wins
/// semantics. The typed path avoids a per-message `serde_json::Value`
/// lookup, which matters because the marker scan runs on every request.
pub fn detect_from_typed_messages(messages: &EnrichmentMessages) -> Option<SessionId> {
    let start = messages.len().saturating_sub(WINDOW);
    for msg in messages[start..].iter().rev() {
        let id = detect_in_text(&msg.content.as_text());
        if id.is_some() {
            return id;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(n: u8) -> SessionId {
        SessionId::from_raw(&format!("sess_{:012x}", n as u64)).unwrap()
    }

    #[test]
    fn detect_in_text_finds_well_formed_marker() {
        let id = sid(2);
        let text = format!("hello world\n<!-- smos:{} -->", id.as_str());
        assert_eq!(detect_in_text(&text), Some(id));
    }

    #[test]
    fn detect_in_text_returns_none_when_missing() {
        assert!(detect_in_text("just text").is_none());
    }

    #[test]
    fn detect_in_text_returns_none_when_session_id_is_malformed() {
        assert!(detect_in_text("<!-- smos:not-a-session-id -->").is_none());
    }

    #[test]
    fn detect_in_text_returns_newest_when_multiple_markers_present() {
        let older = sid(2);
        let newer = sid(3);
        let text = format!(
            "<!-- smos:{} --> middle <!-- smos:{} -->",
            older.as_str(),
            newer.as_str()
        );
        assert_eq!(detect_in_text(&text), Some(newer));
    }

    #[test]
    fn detect_returns_none_for_empty_messages() {
        assert!(detect(&[]).is_none());
    }

    #[test]
    fn detect_returns_none_when_no_message_has_marker() {
        let msgs = vec!["hi".to_string(), "there".to_string()];
        assert!(detect(&msgs).is_none());
    }

    #[test]
    fn detect_scans_last_20_messages_and_picks_newest() {
        let target = sid(7);
        let mut msgs: Vec<String> = (0..25).map(|i| format!("msg {i}")).collect();
        msgs[18] = format!("text <!-- smos:{} -->", target.as_str());
        assert_eq!(detect(&msgs), Some(target));
    }

    #[test]
    fn detect_ignores_marker_outside_window() {
        let stale = sid(1);
        let mut msgs: Vec<String> = (0..25).map(|i| format!("msg {i}")).collect();
        msgs[0] = format!("stale <!-- smos:{} -->", stale.as_str());
        assert!(detect(&msgs).is_none());
    }

    #[test]
    fn detect_from_messages_finds_marker_in_string_content() {
        let id = sid(9);
        let messages = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({
                "role": "assistant",
                "content": format!("hello\n<!-- smos:{} -->", id.as_str()),
            }),
        ];
        assert_eq!(detect_from_messages(&messages), Some(id));
    }

    #[test]
    fn detect_from_messages_finds_marker_in_multipart_text_part() {
        let id = sid(4);
        let messages = vec![serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "text", "text": format!("sure\n<!-- smos:{} -->", id.as_str())},
                {"type": "image_url", "image_url": {"url": "data:..."}}
            ]
        })];
        assert_eq!(detect_from_messages(&messages), Some(id));
    }

    #[test]
    fn detect_from_messages_returns_none_without_marker() {
        let messages = vec![serde_json::json!({"role": "user", "content": "plain"})];
        assert!(detect_from_messages(&messages).is_none());
    }

    #[test]
    fn detect_from_messages_scans_only_trailing_window() {
        let stale = sid(2);
        let fresh = sid(3);
        let mut messages: Vec<Value> = (0..25)
            .map(|i| serde_json::json!({"role": "user", "content": format!("m{i}")}))
            .collect();
        messages[0] = serde_json::json!({
            "role": "assistant",
            "content": format!("old <!-- smos:{} -->", stale.as_str())
        });
        messages[24] = serde_json::json!({
            "role": "assistant",
            "content": format!("new <!-- smos:{} -->", fresh.as_str())
        });
        assert_eq!(detect_from_messages(&messages), Some(fresh));
    }
}
