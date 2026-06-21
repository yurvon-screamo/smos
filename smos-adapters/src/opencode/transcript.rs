//! opencode transcript parser — flatten nested `info`+`parts` into turns.
//!
//! Mirrors `smos-poc/scripts/opencode_source.py::parse_assistant_turn`. The
//! opencode export shape is:
//!
//! ```jsonc
//! {
//!   "info":   { "id": "ses_…", "title": "…", "agent": "head-of-development", … },
//!   "messages": [
//!     { "info": { "id": "msg_…", "role": "assistant", "agent": "head-of-development", … },
//!       "parts": [
//!         { "type": "text",     "text": "TTL=10 prevents refresh loop" },
//!         { "type": "reasoning","text": "Analyzing TTL configuration..." },
//!         { "type": "tool",     "tool": "read_file",
//!           "state": { "input": { "path": "auth.rs" }, "output": "..." } }
//!       ]
//!     }, …
//!   ]
//! }
//! ```
//!
//! Only `role == "assistant"` messages produce a turn. `text` parts feed the
//! visible content; `tool` parts feed the structured tool_calls. Reasoning and
//! other part types are silently dropped — matching the POC (the live response
//! pipeline also extracts only from visible content).

use serde_json::Value;

use smos_application::use_cases::import_opencode_session::AssistantTurn;
use smos_domain::chat::{ToolArguments, ToolCall};

/// Parse an opencode export transcript into flattened assistant turns.
///
/// Returns an empty `Vec` when `transcript` carries no `messages` array (the
/// CLI export of an unknown session id, an HTTP 200 with an empty body, …).
/// Per-message parse failures are skipped silently AT THE PARSER LEVEL but
/// logged at `debug` level with the reason, so bulk imports that drop
/// malformed messages surface in operator logs without aborting the whole
/// import. The `turns_processed` vs `turns_skipped` counters in
/// [`ImportStats`] then distinguish "filtered by user policy" from "dropped
/// by the parser" by comparing the parsed turn count to the import stats.
pub fn parse_transcript(transcript: &Value) -> Vec<AssistantTurn> {
    let Some(messages) = transcript.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut turns = Vec::with_capacity(messages.len());
    for (idx, message) in messages.iter().enumerate() {
        match parse_message(message) {
            Some(turn) => turns.push(turn),
            None => {
                // Log a short reason so bulk imports (where dozens of
                // malformed messages would otherwise vanish silently) stay
                // observable. The reason is the FIRST structural field that
                // failed the parser — enough to diagnose without dumping the
                // whole message.
                let reason = classify_skip(message);
                tracing::debug!(
                    message_index = idx,
                    reason = %reason,
                    "transcript message skipped during parse"
                );
            }
        }
    }
    turns
}

/// Classify why a message was dropped by the parser — for observability only.
///
/// Mirrors the structural checks [`parse_message`] performs. The two functions
/// are NOT auto-synced: a new skip condition added to `parse_message` must be
/// reflected here by hand, otherwise the logged reason will be silently wrong.
/// Kept inline (not shared) because `parse_message` returns `Option<AssistantTurn>`
/// while this function returns a reason label — factoring the common predicates
/// out would force a third shape that does not fit either caller.
fn classify_skip(message: &Value) -> &'static str {
    if message.get("info").is_none() {
        return "missing info";
    }
    let role = message
        .get("info")
        .and_then(|i| i.get("role"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if !role.eq_ignore_ascii_case("assistant") {
        return "non-assistant role";
    }
    if message.get("parts").and_then(Value::as_array).is_none() {
        return "missing parts array";
    }
    "empty turn (no text and no tool calls)"
}

/// Flatten one opencode message into a turn, or `None` if the message is not a
/// usable assistant turn.
fn parse_message(message: &Value) -> Option<AssistantTurn> {
    let info = message.get("info")?;
    let role = info.get("role").and_then(Value::as_str).unwrap_or("");
    if !role.eq_ignore_ascii_case("assistant") {
        return None;
    }

    let parts = message.get("parts").and_then(Value::as_array)?;
    let (mut text_chunks, mut tool_calls) = (Vec::new(), Vec::new());
    for part in parts {
        let ptype = part.get("type").and_then(Value::as_str).unwrap_or("");
        match ptype {
            "text" => {
                if let Some(trimmed) = part
                    .get("text")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                {
                    text_chunks.push(trimmed.to_string());
                }
            }
            "tool" => {
                if let Some(call) = parse_tool_call(part) {
                    tool_calls.push(call);
                }
            }
            // `reasoning`, `step-start`, … are intentionally ignored — they
            // carry internal model state, not user-visible content.
            _ => {}
        }
    }

    let content = text_chunks.join("\n\n");
    if content.is_empty() && tool_calls.is_empty() {
        return None;
    }

    Some(AssistantTurn {
        message_id: info
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        agent: info
            .get("agent")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        content,
        tool_calls,
    })
}

/// Build a domain [`ToolCall`] from an opencode `tool` part.
///
/// `state.input` is normally an object but opencode occasionally serialises it
/// as a JSON **string**; the string is parsed so the pipeline sees real
/// arguments instead of an empty `{}`. Anything that is neither an object nor
/// a parseable JSON object falls back to `{}` (POC `_coerce_tool_input`).
///
/// The resulting arguments are stored verbatim as a JSON-shaped string inside
/// the opaque [`ToolArguments`]; this layer keeps the `serde_json` dependency
/// that the domain deliberately avoids.
fn parse_tool_call(part: &Value) -> Option<ToolCall> {
    let name = part
        .get("tool")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .unwrap_or("tool")
        .to_string();
    let raw_input = part.get("state").and_then(|s| s.get("input"));
    let arguments = coerce_tool_input(raw_input);
    Some(ToolCall {
        name,
        arguments: ToolArguments::from_json(arguments),
    })
}

/// Normalize a `state.input` value into a JSON-shaped string suitable for the
/// opaque [`ToolArguments`].
///
/// - `Object` → re-serialised to JSON.
/// - `String` → if it parses into a JSON object, the original string is kept
///   verbatim (avoids a round-trip that would reorder keys); otherwise `{}`.
/// - anything else (missing, null, array, …) → `{}`.
fn coerce_tool_input(raw_input: Option<&Value>) -> String {
    match raw_input {
        Some(Value::Object(map)) => {
            serde_json::to_string(&Value::Object(map.clone())).unwrap_or_else(|_| "{}".to_string())
        }
        Some(Value::String(s)) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return "{}".to_string();
            }
            match serde_json::from_str::<Value>(trimmed) {
                Ok(Value::Object(_)) => trimmed.to_string(),
                _ => "{}".to_string(),
            }
        }
        _ => "{}".to_string(),
    }
}

#[cfg(test)]
mod tests {
    //! Parser unit tests — pure data in, pure data out. The full
    //! parse → import → store pipeline is exercised by `tests/e2e_import.rs`.

    use super::*;
    use serde_json::json;

    #[test]
    fn parse_assistant_with_text_and_tool() {
        let transcript = json!({
            "info": {"id": "ses_1", "title": "Test"},
            "messages": [{
                "info": {"id": "msg_1", "role": "assistant", "agent": "head-of-development"},
                "parts": [
                    {"type": "text", "text": "TTL=10 prevents refresh loop"},
                    {"type": "tool", "tool": "read_file", "state": {"input": {"path": "auth.rs"}}}
                ]
            }]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].message_id, "msg_1");
        assert_eq!(turns[0].agent, "head-of-development");
        assert_eq!(turns[0].content, "TTL=10 prevents refresh loop");
        assert_eq!(turns[0].tool_calls.len(), 1);
        assert_eq!(turns[0].tool_calls[0].name, "read_file");
        assert_eq!(
            turns[0].tool_calls[0].arguments.as_str(),
            r#"{"path":"auth.rs"}"#
        );
    }

    #[test]
    fn skip_non_assistant_messages() {
        let transcript = json!({
            "messages": [
                {"info": {"role": "user"}, "parts": [{"type": "text", "text": "hello"}]},
                {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "hi"}]}
            ]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].content, "hi");
    }

    #[test]
    fn skip_empty_turn() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": []
            }]
        });
        assert!(parse_transcript(&transcript).is_empty());
    }

    #[test]
    fn multiple_text_parts_joined_with_double_newline() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "text", "text": "First"},
                    {"type": "text", "text": "Second"}
                ]
            }]
        });
        assert_eq!(parse_transcript(&transcript)[0].content, "First\n\nSecond");
    }

    #[test]
    fn tool_call_state_input_string_is_parsed_into_object() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "tool", "tool": "bash", "state": {"input": "{\"cmd\":\"ls\"}"}}
                ]
            }]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns[0].tool_calls[0].name, "bash");
        assert_eq!(turns[0].tool_calls[0].arguments.as_str(), r#"{"cmd":"ls"}"#);
    }

    #[test]
    fn tool_call_state_input_unparseable_string_falls_back_to_empty_object() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "tool", "tool": "bash", "state": {"input": "not json"}}
                ]
            }]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns[0].tool_calls[0].arguments.as_str(), "{}");
    }

    #[test]
    fn tool_call_without_state_input_yields_empty_object() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "tool", "tool": "bash"}
                ]
            }]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns[0].tool_calls[0].arguments.as_str(), "{}");
    }

    #[test]
    fn tool_call_with_empty_tool_name_defaults_to_tool() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "tool", "tool": "", "state": {"input": {"x": 1}}}
                ]
            }]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns[0].tool_calls[0].name, "tool");
    }

    #[test]
    fn ignores_reasoning_and_unknown_parts() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "reasoning", "text": "internal thought"},
                    {"type": "step-start"},
                    {"type": "text", "text": "visible answer"}
                ]
            }]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns[0].content, "visible answer");
        assert!(turns[0].tool_calls.is_empty());
    }

    #[test]
    fn missing_messages_array_returns_empty() {
        assert!(parse_transcript(&json!({"info": {}})).is_empty());
        assert!(parse_transcript(&json!({})).is_empty());
    }

    #[test]
    fn missing_agent_defaults_to_unknown() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [{"type": "text", "text": "no agent here at all"}]
            }]
        });
        assert_eq!(parse_transcript(&transcript)[0].agent, "unknown");
    }

    #[test]
    fn role_case_insensitive_match() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "ASSISTANT", "agent": "hod"},
                "parts": [{"type": "text", "text": "case folded role"}]
            }]
        });
        assert_eq!(parse_transcript(&transcript).len(), 1);
    }

    #[test]
    fn tool_only_turn_without_text_is_kept() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "tool", "tool": "read_file", "state": {"input": {"path": "x.rs"}}}
                ]
            }]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns.len(), 1);
        assert!(turns[0].content.is_empty());
        assert_eq!(turns[0].tool_calls.len(), 1);
    }

    #[test]
    fn whitespace_text_parts_are_skipped() {
        let transcript = json!({
            "messages": [{
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "text", "text": "   "},
                    {"type": "text", "text": "real"}
                ]
            }]
        });
        let turns = parse_transcript(&transcript);
        assert_eq!(turns[0].content, "real");
    }
}
