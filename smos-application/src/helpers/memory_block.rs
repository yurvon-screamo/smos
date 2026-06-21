//! Memory block builder — render the `<smos-memory>` injection block (§3 step 8).
//!
//! The block carries:
//! - An opening tag with the active `session_id`.
//! - Zero or more `[fact_id] document` lines.
//! - A closing tag.
//!
//! Persona lines are intentionally NOT rendered here: the persona lives behind
//! an IO boundary (adapter reads `memory_key/persona.md`). Slice 1 builds the
//! facts-only block; the adapter layer will prepend persona lines when present.

use smos_domain::{FactId, MemoryKey, SessionId};

/// One fact to render in the memory block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBlockEntry<'a> {
    pub id: &'a FactId,
    pub document: &'a str,
}

/// Build the `<smos-memory>` block.
///
/// Format:
/// ```text
/// <smos-memory session="sess_...">
/// [fact_xxx] document text
/// [fact_yyy] document text
/// </smos-memory>
/// ```
///
/// `memory_key` is accepted for forward compatibility with the persona
/// (`[persona-...]` lines) but not currently rendered; slice 1 emits facts only.
pub fn build<'a>(
    facts: impl IntoIterator<Item = MemoryBlockEntry<'a>>,
    session_id: &SessionId,
    _memory_key: &MemoryKey,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("<smos-memory session=\"{}\">", session_id.as_str()));
    for entry in facts {
        lines.push(format!("[{}] {}", entry.id.as_str(), entry.document));
    }
    lines.push("</smos-memory>".to_string());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid() -> SessionId {
        SessionId::from_raw("sess_abcdef012345").unwrap()
    }

    fn key() -> MemoryKey {
        MemoryKey::from_raw("origa").unwrap()
    }

    fn fid(content: &str) -> FactId {
        FactId::from_content(content)
    }

    #[test]
    fn empty_facts_emits_open_and_close_tags() {
        let block = build([], &sid(), &key());
        assert_eq!(
            block,
            "<smos-memory session=\"sess_abcdef012345\">\n</smos-memory>"
        );
    }

    #[test]
    fn each_fact_gets_one_line_with_id_and_document() {
        let id1 = fid("first fact");
        let id2 = fid("second fact");
        let facts = vec![
            MemoryBlockEntry {
                id: &id1,
                document: "First fact text",
            },
            MemoryBlockEntry {
                id: &id2,
                document: "Second fact text",
            },
        ];
        let block = build(facts, &sid(), &key());
        let lines: Vec<&str> = block.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[1].starts_with(&format!("[{}]", id1.as_str())));
        assert!(lines[1].contains("First fact text"));
        assert!(lines[2].starts_with(&format!("[{}]", id2.as_str())));
        assert!(lines[2].contains("Second fact text"));
        assert_eq!(lines[3], "</smos-memory>");
    }

    #[test]
    fn opening_tag_carries_session_id_attribute() {
        let block = build([], &sid(), &key());
        assert!(block.starts_with("<smos-memory session=\"sess_abcdef012345\">"));
    }

    #[test]
    fn closing_tag_is_emitted_last() {
        let id = fid("x");
        let facts = vec![MemoryBlockEntry {
            id: &id,
            document: "y",
        }];
        let block = build(facts, &sid(), &key());
        assert!(block.ends_with("</smos-memory>"));
    }
}
