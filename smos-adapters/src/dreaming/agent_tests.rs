//! Unit tests for [`super::agent`]. Kept in a separate file so the impl
//! file stays under the workspace's 200-line limit.

use super::*;

#[test]
fn audit_trigger_prompt_mentions_every_required_step() {
    let p = AUDIT_TRIGGER_PROMPT;
    assert!(p.contains("count_facts"));
    assert!(p.contains("delete"));
    assert!(p.contains("merge"));
    assert!(p.contains("flag"));
    assert!(p.contains("write_report"));
}
