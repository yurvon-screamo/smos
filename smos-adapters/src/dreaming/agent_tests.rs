//! Unit tests for [`super::agent`]. Kept in a separate file so the impl
//! file stays under the workspace's 200-line limit.

use super::*;

#[test]
fn audit_trigger_prompt_mentions_every_required_step() {
    let p = AUDIT_TRIGGER_PROMPT;
    assert!(
        p.contains("list_memory_keys"),
        "trigger prompt must instruct the LLM to call list_memory_keys first"
    );
    assert!(p.contains("count_facts"));
    assert!(p.contains("delete"));
    assert!(p.contains("merge"));
    assert!(p.contains("flag"));
    assert!(p.contains("write_report"));
}

#[test]
fn system_prompt_documents_list_memory_keys_as_step_zero() {
    let p = super::prompts::SYSTEM_PROMPT;
    assert!(
        p.contains("list_memory_keys"),
        "system prompt must mention list_memory_keys so the LLM can discover namespaces"
    );
}

#[test]
fn local_audit_base_url_appends_v1_to_host_root() {
    assert_eq!(
        local_audit_base_url("http://localhost:28082"),
        "http://localhost:28082/v1"
    );
}

#[test]
fn local_audit_base_url_trims_trailing_slash_before_appending_v1() {
    assert_eq!(
        local_audit_base_url("http://localhost:28082/"),
        "http://localhost:28082/v1",
        "a trailing slash must not produce a double-slash (`//v1`)"
    );
    assert_eq!(
        local_audit_base_url("http://localhost:28082///"),
        "http://localhost:28082/v1",
        "multiple trailing slashes collapse to one separator"
    );
}
