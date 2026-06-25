//! Unit tests for [`super::agent`]. Kept in a separate file so the impl
//! file stays under the workspace's 200-line limit.

use std::time::{Duration, Instant};

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

// Regression (B5): a hung LLM upstream must NOT block the scheduler forever.
// The pre-B5 path awaited `agent.prompt(...)` directly, so a stuck upstream
// would wedge the cron tick. The new path wraps the prompt in
// `prompt_with_budget`, which aborts after `budget` elapses and returns
// `Err(AuditTimeout)`. The test pins the wall-clock cap with a future that
// NEVER resolves (`std::future::pending`) so the only way the function
// returns inside the budget is the timeout branch.
#[tokio::test]
async fn prompt_with_budget_aborts_hanging_prompt_within_budget() {
    let budget = Duration::from_millis(150);
    let started = Instant::now();
    let result: Result<String, AuditTimeout> =
        prompt_with_budget(std::future::pending(), budget).await;

    let elapsed = started.elapsed();
    assert!(
        matches!(result, Err(AuditTimeout { .. })),
        "a never-resolving future must surface as AuditTimeout"
    );
    assert!(
        elapsed >= budget,
        "timeout must respect the budget (elapsed = {elapsed:?}, budget = {budget:?})"
    );
    assert!(
        elapsed < budget * 5,
        "timeout must fire promptly, not at the tokio scheduler's mercy (elapsed = {elapsed:?})"
    );
}

// Regression (B5): a prompt that resolves in time is forwarded unchanged —
// the budget wrapper is transparent on the happy path (no spurious
// timeout, value passed through verbatim).
#[tokio::test]
async fn prompt_with_budget_forwards_value_when_prompt_resolves_in_time() {
    let budget = Duration::from_secs(5);
    let result: Result<&str, AuditTimeout> = prompt_with_budget(async { "ok" }, budget).await;
    assert_eq!(result.unwrap(), "ok");
}
