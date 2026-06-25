use super::*;

#[test]
fn format_tool_calls_renders_name_and_arguments() {
    let calls = vec![ToolCall {
        name: "read_file".into(),
        arguments: smos_domain::chat::ToolArguments::from_json(r#"{"path":"auth.rs"}"#),
    }];
    assert_eq!(
        format_tool_calls(&calls),
        "\n\nTool calls:\n- read_file({\"path\":\"auth.rs\"})"
    );
}

#[test]
fn format_tool_calls_empty_returns_empty_string() {
    assert_eq!(format_tool_calls(&[]), "");
}
