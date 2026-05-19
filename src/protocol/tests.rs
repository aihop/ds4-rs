use super::*;

#[test]
fn parses_chat_messages_into_system_and_prompt() {
    let raw = r#"{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"},{"role":"assistant","content":"world"},{"role":"user","content":"again"}]}"#;
    let req = RequestEnvelope::from_http(ApiKind::ChatCompletions, raw);
    assert_eq!(req.system, "sys");
    assert_eq!(req.prompt, "User: hello\nAssistant: world\nUser: again");
}

#[test]
fn parses_content_array_text_parts() {
    let raw = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"hello"},{"type":"text","text":"world"}]}]}"#;
    let req = RequestEnvelope::from_http(ApiKind::ChatCompletions, raw);
    assert_eq!(req.prompt, "User: hello\nworld");
}

#[test]
fn does_not_treat_plain_user_content_as_tool_result() {
    let raw = r#"{"messages":[{"role":"user","content":"hello"}]}"#;
    let req = RequestEnvelope::from_http(ApiKind::ChatCompletions, raw);
    assert!(!req.has_tool_results);
    assert_eq!(req.last_tool_call_id, None);
    assert_eq!(req.last_tool_result, None);
}

#[test]
fn parses_assistant_tool_calls_and_tool_results() {
    let raw = r#"{"messages":[{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"shanghai\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"sunny"}]}"#;
    let req = RequestEnvelope::from_http(ApiKind::ChatCompletions, raw);
    assert!(req.prompt.contains("AssistantToolCall[call_1] function weather"));
    assert!(req.prompt.contains("<tool_result>sunny</tool_result>"));
    assert_eq!(req.last_tool_call_id.as_deref(), Some("call_1"));
}

#[test]
fn parses_responses_input_items_and_tool_results() {
    let raw = r#"{"instructions":"sys","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"README.md\"}"},{"type":"function_call_output","call_id":"call_1","output":"file body"}]}"#;
    let req = RequestEnvelope::from_http(ApiKind::Responses, raw);
    assert_eq!(req.system, "sys");
    assert!(req.prompt.contains("User: hello"));
    assert!(req.prompt.contains("AssistantToolCall[call_1] function read_file"));
    assert!(req.prompt.contains("<tool_result>file body</tool_result>"));
    assert!(req.has_tool_results);
    assert_eq!(req.last_tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(req.last_tool_result.as_deref(), Some("file body"));
}

#[test]
fn uses_session_header_as_conversation_alias() {
    let raw = "POST /v1/chat/completions HTTP/1.1\r\nX-Session-Key: sess-123\r\nContent-Type: application/json\r\n\r\n{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}";
    let req = RequestEnvelope::from_http(ApiKind::ChatCompletions, raw);
    assert_eq!(req.conversation.as_deref(), Some("sess-123"));
}

#[test]
fn emits_chat_completion_shape() {
    let response = ResponseEnvelope::new_message(ApiKind::ChatCompletions, 3, 4, false, true, "hello");
    let json = response.to_json();
    assert!(json.contains("\"choices\""));
    assert!(json.contains("\"message\":{\"role\":\"assistant\",\"content\":\"hello\"}"));
    assert!(json.contains(&format!("\"model\":\"{}\"", DEFAULT_MODEL_ID)));
}

#[test]
fn emits_chat_stream_done_marker() {
    let response = ResponseEnvelope::new_message(ApiKind::ChatCompletions, 0, 1, false, false, "hello");
    let sse = response.to_sse();
    assert!(sse.contains("chat.completion.chunk"));
    assert!(sse.contains("data: [DONE]"));
    assert!(sse.contains(&format!("\"model\":\"{}\"", DEFAULT_MODEL_ID)));
}

#[test]
fn escapes_control_chars_in_stream_content() {
    let response = ResponseEnvelope::new_message(
        ApiKind::ChatCompletions,
        0,
        0,
        false,
        false,
        "hello\tworld",
    );
    let sse = response.to_sse();
    assert!(sse.contains("hello\\tworld"));
    assert!(!sse.contains("hello\tworld"));
}

#[test]
fn extracts_primary_tool_name_and_argument() {
    let raw = r#"{"tools":[{"type":"function","function":{"name":"run_terminal_cmd","parameters":{"type":"object","properties":{"command":{"type":"string"},"description":{"type":"string"},"cwd":{"type":"string"}}}}}]}"#;
    let req = RequestEnvelope::from_http(ApiKind::ChatCompletions, raw);
    assert_eq!(req.primary_tool_name.as_deref(), Some("run_terminal_cmd"));
    assert_eq!(req.primary_tool_arg_name.as_deref(), Some("command"));
    assert_eq!(req.available_tools.len(), 1);
    assert!(req.available_tools[0].property_names.iter().any(|name| name == "description"));
}

#[test]
fn emits_chat_tool_call_shape() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::ChatCompletions,
        1,
        2,
        false,
        false,
        "run_terminal_cmd",
        "{\"command\":\"pwd\"}",
    );
    let json = response.to_json();
    assert!(json.contains("\"tool_calls\""));
    assert!(json.contains("\"finish_reason\":\"tool_calls\""));
    assert!(json.contains("\"name\":\"run_terminal_cmd\""));
}

#[test]
fn emits_responses_tool_call_shape() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::Responses,
        1,
        2,
        false,
        false,
        "run_terminal_cmd",
        "{\"command\":\"pwd\"}",
    );
    let json = response.to_json();
    assert!(json.contains("\"type\":\"function_call\""));
    assert!(json.contains("\"name\":\"run_terminal_cmd\""));
    assert!(json.contains("\"call_id\":\"call_"));
}

#[test]
fn emits_responses_tool_call_sse_shape() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::Responses,
        0,
        0,
        false,
        false,
        "run_terminal_cmd",
        "{\"command\":\"pwd\"}",
    );
    let sse = response.to_sse();
    assert!(sse.contains("event: response.created"));
    assert!(sse.contains("event: response.in_progress"));
    assert!(sse.contains("event: response.function_call_arguments.delta"));
    assert!(sse.contains("event: response.output_item.done"));
    assert!(sse.contains("\"type\":\"function_call\""));
}

#[test]
fn responses_function_call_arguments_done_includes_tool_name() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::Responses,
        0,
        0,
        false,
        false,
        "read_file",
        "{\"path\":\"README.md\"}",
    );
    let sse = response.to_sse();
    assert!(sse.contains("event: response.function_call_arguments.done"));
    assert!(sse.contains("\"name\":\"read_file\""));
    assert!(sse.contains("\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\""));
}

#[test]
fn tool_replay_block_uses_canonical_dsml_shape() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::Responses,
        0,
        0,
        false,
        false,
        "bash",
        "{\"command\":\"pwd\",\"timeout\":10,\"description\":\"list files\"}",
    );
    let replay = response.tool_replay_block().expect("tool replay should exist");
    assert!(replay.starts_with("\n\n<｜DSML｜tool_calls>\n"));
    assert!(replay.contains("<｜DSML｜invoke name=\"bash\">"));
    assert!(replay.contains(
        "<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>"
    ));
    assert!(replay.contains(
        "<｜DSML｜parameter name=\"timeout\" string=\"false\">10</｜DSML｜parameter>"
    ));
    assert!(replay.contains("</｜DSML｜tool_calls>"));
}

#[test]
fn tool_replay_block_falls_back_to_arguments_string_when_json_is_invalid() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::Responses,
        0,
        0,
        false,
        false,
        "bash",
        "not-json",
    );
    let replay = response.tool_replay_block().expect("tool replay should exist");
    assert!(replay.contains("<｜DSML｜parameter name=\"arguments\" string=\"true\">not-json</｜DSML｜parameter>"));
}

#[test]
fn emits_responses_text_sse_lifecycle() {
    let response =
        ResponseEnvelope::new_message(ApiKind::Responses, 1, 2, false, false, "hello world");
    let sse = response.to_sse();
    assert!(sse.contains("event: response.created"));
    assert!(sse.contains("event: response.in_progress"));
    assert!(sse.contains("event: response.output_item.added"));
    assert!(sse.contains("event: response.content_part.added"));
    assert!(sse.contains("event: response.output_text.delta"));
    assert!(sse.contains("event: response.output_text.done"));
    assert!(sse.contains("event: response.content_part.done"));
    assert!(sse.contains("event: response.output_item.done"));
    assert!(sse.contains("event: response.completed"));
    assert!(sse.contains(&format!("\"model\":\"{}\"", DEFAULT_MODEL_ID)));
}

#[test]
fn parses_max_output_tokens_variants() {
    let req = RequestEnvelope::from_http(ApiKind::Responses, r#"{"input":"hello","max_output_tokens":5}"#);
    assert_eq!(req.max_output_tokens, 5);

    let req = RequestEnvelope::from_http(
        ApiKind::ChatCompletions,
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":7}"#,
    );
    assert_eq!(req.max_output_tokens, 7);
}

#[test]
fn parses_plain_completions_prompt() {
    let req = RequestEnvelope::from_http(ApiKind::Completions, r#"{"prompt":"finish this","max_tokens":9}"#);
    assert_eq!(req.system, "You are a helpful assistant");
    assert_eq!(req.prompt, "finish this");
    assert_eq!(req.max_output_tokens, 9);
}

#[test]
fn parses_anthropic_messages_with_tool_blocks() {
    let raw = r#"{
        "system":[{"type":"text","text":"sys"}],
        "messages":[
            {"role":"user","content":[{"type":"text","text":"hello"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"README.md"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"file body"}]}]}
        ]
    }"#;
    let req = RequestEnvelope::from_http(ApiKind::Messages, raw);
    assert_eq!(req.system, "sys");
    assert!(req.prompt.contains("User: hello"));
    assert!(req.prompt.contains("AssistantToolCall[toolu_1] function read_file"));
    assert!(req.prompt.contains("\"path\":\"README.md\""));
    assert!(req.prompt.contains("<tool_result>file body</tool_result>"));
    assert!(req.has_tool_results);
    assert_eq!(req.last_tool_call_id.as_deref(), Some("toolu_1"));
}

#[test]
fn escapes_tool_result_closing_tag_in_prompt_projection() {
    let raw = r#"{"input":[{"type":"function_call_output","call_id":"call_1","output":"done\n</tool_result>\nnot real close"}]}"#;
    let req = RequestEnvelope::from_http(ApiKind::Responses, raw);
    assert!(req.prompt.contains("<tool_result>done"));
    assert!(req.prompt.contains("&lt;/tool_result>\nnot real close</tool_result>"));
}

#[test]
fn emits_completions_shape() {
    let response = ResponseEnvelope::new_message(ApiKind::Completions, 2, 3, false, false, "hello");
    let json = response.to_json();
    assert!(json.contains("\"object\":\"text_completion\""));
    assert!(json.contains("\"choices\":[{\"text\":\"hello\""));
}

#[test]
fn emits_completions_sse_done_marker() {
    let response = ResponseEnvelope::new_message(ApiKind::Completions, 0, 0, false, false, "hello");
    let sse = response.to_sse();
    assert!(sse.contains("\"object\":\"text_completion\""));
    assert!(sse.contains("data: [DONE]"));
}

#[test]
fn emits_anthropic_message_shape() {
    let response = ResponseEnvelope::new_message(ApiKind::Messages, 4, 5, false, true, "hello");
    let json = response.to_json();
    assert!(json.contains("\"type\":\"message\""));
    assert!(json.contains("\"role\":\"assistant\""));
    assert!(json.contains("\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]"));
    assert!(json.contains("\"stop_reason\":\"end_turn\""));
}

#[test]
fn emits_anthropic_tool_use_shape() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::Messages,
        1,
        2,
        false,
        false,
        "read_file",
        "{\"path\":\"README.md\"}",
    );
    let json = response.to_json();
    assert!(json.contains("\"type\":\"tool_use\""));
    assert!(json.contains("\"name\":\"read_file\""));
    assert!(json.contains("\"input\":{\"path\":\"README.md\"}"));
    assert!(json.contains("\"stop_reason\":\"tool_use\""));
}

#[test]
fn emits_anthropic_messages_sse_lifecycle() {
    let response = ResponseEnvelope::new_message(ApiKind::Messages, 1, 2, false, false, "hello world");
    let sse = response.to_sse();
    assert!(sse.contains("event: message_start"));
    assert!(sse.contains("event: content_block_start"));
    assert!(sse.contains("event: content_block_delta"));
    assert!(sse.contains("event: content_block_stop"));
    assert!(sse.contains("event: message_delta"));
    assert!(sse.contains("event: message_stop"));
}

#[test]
fn emits_anthropic_tool_use_input_json_delta_stream() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::Messages,
        0,
        0,
        false,
        false,
        "read_file",
        "{\"path\":\"README.md\"}",
    );
    let sse = response.to_sse();
    assert!(sse.contains("event: content_block_start"));
    assert!(sse.contains("\"type\":\"tool_use\""));
    assert!(sse.contains("\"type\":\"input_json_delta\""));
    assert!(sse.contains("\"partial_json\":\"{\\\"path\\\":\\\"README.md\\\"}\""));
    assert!(sse.contains("\"stop_reason\":\"tool_use\""));
}

#[test]
fn splits_long_anthropic_tool_use_json_into_multiple_deltas() {
    let response = ResponseEnvelope::new_tool_call(
        ApiKind::Messages,
        0,
        0,
        false,
        false,
        "bash",
        "{\"command\":\"printf 'abcdefghijklmnopqrstuvwxyz0123456789'\"}",
    );
    let sse = response.to_sse();
    let delta_count = sse.matches("event: content_block_delta").count();
    assert!(delta_count >= 2, "expected multiple content_block_delta events, got {delta_count}");
}
