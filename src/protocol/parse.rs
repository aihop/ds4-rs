use super::jsonish::{
    extract_first_property_name, extract_header_value, extract_json_bool, extract_json_string,
    extract_json_text_or_array, extract_json_usize, extract_last_json_string, extract_message_content,
    extract_nested_json_string, extract_property_names, extract_raw_json_value,
    extract_top_level_objects, join_lines,
};
use super::{
    ApiKind, ParsedChatRequest, ParsedMessagesRequest, ParsedResponsesRequest, RequestEnvelope,
    RequestTool,
};

impl RequestEnvelope {
    pub fn from_http(api: ApiKind, raw: &str) -> Self {
        let chat = match api {
            ApiKind::ChatCompletions => parse_chat_request(raw),
            ApiKind::Responses | ApiKind::Completions | ApiKind::Messages => None,
        };
        let responses = match api {
            ApiKind::Responses => parse_responses_request(raw),
            ApiKind::ChatCompletions | ApiKind::Completions | ApiKind::Messages => None,
        };
        let messages = match api {
            ApiKind::Messages => parse_messages_request(raw),
            ApiKind::ChatCompletions | ApiKind::Responses | ApiKind::Completions => None,
        };
        let (system, prompt) = match api {
            ApiKind::ChatCompletions => chat
                .as_ref()
                .map(|parsed| (parsed.system.clone(), parsed.prompt.clone()))
                .unwrap_or_else(|| {
                    (
                        extract_json_string(raw, "system")
                            .unwrap_or_else(|| "You are a helpful assistant".to_string()),
                        extract_json_string(raw, "prompt")
                            .or_else(|| extract_json_string(raw, "input"))
                            .unwrap_or_else(|| "hello".to_string()),
                    )
                }),
            ApiKind::Responses => responses
                .as_ref()
                .map(|parsed| (parsed.system.clone(), parsed.prompt.clone()))
                .unwrap_or_else(|| {
                    (
                        extract_json_string(raw, "instructions")
                            .or_else(|| extract_json_string(raw, "system"))
                            .unwrap_or_else(|| "You are a helpful assistant".to_string()),
                        extract_json_string(raw, "prompt")
                            .or_else(|| extract_json_string(raw, "input"))
                            .unwrap_or_else(|| "hello".to_string()),
                    )
                }),
            ApiKind::Completions => (
                extract_json_string(raw, "system")
                    .unwrap_or_else(|| "You are a helpful assistant".to_string()),
                extract_json_string(raw, "prompt")
                    .or_else(|| extract_json_string(raw, "input"))
                    .unwrap_or_else(|| "hello".to_string()),
            ),
            ApiKind::Messages => messages
                .as_ref()
                .map(|parsed| (parsed.system.clone(), parsed.prompt.clone()))
                .unwrap_or_else(|| {
                    (
                        extract_json_text_or_array(raw, "system")
                            .or_else(|| extract_json_string(raw, "instructions"))
                            .or_else(|| extract_json_string(raw, "system"))
                            .unwrap_or_else(|| "You are a helpful assistant".to_string()),
                        extract_json_string(raw, "prompt")
                            .or_else(|| extract_json_string(raw, "input"))
                            .unwrap_or_else(|| "hello".to_string()),
                    )
                }),
        };
        let previous_response_id = extract_json_string(raw, "previous_response_id");
        let conversation = extract_json_string(raw, "conversation")
            .or_else(|| extract_header_value(raw, "X-Session-Key"))
            .or_else(|| extract_header_value(raw, "X-Conversation-Key"))
            .or_else(|| extract_header_value(raw, "OpenAI-Conversation-ID"))
            .or_else(|| extract_header_value(raw, "Anthropic-Conversation-ID"));
        let available_tools = extract_tools(raw);
        let has_tools = !available_tools.is_empty();
        let (primary_tool_name, primary_tool_arg_name) = available_tools
            .first()
            .cloned()
            .map(|tool| (Some(tool.name), tool.first_arg_name))
            .unwrap_or((None, None));
        let raw_last_tool_call_id = extract_last_json_string(raw, "tool_call_id");
        let raw_last_content = extract_last_json_string(raw, "content");
        let raw_has_tool_results = raw.contains("\"role\":\"tool\"") || raw.contains("\"tool_call_id\"");
        let raw_last_tool_result = if raw_has_tool_results {
            raw_last_content.clone()
        } else {
            None
        };
        let (last_tool_call_id, has_tool_results, last_tool_result) = chat
            .map(|parsed| {
                (
                    parsed.last_tool_call_id.or_else(|| raw_last_tool_call_id.clone()),
                    parsed.has_tool_results || raw_has_tool_results,
                    parsed.last_tool_result.or_else(|| raw_last_tool_result.clone()),
                )
            })
            .or_else(|| {
                responses.as_ref().map(|parsed| {
                    (
                        parsed.last_tool_call_id.clone().or_else(|| raw_last_tool_call_id.clone()),
                        parsed.has_tool_results || raw_has_tool_results,
                        parsed.last_tool_result.clone().or_else(|| raw_last_tool_result.clone()),
                    )
                })
            })
            .or_else(|| {
                messages.as_ref().map(|parsed| {
                    (
                        parsed.last_tool_call_id.clone().or_else(|| raw_last_tool_call_id.clone()),
                        parsed.has_tool_results || raw_has_tool_results,
                        parsed.last_tool_result.clone().or_else(|| raw_last_tool_result.clone()),
                    )
                })
            })
            .unwrap_or((raw_last_tool_call_id, raw_has_tool_results, raw_last_tool_result));
        let stream = extract_json_bool(raw, "stream").unwrap_or(false);
        let max_output_tokens = extract_json_usize(raw, "max_output_tokens")
            .or_else(|| extract_json_usize(raw, "max_completion_tokens"))
            .or_else(|| extract_json_usize(raw, "max_tokens"))
            .unwrap_or(0);
        Self {
            api,
            system,
            prompt,
            previous_response_id,
            conversation,
            available_tools,
            has_tools,
            has_tool_results,
            primary_tool_name,
            primary_tool_arg_name,
            last_tool_call_id,
            last_tool_result,
            stream,
            max_output_tokens,
        }
    }
}

pub(super) fn parse_chat_request(raw: &str) -> Option<ParsedChatRequest> {
    let mut system_parts = Vec::new();
    let mut prompt_parts = Vec::new();
    let mut last_tool_call_id = None;
    let mut has_tool_results = false;
    let mut last_tool_result = None;
    for message in extract_top_level_objects(raw, "messages")? {
        let role = extract_json_string(message, "role")?;
        let content = extract_message_content(message).unwrap_or_default();
        match role.as_str() {
            "system" | "developer" => {
                if !content.is_empty() {
                    system_parts.push(content);
                }
            }
            "user" => {
                if !content.is_empty() {
                    prompt_parts.push(format!("User: {content}"));
                }
            }
            "assistant" => {
                if !content.is_empty() {
                    prompt_parts.push(format!("Assistant: {content}"));
                }
                for tool_call in extract_top_level_objects(message, "tool_calls").unwrap_or_default() {
                    let call_id =
                        extract_json_string(tool_call, "id").unwrap_or_else(|| "call_unknown".to_string());
                    let call_type = extract_json_string(tool_call, "type")
                        .unwrap_or_else(|| "function".to_string());
                    let name = extract_json_string(tool_call, "name")
                        .or_else(|| extract_json_string(tool_call, "function"))
                        .or_else(|| extract_nested_json_string(tool_call, "function", "name"))
                        .unwrap_or_else(|| "tool".to_string());
                    let arguments = extract_json_string(tool_call, "arguments")
                        .or_else(|| extract_nested_json_string(tool_call, "function", "arguments"))
                        .unwrap_or_default();
                    prompt_parts.push(format!(
                        "AssistantToolCall[{call_id}] {call_type} {name}({arguments})"
                    ));
                    last_tool_call_id = Some(call_id);
                }
            }
            "tool" => {
                let tool_call_id =
                    extract_json_string(message, "tool_call_id").unwrap_or_else(|| "call_unknown".to_string());
                if !content.is_empty() {
                    prompt_parts.push(render_tool_result_text(&content));
                    last_tool_result = Some(content.clone());
                }
                last_tool_call_id = Some(tool_call_id);
                has_tool_results = true;
            }
            _ => {}
        }
    }
    if system_parts.is_empty() && prompt_parts.is_empty() {
        return None;
    }
    Some(ParsedChatRequest {
        system: join_lines(&system_parts).unwrap_or_else(|| "You are a helpful assistant".to_string()),
        prompt: join_lines(&prompt_parts).unwrap_or_else(|| "hello".to_string()),
        last_tool_call_id,
        has_tool_results,
        last_tool_result,
    })
}

pub(super) fn parse_responses_request(raw: &str) -> Option<ParsedResponsesRequest> {
    let mut system_parts = Vec::new();
    let mut prompt_parts = Vec::new();
    let mut last_tool_call_id = None;
    let mut has_tool_results = false;
    let mut last_tool_result = None;

    if let Some(instructions) = extract_json_string(raw, "instructions") {
        if !instructions.is_empty() {
            system_parts.push(instructions);
        }
    }

    if let Some(items) = extract_top_level_objects(raw, "input") {
        for item in items {
            let item_type = extract_json_string(item, "type").unwrap_or_else(|| "message".to_string());
            match item_type.as_str() {
                "message" => {
                    let role = extract_json_string(item, "role").unwrap_or_else(|| "user".to_string());
                    let content = extract_message_content(item)
                        .or_else(|| extract_json_string(item, "text"))
                        .or_else(|| extract_json_string(item, "content"))
                        .unwrap_or_default();
                    match role.as_str() {
                        "system" | "developer" => {
                            if !content.is_empty() {
                                system_parts.push(content);
                            }
                        }
                        "assistant" => {
                            if !content.is_empty() {
                                prompt_parts.push(format!("Assistant: {content}"));
                            }
                        }
                        _ => {
                            if !content.is_empty() {
                                prompt_parts.push(format!("User: {content}"));
                            }
                        }
                    }
                }
                "function_call" | "custom_tool_call" => {
                    let call_id = extract_json_string(item, "call_id")
                        .or_else(|| extract_json_string(item, "id"))
                        .unwrap_or_else(|| "call_unknown".to_string());
                    let name = extract_json_string(item, "name")
                        .or_else(|| extract_json_string(item, "function"))
                        .unwrap_or_else(|| "tool".to_string());
                    let arguments = extract_json_string(item, "arguments")
                        .or_else(|| extract_json_string(item, "input"))
                        .unwrap_or_default();
                    prompt_parts.push(format!(
                        "AssistantToolCall[{call_id}] function {name}({arguments})"
                    ));
                    last_tool_call_id = Some(call_id);
                }
                "function_call_output" | "custom_tool_call_output" => {
                    let call_id = extract_json_string(item, "call_id")
                        .or_else(|| extract_json_string(item, "tool_call_id"))
                        .or_else(|| extract_json_string(item, "id"))
                        .unwrap_or_else(|| "call_unknown".to_string());
                    let output = extract_json_string(item, "output")
                        .or_else(|| extract_json_string(item, "content"))
                        .or_else(|| extract_json_string(item, "text"))
                        .unwrap_or_default();
                    if !output.is_empty() {
                        prompt_parts.push(render_tool_result_text(&output));
                        last_tool_result = Some(output.clone());
                    }
                    last_tool_call_id = Some(call_id);
                    has_tool_results = true;
                }
                "reasoning" => {
                    if let Some(summary) = extract_json_string(item, "summary") {
                        if !summary.is_empty() {
                            prompt_parts.push(format!("Assistant: {summary}"));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if system_parts.is_empty() && prompt_parts.is_empty() {
        let prompt = extract_json_string(raw, "prompt").or_else(|| extract_json_string(raw, "input"))?;
        return Some(ParsedResponsesRequest {
            system: extract_json_string(raw, "instructions")
                .or_else(|| extract_json_string(raw, "system"))
                .unwrap_or_else(|| "You are a helpful assistant".to_string()),
            prompt,
            last_tool_call_id,
            has_tool_results,
            last_tool_result,
        });
    }

    Some(ParsedResponsesRequest {
        system: join_lines(&system_parts).unwrap_or_else(|| "You are a helpful assistant".to_string()),
        prompt: join_lines(&prompt_parts).unwrap_or_else(|| "hello".to_string()),
        last_tool_call_id,
        has_tool_results,
        last_tool_result,
    })
}

pub(super) fn parse_messages_request(raw: &str) -> Option<ParsedMessagesRequest> {
    let mut system_parts = Vec::new();
    let mut prompt_parts = Vec::new();
    let mut last_tool_call_id = None;
    let mut has_tool_results = false;
    let mut last_tool_result = None;

    if let Some(system) = extract_json_text_or_array(raw, "system") {
        if !system.is_empty() {
            system_parts.push(system);
        }
    }

    for message in extract_top_level_objects(raw, "messages")? {
        let role = extract_json_string(message, "role").unwrap_or_else(|| "user".to_string());
        let content_blocks = extract_top_level_objects(message, "content");
        if let Some(blocks) = content_blocks {
            for block in blocks {
                let block_type = extract_json_string(block, "type").unwrap_or_else(|| "text".to_string());
                match block_type.as_str() {
                    "text" | "input_text" | "output_text" => {
                        let text = extract_json_string(block, "text")
                            .or_else(|| extract_json_string(block, "content"))
                            .unwrap_or_default();
                        if text.is_empty() {
                            continue;
                        }
                        match role.as_str() {
                            "assistant" => prompt_parts.push(format!("Assistant: {text}")),
                            "system" | "developer" => system_parts.push(text),
                            _ => prompt_parts.push(format!("User: {text}")),
                        }
                    }
                    "tool_use" => {
                        let call_id = extract_json_string(block, "id")
                            .or_else(|| extract_json_string(block, "tool_use_id"))
                            .unwrap_or_else(|| "call_unknown".to_string());
                        let name = extract_json_string(block, "name")
                            .unwrap_or_else(|| "tool".to_string());
                        let arguments =
                            extract_raw_json_value(block, "input").unwrap_or_else(|| "{}".to_string());
                        prompt_parts.push(format!(
                            "AssistantToolCall[{call_id}] function {name}({arguments})"
                        ));
                        last_tool_call_id = Some(call_id);
                    }
                    "tool_result" => {
                        let call_id = extract_json_string(block, "tool_use_id")
                            .or_else(|| extract_json_string(block, "id"))
                            .unwrap_or_else(|| "call_unknown".to_string());
                        let output = extract_json_text_or_array(block, "content")
                            .or_else(|| extract_json_string(block, "text"))
                            .unwrap_or_default();
                        if !output.is_empty() {
                            prompt_parts.push(render_tool_result_text(&output));
                            last_tool_result = Some(output.clone());
                        }
                        last_tool_call_id = Some(call_id);
                        has_tool_results = true;
                    }
                    _ => {}
                }
            }
            continue;
        }

        let content = extract_message_content(message).unwrap_or_default();
        if content.is_empty() {
            continue;
        }
        match role.as_str() {
            "assistant" => prompt_parts.push(format!("Assistant: {content}")),
            "system" | "developer" => system_parts.push(content),
            _ => prompt_parts.push(format!("User: {content}")),
        }
    }

    if system_parts.is_empty() && prompt_parts.is_empty() {
        return None;
    }
    Some(ParsedMessagesRequest {
        system: join_lines(&system_parts).unwrap_or_else(|| "You are a helpful assistant".to_string()),
        prompt: join_lines(&prompt_parts).unwrap_or_else(|| "hello".to_string()),
        last_tool_call_id,
        has_tool_results,
        last_tool_result,
    })
}

fn render_tool_result_text(output: &str) -> String {
    format!("<tool_result>{}</tool_result>", escape_tool_result_text(output))
}

fn escape_tool_result_text(output: &str) -> String {
    output.replace("</tool_result>", "&lt;/tool_result>")
}

fn extract_tools(raw: &str) -> Vec<RequestTool> {
    let mut out = Vec::new();
    for tool in extract_top_level_objects(raw, "tools").unwrap_or_default() {
        let Some(name) = extract_json_string(tool, "name")
            .or_else(|| extract_nested_json_string(tool, "function", "name"))
        else {
            continue;
        };
        out.push(RequestTool {
            name,
            first_arg_name: extract_first_property_name(tool),
            property_names: extract_property_names(tool),
        });
    }
    out
}
