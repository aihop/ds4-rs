use serde_json::Value;
use super::{
    ApiKind, ParsedChatRequest, ParsedMessagesRequest, ParsedResponsesRequest, RequestEnvelope,
    RequestTool,
};

impl RequestEnvelope {
    pub fn from_http(api: ApiKind, raw: &str) -> Self {
        let value: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
        
        let chat = match api {
            ApiKind::ChatCompletions => parse_chat_request(&value),
            _ => None,
        };
        let responses = match api {
            ApiKind::Responses => parse_responses_request(&value),
            _ => None,
        };
        let messages = match api {
            ApiKind::Messages => parse_messages_request(&value),
            _ => None,
        };
        
        let system;
        let prompt;
        
        if let Some(c) = &chat {
            system = c.system.clone();
            prompt = c.prompt.clone();
        } else if let Some(r) = &responses {
            system = r.system.clone();
            prompt = r.prompt.clone();
        } else if let Some(m) = &messages {
            system = m.system.clone();
            prompt = m.prompt.clone();
        } else {
            system = value.get("system").and_then(|v| v.as_str()).unwrap_or("You are a helpful assistant").to_string();
            prompt = value.get("prompt").or_else(|| value.get("input")).and_then(|v| v.as_str()).unwrap_or("hello").to_string();
        }
        
        let previous_response_id = value.get("previous_response_id").and_then(|v| v.as_str()).map(|s| s.to_string());
        let conversation = value.get("conversation").and_then(|v| v.as_str()).map(|s| s.to_string());
        let available_tools = extract_tools(&value);
        let has_tools = !available_tools.is_empty();
        let (primary_tool_name, primary_tool_arg_name) = available_tools
            .first()
            .cloned()
            .map(|tool| (Some(tool.name), tool.first_arg_name))
            .unwrap_or((None, None));
            
        let raw_last_tool_call_id = value.get("tool_call_id").and_then(|v| v.as_str()).map(|s| s.to_string());
        let raw_last_content = value.get("content").and_then(|v| v.as_str()).map(|s| s.to_string());
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
            
        let stream = value.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
        let max_output_tokens = value.get("max_output_tokens")
            .or_else(|| value.get("max_completion_tokens"))
            .or_else(|| value.get("max_tokens"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
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

fn parse_chat_request(value: &Value) -> Option<ParsedChatRequest> {
    let mut system_parts = Vec::new();
    let mut prompt_parts = Vec::new();
    let mut last_tool_call_id = None;
    let mut has_tool_results = false;
    let mut last_tool_result = None;

    if let Some(messages) = value.get("messages").and_then(|v| v.as_array()) {
        for message in messages {
            let role = message.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let content = message.get("content").and_then(|v| v.as_str()).unwrap_or("");
            
            match role {
                "system" | "developer" => {
                    if !content.is_empty() {
                        system_parts.push(content.to_string());
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
                    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
                        for tool_call in tool_calls {
                            let call_id = tool_call.get("id").and_then(|v| v.as_str()).unwrap_or("call_unknown");
                            let call_type = tool_call.get("type").and_then(|v| v.as_str()).unwrap_or("function");
                            let name = tool_call.get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("tool");
                            let arguments = tool_call.get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                                
                            prompt_parts.push(format!("AssistantToolCall[{call_id}] {call_type} {name}({arguments})"));
                            last_tool_call_id = Some(call_id.to_string());
                        }
                    }
                }
                "tool" => {
                    let tool_call_id = message.get("tool_call_id").and_then(|v| v.as_str()).unwrap_or("call_unknown");
                    if !content.is_empty() {
                        prompt_parts.push(render_tool_result_text(content));
                        last_tool_result = Some(content.to_string());
                    }
                    last_tool_call_id = Some(tool_call_id.to_string());
                    has_tool_results = true;
                }
                _ => {}
            }
        }
    }

    if system_parts.is_empty() && prompt_parts.is_empty() {
        return None;
    }

    Some(ParsedChatRequest {
        system: if system_parts.is_empty() { "You are a helpful assistant".to_string() } else { system_parts.join("\n") },
        prompt: if prompt_parts.is_empty() { "hello".to_string() } else { prompt_parts.join("\n") },
        last_tool_call_id,
        has_tool_results,
        last_tool_result,
    })
}

fn parse_responses_request(value: &Value) -> Option<ParsedResponsesRequest> {
    let mut system_parts = Vec::new();
    let mut prompt_parts = Vec::new();
    let mut last_tool_call_id = None;
    let mut has_tool_results = false;
    let mut last_tool_result = None;

    if let Some(instructions) = value.get("instructions").and_then(|v| v.as_str()) {
        if !instructions.is_empty() {
            system_parts.push(instructions.to_string());
        }
    }

    if let Some(items) = value.get("input").and_then(|v| v.as_array()) {
        for item in items {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("message");
            match item_type {
                "message" => {
                    let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                    let content = item.get("content").or_else(|| item.get("text")).and_then(|v| v.as_str()).unwrap_or("");
                    match role {
                        "system" | "developer" => {
                            if !content.is_empty() {
                                system_parts.push(content.to_string());
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
                    let call_id = item.get("call_id").or_else(|| item.get("id")).and_then(|v| v.as_str()).unwrap_or("call_unknown");
                    let name = item.get("name").or_else(|| item.get("function")).and_then(|v| v.as_str()).unwrap_or("tool");
                    let arguments = item.get("arguments").or_else(|| item.get("input")).and_then(|v| v.as_str()).unwrap_or("");
                    prompt_parts.push(format!("AssistantToolCall[{call_id}] function {name}({arguments})"));
                    last_tool_call_id = Some(call_id.to_string());
                }
                "function_call_output" | "custom_tool_call_output" => {
                    let call_id = item.get("call_id").or_else(|| item.get("tool_call_id")).or_else(|| item.get("id")).and_then(|v| v.as_str()).unwrap_or("call_unknown");
                    let output = item.get("output").or_else(|| item.get("content")).or_else(|| item.get("text")).and_then(|v| v.as_str()).unwrap_or("");
                    if !output.is_empty() {
                        prompt_parts.push(render_tool_result_text(output));
                        last_tool_result = Some(output.to_string());
                    }
                    last_tool_call_id = Some(call_id.to_string());
                    has_tool_results = true;
                }
                "reasoning" => {
                    if let Some(summary) = item.get("summary").and_then(|v| v.as_str()) {
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
        let prompt = value.get("prompt").or_else(|| value.get("input")).and_then(|v| v.as_str())?;
        return Some(ParsedResponsesRequest {
            system: value.get("instructions").or_else(|| value.get("system")).and_then(|v| v.as_str()).unwrap_or("You are a helpful assistant").to_string(),
            prompt: prompt.to_string(),
            last_tool_call_id,
            has_tool_results,
            last_tool_result,
        });
    }

    Some(ParsedResponsesRequest {
        system: if system_parts.is_empty() { "You are a helpful assistant".to_string() } else { system_parts.join("\n") },
        prompt: if prompt_parts.is_empty() { "hello".to_string() } else { prompt_parts.join("\n") },
        last_tool_call_id,
        has_tool_results,
        last_tool_result,
    })
}

fn parse_messages_request(value: &Value) -> Option<ParsedMessagesRequest> {
    let mut system_parts = Vec::new();
    let mut prompt_parts = Vec::new();
    let mut last_tool_call_id = None;
    let mut has_tool_results = false;
    let mut last_tool_result = None;

    if let Some(system) = value.get("system") {
        if let Some(s) = system.as_str() {
            if !s.is_empty() {
                system_parts.push(s.to_string());
            }
        } else if let Some(arr) = system.as_array() {
            for item in arr {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        system_parts.push(text.to_string());
                    }
                }
            }
        }
    }

    if let Some(messages) = value.get("messages").and_then(|v| v.as_array()) {
        for message in messages {
            let role = message.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            
            if let Some(content_blocks) = message.get("content").and_then(|v| v.as_array()) {
                for block in content_blocks {
                    let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("text");
                    match block_type {
                        "text" | "input_text" | "output_text" => {
                            let text = block.get("text").or_else(|| block.get("content")).and_then(|v| v.as_str()).unwrap_or("");
                            if text.is_empty() { continue; }
                            match role {
                                "assistant" => prompt_parts.push(format!("Assistant: {text}")),
                                "system" | "developer" => system_parts.push(text.to_string()),
                                _ => prompt_parts.push(format!("User: {text}")),
                            }
                        }
                        "tool_use" => {
                            let call_id = block.get("id").or_else(|| block.get("tool_use_id")).and_then(|v| v.as_str()).unwrap_or("call_unknown");
                            let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                            let arguments = block.get("input").map(|v| v.to_string()).unwrap_or_else(|| "{}".to_string());
                            prompt_parts.push(format!("AssistantToolCall[{call_id}] function {name}({arguments})"));
                            last_tool_call_id = Some(call_id.to_string());
                        }
                        "tool_result" => {
                            let call_id = block.get("tool_use_id").or_else(|| block.get("id")).and_then(|v| v.as_str()).unwrap_or("call_unknown");
                            let output = block.get("content").or_else(|| block.get("text")).and_then(|v| v.as_str()).unwrap_or("");
                            if !output.is_empty() {
                                prompt_parts.push(render_tool_result_text(output));
                                last_tool_result = Some(output.to_string());
                            }
                            last_tool_call_id = Some(call_id.to_string());
                            has_tool_results = true;
                        }
                        _ => {}
                    }
                }
                continue;
            }

            let content = message.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if content.is_empty() { continue; }
            match role {
                "assistant" => prompt_parts.push(format!("Assistant: {content}")),
                "system" | "developer" => system_parts.push(content.to_string()),
                _ => prompt_parts.push(format!("User: {content}")),
            }
        }
    }

    if system_parts.is_empty() && prompt_parts.is_empty() {
        return None;
    }

    Some(ParsedMessagesRequest {
        system: if system_parts.is_empty() { "You are a helpful assistant".to_string() } else { system_parts.join("\n") },
        prompt: if prompt_parts.is_empty() { "hello".to_string() } else { prompt_parts.join("\n") },
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

fn extract_tools(value: &Value) -> Vec<RequestTool> {
    let mut out = Vec::new();
    if let Some(tools) = value.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            if let Some(name) = tool.get("name").or_else(|| tool.get("function").and_then(|f| f.get("name"))).and_then(|v| v.as_str()) {
                let mut property_names = Vec::new();
                if let Some(parameters) = tool.get("parameters").or_else(|| tool.get("input_schema")) {
                    if let Some(properties) = parameters.get("properties").and_then(|v| v.as_object()) {
                        for key in properties.keys() {
                            property_names.push(key.clone());
                        }
                    }
                }
                let first_arg_name = property_names.first().cloned();
                out.push(RequestTool {
                    name: name.to_string(),
                    first_arg_name,
                    property_names,
                });
            }
        }
    }
    out
}
