use std::time::{SystemTime, UNIX_EPOCH};

use super::jsonish::{json_escape, stream_chunks};
use super::{ApiKind, AssistantToolCall, ResponseEnvelope, DEFAULT_MODEL_ID};

impl ResponseEnvelope {
    pub fn new_message(
        api: ApiKind,
        cached_tokens: usize,
        replay_tokens: usize,
        rebuilt: bool,
        continuation_hit: bool,
        message: impl Into<String>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let millis = now.as_millis();
        let created = now.as_secs();
        let prefix = match api {
            ApiKind::ChatCompletions => "chatcmpl_rust",
            ApiKind::Responses => "resp_rust",
            ApiKind::Completions => "cmpl_rust",
            ApiKind::Messages => "msg_rust",
        };
        Self {
            api,
            id: format!("{prefix}_{millis}"),
            object: match api {
                ApiKind::ChatCompletions => "chat.completion".to_string(),
                ApiKind::Responses => "response".to_string(),
                ApiKind::Completions => "text_completion".to_string(),
                ApiKind::Messages => "message".to_string(),
            },
            created,
            cached_tokens,
            replay_tokens,
            rebuilt,
            continuation_hit,
            message: message.into(),
            tool_call: None,
        }
    }

    pub fn new_tool_call(
        api: ApiKind,
        cached_tokens: usize,
        replay_tokens: usize,
        rebuilt: bool,
        continuation_hit: bool,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        let mut response = Self::new_message(
            api,
            cached_tokens,
            replay_tokens,
            rebuilt,
            continuation_hit,
            "",
        );
        let call_id = format!("call_{}", response.id);
        response.tool_call = Some(AssistantToolCall {
            id: call_id,
            name: name.into(),
            arguments: arguments.into(),
        });
        response
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call.as_ref().map(|call| call.id.as_str())
    }

    pub fn tool_replay_block(&self) -> Option<String> {
        self.tool_call.as_ref().map(canonical_dsml_tool_block)
    }

    pub fn to_json(&self) -> String {
        match self.api {
            ApiKind::ChatCompletions => self.to_chat_json(),
            ApiKind::Responses => self.to_responses_json(),
            ApiKind::Completions => self.to_completions_json(),
            ApiKind::Messages => self.to_messages_json(),
        }
    }

    pub fn to_sse(&self) -> String {
        match self.api {
            ApiKind::ChatCompletions => self.chat_sse(),
            ApiKind::Responses => self.responses_sse(),
            ApiKind::Completions => self.completions_sse(),
            ApiKind::Messages => self.messages_sse(),
        }
    }

    fn to_chat_json(&self) -> String {
        if let Some(call) = &self.tool_call {
            let prompt_tokens = self.cached_tokens + self.replay_tokens;
            return format!(
                concat!(
                    "{{\"id\":\"{}\",\"object\":\"{}\",\"created\":{},\"model\":\"{}\",",
                    "\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":null,",
                    "\"tool_calls\":[{{\"id\":\"{}\",\"type\":\"function\",\"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}]}},",
                    "\"finish_reason\":\"tool_calls\"}}],",
                    "\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":0,\"total_tokens\":{}}},",
                    "\"cached_tokens\":{},\"replay_tokens\":{},\"rebuilt\":{},\"continuation_hit\":{}}}"
                ),
                json_escape(&self.id),
                json_escape(&self.object),
                self.created,
                json_escape(DEFAULT_MODEL_ID),
                json_escape(&call.id),
                json_escape(&call.name),
                json_escape(&call.arguments),
                prompt_tokens,
                prompt_tokens,
                self.cached_tokens,
                self.replay_tokens,
                self.rebuilt,
                self.continuation_hit
            );
        }
        let completion_tokens = self.message.chars().count().max(1);
        let prompt_tokens = self.cached_tokens + self.replay_tokens;
        format!(
            concat!(
                "{{\"id\":\"{}\",\"object\":\"{}\",\"created\":{},\"model\":\"{}\",",
                "\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"stop\"}}],",
                "\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}},",
                "\"cached_tokens\":{},\"replay_tokens\":{},\"rebuilt\":{},\"continuation_hit\":{}}}"
            ),
            json_escape(&self.id),
            json_escape(&self.object),
            self.created,
            json_escape(DEFAULT_MODEL_ID),
            json_escape(&self.message),
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            self.cached_tokens,
            self.replay_tokens,
            self.rebuilt,
            self.continuation_hit
        )
    }

    fn to_responses_json(&self) -> String {
        if let Some(call) = &self.tool_call {
            return format!(
                concat!(
                    "{{\"id\":\"{}\",\"object\":\"{}\",\"created_at\":{},\"status\":\"completed\",",
                    "\"output\":[{{\"id\":\"fc_{}\",\"type\":\"function_call\",\"status\":\"completed\",",
                    "\"call_id\":\"{}\",\"name\":\"{}\",\"arguments\":\"{}\"}}],",
                    "\"output_text\":\"\",\"cached_tokens\":{},\"replay_tokens\":{},\"rebuilt\":{},\"continuation_hit\":{}}}"
                ),
                json_escape(&self.id),
                json_escape(&self.object),
                self.created,
                json_escape(&self.id),
                json_escape(&call.id),
                json_escape(&call.name),
                json_escape(&call.arguments),
                self.cached_tokens,
                self.replay_tokens,
                self.rebuilt,
                self.continuation_hit
            );
        }
        format!(
            concat!(
                "{{\"id\":\"{}\",\"object\":\"{}\",\"created_at\":{},\"status\":\"completed\",",
                "\"output\":[{{\"id\":\"msg_{}\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",",
                "\"content\":[{{\"type\":\"output_text\",\"text\":\"{}\"}}]}}],",
                "\"output_text\":\"{}\",\"cached_tokens\":{},\"replay_tokens\":{},\"rebuilt\":{},\"continuation_hit\":{}}}"
            ),
            json_escape(&self.id),
            json_escape(&self.object),
            self.created,
            json_escape(&self.id),
            json_escape(&self.message),
            json_escape(&self.message),
            self.cached_tokens,
            self.replay_tokens,
            self.rebuilt,
            self.continuation_hit
        )
    }

    fn to_completions_json(&self) -> String {
        let completion_tokens = self.message.chars().count().max(1);
        let prompt_tokens = self.cached_tokens + self.replay_tokens;
        format!(
            concat!(
                "{{\"id\":\"{}\",\"object\":\"{}\",\"created\":{},\"model\":\"{}\",",
                "\"choices\":[{{\"text\":\"{}\",\"index\":0,\"finish_reason\":\"stop\"}}],",
                "\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}},",
                "\"cached_tokens\":{},\"replay_tokens\":{},\"rebuilt\":{},\"continuation_hit\":{}}}"
            ),
            json_escape(&self.id),
            json_escape(&self.object),
            self.created,
            json_escape(DEFAULT_MODEL_ID),
            json_escape(&self.message),
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            self.cached_tokens,
            self.replay_tokens,
            self.rebuilt,
            self.continuation_hit
        )
    }

    fn to_messages_json(&self) -> String {
        let input_tokens = self.cached_tokens + self.replay_tokens;
        let output_tokens = self.message.chars().count().max(1);
        if let Some(call) = &self.tool_call {
            return format!(
                concat!(
                    "{{\"id\":\"{}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"{}\",",
                    "\"content\":[{{\"type\":\"tool_use\",\"id\":\"{}\",\"name\":\"{}\",\"input\":{}}}],",
                    "\"stop_reason\":\"tool_use\",\"stop_sequence\":null,",
                    "\"usage\":{{\"input_tokens\":{},\"output_tokens\":{}}},",
                    "\"cached_tokens\":{},\"replay_tokens\":{},\"rebuilt\":{},\"continuation_hit\":{}}}"
                ),
                json_escape(&self.id),
                json_escape(DEFAULT_MODEL_ID),
                json_escape(&call.id),
                json_escape(&call.name),
                raw_json_or_string(&call.arguments),
                input_tokens,
                output_tokens,
                self.cached_tokens,
                self.replay_tokens,
                self.rebuilt,
                self.continuation_hit
            );
        }
        format!(
            concat!(
                "{{\"id\":\"{}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"{}\",",
                "\"content\":[{{\"type\":\"text\",\"text\":\"{}\"}}],",
                "\"stop_reason\":\"end_turn\",\"stop_sequence\":null,",
                "\"usage\":{{\"input_tokens\":{},\"output_tokens\":{}}},",
                "\"cached_tokens\":{},\"replay_tokens\":{},\"rebuilt\":{},\"continuation_hit\":{}}}"
            ),
            json_escape(&self.id),
            json_escape(DEFAULT_MODEL_ID),
            json_escape(&self.message),
            input_tokens,
            output_tokens,
            self.cached_tokens,
            self.replay_tokens,
            self.rebuilt,
            self.continuation_hit
        )
    }

    fn chat_sse(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}\n\n",
            json_escape(&self.id),
            self.created,
            json_escape(DEFAULT_MODEL_ID)
        ));
        if let Some(call) = &self.tool_call {
            out.push_str(&format!(
                "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"id\":\"{}\",\"type\":\"function\",\"function\":{{\"name\":\"{}\"}}}}]}},\"finish_reason\":null}}]}}\n\n",
                json_escape(&self.id),
                self.created,
                json_escape(DEFAULT_MODEL_ID),
                json_escape(&call.id),
                json_escape(&call.name)
            ));
            for chunk in stream_chunks(&call.arguments, 24) {
                out.push_str(&format!(
                    "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{{\"index\":0,\"function\":{{\"arguments\":\"{}\"}}}}]}},\"finish_reason\":null}}]}}\n\n",
                    json_escape(&self.id),
                    self.created,
                    json_escape(DEFAULT_MODEL_ID),
                    json_escape(&chunk)
                ));
            }
            out.push_str(&format!(
                "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n",
                json_escape(&self.id),
                self.created,
                json_escape(DEFAULT_MODEL_ID)
            ));
            out.push_str("data: [DONE]\n\n");
            return out;
        }
        for chunk in stream_chunks(&self.message, 24) {
            out.push_str(&format!(
                "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}\n\n",
                json_escape(&self.id),
                self.created,
                json_escape(DEFAULT_MODEL_ID),
                json_escape(&chunk)
            ));
        }
        out.push_str(&format!(
            "data: {{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n",
            json_escape(&self.id),
            self.created,
            json_escape(DEFAULT_MODEL_ID)
        ));
        out.push_str("data: [DONE]\n\n");
        out
    }

    fn responses_sse(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.responses_lifecycle_header());
        if let Some(call) = &self.tool_call {
            let item_id = format!("fc_{}", self.id);
            out.push_str("event: response.output_item.added\n");
            out.push_str(&format!(
                concat!(
                    "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":",
                    "{{\"id\":\"{}\",\"type\":\"function_call\",\"status\":\"in_progress\",",
                    "\"call_id\":\"{}\",\"name\":\"{}\",\"arguments\":\"\"}}}}\n\n"
                ),
                json_escape(&item_id),
                json_escape(&call.id),
                json_escape(&call.name)
            ));
            for chunk in stream_chunks(&call.arguments, 24) {
                out.push_str("event: response.function_call_arguments.delta\n");
                out.push_str(&format!(
                    concat!(
                        "data: {{\"type\":\"response.function_call_arguments.delta\",",
                        "\"output_index\":0,\"item_id\":\"{}\",\"delta\":\"{}\"}}\n\n"
                    ),
                    json_escape(&item_id),
                    json_escape(&chunk)
                ));
            }
            out.push_str("event: response.function_call_arguments.done\n");
            out.push_str(&format!(
                concat!(
                    "data: {{\"type\":\"response.function_call_arguments.done\",",
                    "\"output_index\":0,\"item_id\":\"{}\",\"name\":\"{}\",\"arguments\":\"{}\"}}\n\n"
                ),
                json_escape(&item_id),
                json_escape(&call.name),
                json_escape(&call.arguments)
            ));
            out.push_str("event: response.output_item.done\n");
            out.push_str(&format!(
                concat!(
                    "data: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":",
                    "{{\"id\":\"{}\",\"type\":\"function_call\",\"status\":\"completed\",",
                    "\"call_id\":\"{}\",\"name\":\"{}\",\"arguments\":\"{}\"}}}}\n\n"
                ),
                json_escape(&item_id),
                json_escape(&call.id),
                json_escape(&call.name),
                json_escape(&call.arguments)
            ));
            out.push_str("event: response.completed\n");
            out.push_str(&format!("data: {}\n\n", self.to_responses_json()));
            return out;
        }
        let item_id = format!("msg_{}", self.id);
        out.push_str("event: response.output_item.added\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":",
                "{{\"id\":\"{}\",\"type\":\"message\",\"status\":\"in_progress\",",
                "\"role\":\"assistant\",\"content\":[]}}}}\n\n"
            ),
            json_escape(&item_id)
        ));
        out.push_str("event: response.content_part.added\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"response.content_part.added\",\"item_id\":\"{}\",",
                "\"output_index\":0,\"content_index\":0,\"part\":",
                "{{\"type\":\"output_text\",\"text\":\"\",\"annotations\":[]}}}}\n\n"
            ),
            json_escape(&item_id)
        ));
        for chunk in stream_chunks(&self.message, 24) {
            out.push_str("event: response.output_text.delta\n");
            out.push_str(&format!(
                concat!(
                    "data: {{\"type\":\"response.output_text.delta\",\"item_id\":\"{}\",",
                    "\"output_index\":0,\"content_index\":0,\"delta\":\"{}\"}}\n\n"
                ),
                json_escape(&item_id),
                json_escape(&chunk)
            ));
        }
        out.push_str("event: response.output_text.done\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"response.output_text.done\",\"item_id\":\"{}\",",
                "\"output_index\":0,\"content_index\":0,\"text\":\"{}\"}}\n\n"
            ),
            json_escape(&item_id),
            json_escape(&self.message)
        ));
        out.push_str("event: response.content_part.done\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"response.content_part.done\",\"item_id\":\"{}\",",
                "\"output_index\":0,\"content_index\":0,\"part\":",
                "{{\"type\":\"output_text\",\"text\":\"{}\",\"annotations\":[]}}}}\n\n"
            ),
            json_escape(&item_id),
            json_escape(&self.message)
        ));
        out.push_str("event: response.output_item.done\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":",
                "{{\"id\":\"{}\",\"type\":\"message\",\"status\":\"completed\",\"role\":\"assistant\",",
                "\"content\":[{{\"type\":\"output_text\",\"text\":\"{}\",\"annotations\":[]}}]}}}}\n\n"
            ),
            json_escape(&item_id),
            json_escape(&self.message)
        ));
        out.push_str("event: response.completed\n");
        out.push_str(&format!("data: {}\n\n", self.to_responses_json()));
        out
    }

    fn completions_sse(&self) -> String {
        let mut out = String::new();
        for chunk in stream_chunks(&self.message, 24) {
            out.push_str(&format!(
                "data: {{\"id\":\"{}\",\"object\":\"text_completion\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"text\":\"{}\",\"index\":0,\"finish_reason\":null}}]}}\n\n",
                json_escape(&self.id),
                self.created,
                json_escape(DEFAULT_MODEL_ID),
                json_escape(&chunk)
            ));
        }
        out.push_str(&format!(
            "data: {{\"id\":\"{}\",\"object\":\"text_completion\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"text\":\"\",\"index\":0,\"finish_reason\":\"stop\"}}]}}\n\n",
            json_escape(&self.id),
            self.created,
            json_escape(DEFAULT_MODEL_ID)
        ));
        out.push_str("data: [DONE]\n\n");
        out
    }

    fn messages_sse(&self) -> String {
        let mut out = String::new();
        let input_tokens = self.cached_tokens + self.replay_tokens;
        let output_tokens = self.message.chars().count().max(1);
        out.push_str("event: message_start\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"message_start\",\"message\":",
                "{{\"id\":\"{}\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"{}\",",
                "\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,",
                "\"usage\":{{\"input_tokens\":{},\"output_tokens\":0}}}}}}\n\n"
            ),
            json_escape(&self.id),
            json_escape(DEFAULT_MODEL_ID),
            input_tokens
        ));
        if let Some(call) = &self.tool_call {
            out.push_str("event: content_block_start\n");
            out.push_str(&format!(
                concat!(
                    "data: {{\"type\":\"content_block_start\",\"index\":0,\"content_block\":",
                    "{{\"type\":\"tool_use\",\"id\":\"{}\",\"name\":\"{}\",\"input\":{{}}}}}}\n\n"
                ),
                json_escape(&call.id),
                json_escape(&call.name)
            ));
            for chunk in stream_chunks(&call.arguments, 24) {
                out.push_str("event: content_block_delta\n");
                out.push_str(&format!(
                    concat!(
                        "data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":",
                        "{{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}}}\n\n"
                    ),
                    json_escape(&chunk)
                ));
            }
            out.push_str("event: content_block_stop\n");
            out.push_str("data: {\"type\":\"content_block_stop\",\"index\":0}\n\n");
            out.push_str("event: message_delta\n");
            out.push_str(&format!(
                concat!(
                    "data: {{\"type\":\"message_delta\",\"delta\":",
                    "{{\"stop_reason\":\"tool_use\",\"stop_sequence\":null}},",
                    "\"usage\":{{\"output_tokens\":{}}}}}\n\n"
                ),
                output_tokens
            ));
            out.push_str("event: message_stop\n");
            out.push_str("data: {\"type\":\"message_stop\"}\n\n");
            return out;
        }
        out.push_str("event: content_block_start\n");
        out.push_str(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        );
        for chunk in stream_chunks(&self.message, 24) {
            out.push_str("event: content_block_delta\n");
            out.push_str(&format!(
                concat!(
                    "data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":",
                    "{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n\n"
                ),
                json_escape(&chunk)
            ));
        }
        out.push_str("event: content_block_stop\n");
        out.push_str("data: {\"type\":\"content_block_stop\",\"index\":0}\n\n");
        out.push_str("event: message_delta\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"message_delta\",\"delta\":",
                "{{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}},",
                "\"usage\":{{\"output_tokens\":{}}}}}\n\n"
            ),
            output_tokens
        ));
        out.push_str("event: message_stop\n");
        out.push_str("data: {\"type\":\"message_stop\"}\n\n");
        out
    }

    fn responses_lifecycle_header(&self) -> String {
        let mut out = String::new();
        out.push_str("event: response.created\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"response.created\",\"response\":",
                "{{\"id\":\"{}\",\"object\":\"response\",\"created_at\":{},",
                "\"status\":\"in_progress\",\"model\":\"{}\",\"output\":[]}}}}\n\n"
            ),
            json_escape(&self.id),
            self.created,
            json_escape(DEFAULT_MODEL_ID)
        ));
        out.push_str("event: response.in_progress\n");
        out.push_str(&format!(
            concat!(
                "data: {{\"type\":\"response.in_progress\",\"response\":",
                "{{\"id\":\"{}\",\"object\":\"response\",\"created_at\":{},",
                "\"status\":\"in_progress\",\"model\":\"{}\",\"output\":[]}}}}\n\n"
            ),
            json_escape(&self.id),
            self.created,
            json_escape(DEFAULT_MODEL_ID)
        ));
        out
    }
}

fn raw_json_or_string(input: &str) -> String {
    let trimmed = input.trim();
    if trimmed.starts_with('{')
        || trimmed.starts_with('[')
        || trimmed == "true"
        || trimmed == "false"
        || trimmed == "null"
        || trimmed
            .chars()
            .next()
            .is_some_and(|ch| ch == '-' || ch.is_ascii_digit())
    {
        trimmed.to_string()
    } else {
        format!("\"{}\"", json_escape(trimmed))
    }
}

#[derive(Debug)]
struct DsmlArg {
    key: String,
    value: String,
    is_string: bool,
}

fn canonical_dsml_tool_block(call: &AssistantToolCall) -> String {
    let mut out = String::new();
    out.push_str("\n\n<｜DSML｜tool_calls>\n");
    out.push_str("<｜DSML｜invoke name=\"");
    out.push_str(&escape_dsml_attr(&call.name));
    out.push_str("\">\n");
    if let Some(args) = parse_json_object_arguments(&call.arguments) {
        for arg in args {
            out.push_str("<｜DSML｜parameter name=\"");
            out.push_str(&escape_dsml_attr(&arg.key));
            out.push_str("\" string=\"");
            out.push_str(if arg.is_string { "true" } else { "false" });
            out.push_str("\">");
            if arg.is_string {
                out.push_str(&escape_dsml_parameter_text(&arg.value));
            } else {
                out.push_str(&arg.value);
            }
            out.push_str("</｜DSML｜parameter>\n");
        }
    } else {
        out.push_str("<｜DSML｜parameter name=\"arguments\" string=\"true\">");
        out.push_str(&escape_dsml_parameter_text(&call.arguments));
        out.push_str("</｜DSML｜parameter>\n");
    }
    out.push_str("</｜DSML｜invoke>\n");
    out.push_str("</｜DSML｜tool_calls>");
    out
}

fn escape_dsml_attr(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_dsml_parameter_text(input: &str) -> String {
    input.replace("</｜DSML｜parameter>", "&lt;/｜DSML｜parameter>")
}

fn parse_json_object_arguments(input: &str) -> Option<Vec<DsmlArg>> {
    let input = input.trim();
    if !input.starts_with('{') || !input.ends_with('}') {
        return None;
    }
    let mut out = Vec::new();
    let mut pos = 1usize;
    while pos < input.len() {
        pos = skip_json_ws(input, pos);
        if input[pos..].starts_with('}') {
            return Some(out);
        }
        let (key, next) = parse_json_string_at(input, pos)?;
        pos = skip_json_ws(input, next);
        if !input[pos..].starts_with(':') {
            return None;
        }
        pos = skip_json_ws(input, pos + 1);
        let end = find_json_value_end(input, pos)?;
        let raw = input[pos..end].trim();
        let (value, is_string) = if raw.starts_with('"') {
            let (decoded, decoded_end) = parse_json_string_at(input, pos)?;
            if decoded_end != end {
                return None;
            }
            (decoded, true)
        } else {
            (raw.to_string(), false)
        };
        out.push(DsmlArg { key, value, is_string });
        pos = skip_json_ws(input, end);
        if input[pos..].starts_with(',') {
            pos += 1;
            continue;
        }
        if input[pos..].starts_with('}') {
            return Some(out);
        }
        return None;
    }
    None
}

fn skip_json_ws(input: &str, mut pos: usize) -> usize {
    while let Some(ch) = input[pos..].chars().next() {
        if ch.is_whitespace() {
            pos += ch.len_utf8();
        } else {
            break;
        }
    }
    pos
}

fn parse_json_string_at(input: &str, start: usize) -> Option<(String, usize)> {
    if !input.get(start..)?.starts_with('"') {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for (rel, ch) in input[start + 1..].char_indices() {
        if escaped {
            out.push(match ch {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '"' => '"',
                '\\' => '\\',
                other => other,
            });
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '"' => return Some((out, start + 1 + rel + 1)),
            other => out.push(other),
        }
    }
    None
}

fn find_json_value_end(input: &str, start: usize) -> Option<usize> {
    let first = input[start..].chars().next()?;
    match first {
        '"' => parse_json_string_at(input, start).map(|(_, end)| end),
        '{' => find_matching_delim(input, start, '{', '}').map(|end| end + 1),
        '[' => find_matching_delim(input, start, '[', ']').map(|end| end + 1),
        _ => Some(
            start
                + input[start..]
                    .find(|ch: char| ch == ',' || ch == '}' || ch.is_whitespace())
                    .unwrap_or(input.len() - start),
        ),
    }
}

fn find_matching_delim(input: &str, start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (rel, ch) in input[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(start + rel);
            }
        }
    }
    None
}
