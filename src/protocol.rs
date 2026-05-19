mod parse;
mod render;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL_ID: &str = "deepseek-v4-flash";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiKind {
    ChatCompletions,
    Responses,
    Completions,
    Messages,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestEnvelope {
    pub api: ApiKind,
    pub system: String,
    pub prompt: String,
    pub previous_response_id: Option<String>,
    pub conversation: Option<String>,
    pub available_tools: Vec<RequestTool>,
    pub has_tools: bool,
    pub has_tool_results: bool,
    pub primary_tool_name: Option<String>,
    pub primary_tool_arg_name: Option<String>,
    pub last_tool_call_id: Option<String>,
    pub last_tool_result: Option<String>,
    pub stream: bool,
    pub max_output_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestTool {
    pub name: String,
    pub first_arg_name: Option<String>,
    pub property_names: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssistantToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResponseEnvelope {
    pub api: ApiKind,
    pub id: String,
    pub object: String,
    pub created: u64,
    pub cached_tokens: usize,
    pub replay_tokens: usize,
    pub rebuilt: bool,
    pub continuation_hit: bool,
    pub message: String,
    pub tool_call: Option<AssistantToolCall>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParsedChatRequest {
    system: String,
    prompt: String,
    last_tool_call_id: Option<String>,
    has_tool_results: bool,
    last_tool_result: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParsedResponsesRequest {
    system: String,
    prompt: String,
    last_tool_call_id: Option<String>,
    has_tool_results: bool,
    last_tool_result: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParsedMessagesRequest {
    system: String,
    prompt: String,
    last_tool_call_id: Option<String>,
    has_tool_results: bool,
    last_tool_result: Option<String>,
}
