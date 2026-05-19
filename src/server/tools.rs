use crate::protocol::{RequestEnvelope, RequestTool};

pub(super) fn should_short_circuit_tool_planning(request: &RequestEnvelope) -> bool {
    request.has_tools
        && !request.has_tool_results
        && prompt_likely_needs_tool(&super::reply::latest_user_text(&request.prompt))
}

pub(super) fn choose_tool_call(
    request: &RequestEnvelope,
    json_escape: impl Fn(&str) -> String,
) -> (String, String) {
    let latest = super::reply::latest_user_text(&request.prompt);
    let tool = select_best_tool(&request.available_tools, &latest)
        .or_else(|| request.available_tools.first())
        .cloned()
        .unwrap_or(RequestTool {
            name: request
                .primary_tool_name
                .clone()
                .unwrap_or_else(|| "tool".to_string()),
            first_arg_name: request.primary_tool_arg_name.clone(),
            property_names: Vec::new(),
        });
    let args = build_tool_arguments_for_tool(&tool, &latest, json_escape);
    (tool.name, args)
}

fn select_best_tool<'a>(tools: &'a [RequestTool], prompt: &str) -> Option<&'a RequestTool> {
    let intent = detect_tool_intent(prompt);

    if intent.wants_shell {
        if let Some(tool) = find_tool(tools, &["bash", "run_terminal_cmd", "terminal", "shell"]) {
            return Some(tool);
        }
    }
    if intent.wants_read {
        if let Some(tool) = find_tool(tools, &["read", "read_file", "file_read"]) {
            return Some(tool);
        }
    }
    if intent.wants_search {
        if let Some(tool) = find_tool(tools, &["grep", "search", "ripgrep"]) {
            return Some(tool);
        }
    }
    if intent.wants_glob {
        if let Some(tool) = find_tool(tools, &["glob", "find_files"]) {
            return Some(tool);
        }
    }
    find_tool(tools, &["bash", "read", "grep", "glob", "question"])
}

fn find_tool<'a>(tools: &'a [RequestTool], names: &[&str]) -> Option<&'a RequestTool> {
    tools
        .iter()
        .find(|tool| names.iter().any(|name| tool.name.eq_ignore_ascii_case(name)))
}

#[derive(Clone, Copy, Debug, Default)]
struct ToolIntent {
    wants_shell: bool,
    wants_read: bool,
    wants_search: bool,
    wants_glob: bool,
}

fn prompt_likely_needs_tool(prompt: &str) -> bool {
    let intent = detect_tool_intent(prompt);
    intent.wants_shell || intent.wants_read || intent.wants_search || intent.wants_glob
}

fn detect_tool_intent(prompt: &str) -> ToolIntent {
    let lower = prompt.to_lowercase();
    ToolIntent {
        wants_shell: [
            "list files",
            "current directory",
            "pwd",
            "ls",
            "git ",
            "cargo ",
            "npm ",
            "pnpm ",
            "yarn ",
            "run ",
            "command",
            "terminal",
            "shell",
            "bash",
        ]
        .iter()
        .any(|needle| lower.contains(needle)),
        wants_read: ["read ", "open ", "show file", "view file"]
            .iter()
            .any(|needle| lower.contains(needle)),
        wants_search: ["grep", "search", "find text", "find string"]
            .iter()
            .any(|needle| lower.contains(needle)),
        wants_glob: ["find files", "glob", "*.", "file pattern"]
            .iter()
            .any(|needle| lower.contains(needle)),
    }
}

pub(super) fn build_tool_arguments_for_tool(
    tool: &RequestTool,
    prompt: &str,
    json_escape: impl Fn(&str) -> String,
) -> String {
    let lower_name = tool.name.to_lowercase();
    if matches!(lower_name.as_str(), "bash" | "run_terminal_cmd" | "terminal" | "shell") {
        let command = infer_shell_command(prompt);
        if tool_accepts_property(tool, "description") {
            return format!(
                "{{\"command\":\"{}\",\"description\":\"{}\"}}",
                json_escape(&command),
                json_escape("Runs requested shell command")
            );
        }
        return format!("{{\"command\":\"{}\"}}", json_escape(&command));
    }
    if matches!(lower_name.as_str(), "read" | "read_file" | "file_read") {
        let path = infer_path(prompt).unwrap_or_else(|| prompt.to_string());
        let key = tool.first_arg_name.as_deref().unwrap_or("file_path");
        return format!("{{\"{}\":\"{}\"}}", json_escape(key), json_escape(&path));
    }
    if matches!(lower_name.as_str(), "grep" | "search" | "ripgrep") {
        let key = tool.first_arg_name.as_deref().unwrap_or("pattern");
        return format!("{{\"{}\":\"{}\"}}", json_escape(key), json_escape(prompt));
    }
    build_tool_arguments(tool.first_arg_name.as_deref(), prompt, json_escape)
}

fn build_tool_arguments(
    arg_name: Option<&str>,
    prompt: &str,
    json_escape: impl Fn(&str) -> String,
) -> String {
    match arg_name {
        Some(name) if !name.is_empty() => {
            format!("{{\"{}\":\"{}\"}}", json_escape(name), json_escape(prompt))
        }
        _ => format!("{{\"input\":\"{}\"}}", json_escape(prompt)),
    }
}

fn infer_shell_command(prompt: &str) -> String {
    let lower = prompt.to_lowercase();
    if lower.contains("list files") || lower.contains("ls") {
        "ls -la".to_string()
    } else if lower.contains("current directory") || lower == "pwd" {
        "pwd".to_string()
    } else if lower.contains("git status") {
        "git status".to_string()
    } else if lower.contains("cargo test") {
        "cargo test".to_string()
    } else if lower.contains("npm test") {
        "npm test".to_string()
    } else {
        prompt.to_string()
    }
}

fn tool_accepts_property(tool: &RequestTool, property_name: &str) -> bool {
    tool.property_names
        .iter()
        .any(|name| name.eq_ignore_ascii_case(property_name))
}

fn infer_path(prompt: &str) -> Option<String> {
    let bytes = prompt.as_bytes();
    for quote in [b'`', b'"', b'\''] {
        let Some(start) = bytes.iter().position(|b| *b == quote) else {
            continue;
        };
        let Some(end_rel) = bytes[start + 1..].iter().position(|b| *b == quote) else {
            continue;
        };
        let end = start + 1 + end_rel;
        if end > start + 1 {
            return Some(prompt[start + 1..end].to_string());
        }
    }
    None
}
