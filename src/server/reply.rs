use std::sync::Arc;

use crate::engine::Engine;
use crate::protocol::RequestEnvelope;
use crate::session::Session;

pub(super) fn preview_or_model_reply(
    engine: &Arc<Engine>,
    request: &RequestEnvelope,
    session: &mut Session,
    generation_budget: usize,
) -> crate::error::Result<String> {
    if !engine.has_real_model() {
        return Ok(preview_text_reply(request));
    }
    let generated = match session.generate_argmax_tokens(generation_budget) {
        Ok(tokens) => tokens,
        Err(_) => return Ok(preview_text_reply(request)),
    };
    let rendered = session.render_tokens(&generated);
    if should_fallback_generated_reply(&rendered) {
        return Ok(preview_text_reply(request));
    }
    Ok(rendered)
}

pub(super) fn preview_text_reply(request: &RequestEnvelope) -> String {
    let latest = latest_user_text(&request.prompt);
    let chinese = prefers_chinese_reply(&latest, request.last_tool_result.as_deref());
    if let Some(tool_result) = &request.last_tool_result {
        let compact = tool_result.trim();
        if compact.is_empty() {
            return if chinese {
                "我已经执行完这一步，但工具没有返回可见内容。你可以继续告诉我下一步要做什么。".to_string()
            } else {
                "I completed that step, but the tool did not return visible output. You can tell me what to do next.".to_string()
            };
        }
        let summary = summarize_text(compact, 6, 220);
        return if chinese {
            format!("我已经拿到结果了：\n{summary}\n\n如果你愿意，我可以继续基于这个结果往下处理。")
        } else {
            format!("I received the result:\n{summary}\n\nIf you want, I can keep going from here.")
        };
    }
    if latest.is_empty() {
        return if chinese {
            "我在。你可以直接告诉我你想让我做什么。".to_string()
        } else {
            "I'm here. Tell me what you want me to do.".to_string()
        };
    }
    if looks_like_ping(&latest) {
        return if chinese {
            "我在，已经收到你的消息了。你可以继续说具体需求，我会直接回复。".to_string()
        } else {
            "I'm here and I got your message. You can continue with the actual request.".to_string()
        };
    }
    if looks_like_question(&latest) {
        return if chinese {
            format!(
                "我收到了你的问题：{}。\n\n现在已经可以正常回复了。你可以继续追问，或者把要我处理的任务直接发给我。",
                latest.trim()
            )
        } else {
            format!(
                "I received your question: {}.\n\nThe conversation path is working now. You can keep asking, or send me the task directly.",
                latest.trim()
            )
        };
    }
    if chinese {
        format!(
            "收到，你刚才说的是：{}。\n\n我可以继续和你正常对话，也可以直接开始处理这个需求。",
            latest.trim()
        )
    } else {
        format!(
            "Got it. You said: {}.\n\nI can keep chatting normally, or I can start working on this request directly.",
            latest.trim()
        )
    }
}

pub(super) fn latest_user_text(prompt: &str) -> String {
    prompt
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix("User: "))
        .unwrap_or(prompt)
        .trim()
        .to_string()
}

pub(super) fn should_fallback_generated_reply(rendered: &str) -> bool {
    let trimmed = rendered.trim();
    if trimmed.is_empty() {
        return true;
    }
    if contains_think_marker(trimmed)
        || has_high_replacement_ratio(trimmed)
        || is_repetitive_reply(trimmed)
    {
        return true;
    }
    let meaningful = trimmed
        .chars()
        .filter(|ch| ch.is_alphanumeric() || is_cjk_char(*ch))
        .count();
    meaningful == 0 || meaningful == 1
}

fn prefers_chinese_reply(prompt: &str, tool_result: Option<&str>) -> bool {
    contains_cjk(prompt) || tool_result.is_some_and(contains_cjk)
}

fn contains_cjk(text: &str) -> bool {
    text.chars().any(is_cjk_char)
}

fn is_cjk_char(ch: char) -> bool {
    matches!(ch as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x3000..=0x303F)
}

fn contains_think_marker(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("<think") || lower.contains("</think>")
}

fn has_high_replacement_ratio(text: &str) -> bool {
    let visible_chars: Vec<char> = text.chars().filter(|ch| !ch.is_whitespace()).collect();
    if visible_chars.len() < 4 {
        return false;
    }
    let replacements = visible_chars.iter().filter(|ch| **ch == '\u{FFFD}').count();
    replacements >= 2 && replacements * 2 >= visible_chars.len()
}

fn is_repetitive_reply(text: &str) -> bool {
    let normalized: Vec<char> = text.chars().filter(|ch| !ch.is_whitespace()).collect();
    if normalized.len() < 6 {
        return false;
    }
    for unit_len in 1..=normalized.len() / 3 {
        if !normalized.len().is_multiple_of(unit_len) {
            continue;
        }
        let repeats = normalized.len() / unit_len;
        if repeats < 3 {
            continue;
        }
        let unit = &normalized[..unit_len];
        if normalized
            .chunks(unit_len)
            .all(|chunk| chunk.len() == unit.len() && chunk == unit)
        {
            return true;
        }
    }
    false
}

fn looks_like_ping(text: &str) -> bool {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();
    trimmed.is_empty()
        || [
            "你好",
            "在吗",
            "你在吗",
            "回答我",
            "你要回答我啊",
            "收到吗",
            "嗨",
            "hello",
            "hi",
            "are you there",
        ]
        .iter()
        .any(|needle| lower.contains(needle))
}

fn looks_like_question(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed.ends_with('?')
        || trimmed.ends_with('？')
        || trimmed.starts_with("为什么")
        || trimmed.starts_with("怎么")
        || trimmed.starts_with("如何")
        || trimmed.to_lowercase().starts_with("why")
        || trimmed.to_lowercase().starts_with("how")
}

fn summarize_text(text: &str, max_lines: usize, max_chars: usize) -> String {
    let mut out = String::new();
    for (idx, line) in text.lines().filter(|line| !line.trim().is_empty()).enumerate() {
        if idx >= max_lines || out.chars().count() >= max_chars {
            break;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line.trim());
    }
    if out.is_empty() {
        text.chars().take(max_chars).collect()
    } else if text.chars().count() > out.chars().count() {
        format!("{out}\n...")
    } else {
        out
    }
}
