pub(super) fn extract_message_content(input: &str) -> Option<String> {
    if let Some(text) = extract_json_string(input, "content") {
        return Some(text);
    }
    let start = find_json_value_start(input, "content")?;
    if !input[start..].starts_with('[') {
        return None;
    }
    let end = find_matching_delim(input, start, '[', ']')?;
    let array = &input[start..=end];
    let mut parts = Vec::new();
    for obj in split_top_level_objects(array) {
        let text = extract_json_string(obj, "text")
            .or_else(|| extract_json_string(obj, "content"))
            .unwrap_or_default();
        if !text.is_empty() {
            parts.push(text);
        }
    }
    join_lines(&parts)
}

pub(super) fn extract_json_text_or_array(input: &str, key: &str) -> Option<String> {
    if let Some(text) = extract_json_string(input, key) {
        return Some(text);
    }
    let start = find_json_value_start(input, key)?;
    if !input[start..].starts_with('[') {
        return None;
    }
    let end = find_matching_delim(input, start, '[', ']')?;
    let array = &input[start..=end];
    let mut parts = Vec::new();
    for obj in split_top_level_objects(array) {
        let text = extract_json_string(obj, "text")
            .or_else(|| extract_json_string(obj, "content"))
            .unwrap_or_default();
        if !text.is_empty() {
            parts.push(text);
        }
    }
    join_lines(&parts)
}

pub(super) fn extract_top_level_objects<'a>(input: &'a str, key: &str) -> Option<Vec<&'a str>> {
    let start = find_json_value_start(input, key)?;
    if !input[start..].starts_with('[') {
        return None;
    }
    let end = find_matching_delim(input, start, '[', ']')?;
    Some(split_top_level_objects(&input[start..=end]))
}

fn split_top_level_objects(input: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < input.len() {
        let Some(rel) = input[idx..].find('{') else {
            break;
        };
        let start = idx + rel;
        let Some(end) = find_matching_delim(input, start, '{', '}') else {
            break;
        };
        out.push(&input[start..=end]);
        idx = end + 1;
    }
    out
}

pub(super) fn join_lines(parts: &[String]) -> Option<String> {
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

pub(super) fn stream_chunks(input: &str, chunk_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut chunk = String::new();
    let mut count = 0usize;
    for ch in input.chars() {
        chunk.push(ch);
        count += 1;
        if count >= chunk_len {
            out.push(std::mem::take(&mut chunk));
            count = 0;
        }
    }
    if !chunk.is_empty() {
        out.push(chunk);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn find_json_value_start(input: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let key_pos = input.find(&needle)? + needle.len();
    let colon_rel = input[key_pos..].find(':')?;
    let mut pos = key_pos + colon_rel + 1;
    while let Some(ch) = input[pos..].chars().next() {
        if ch.is_whitespace() {
            pos += ch.len_utf8();
        } else {
            break;
        }
    }
    Some(pos)
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

pub fn extract_json_string(input: &str, key: &str) -> Option<String> {
    let start = find_json_value_start(input, key)?;
    let (value, _) = parse_json_string_at(input, start)?;
    Some(value)
}

pub(super) fn extract_nested_json_string(
    input: &str,
    object_key: &str,
    nested_key: &str,
) -> Option<String> {
    let start = find_json_value_start(input, object_key)?;
    if !input[start..].starts_with('{') {
        return None;
    }
    let end = find_matching_delim(input, start, '{', '}')?;
    extract_json_string(&input[start..=end], nested_key)
}

pub(super) fn extract_raw_json_value(input: &str, key: &str) -> Option<String> {
    let start = find_json_value_start(input, key)?;
    let tail = &input[start..];
    let end = match tail.chars().next()? {
        '"' => parse_json_string_at(input, start).map(|(_, next)| next)?,
        '{' => find_matching_delim(input, start, '{', '}')? + 1,
        '[' => find_matching_delim(input, start, '[', ']')? + 1,
        _ => start
            + tail
                .find(|ch: char| ch == ',' || ch == '}' || ch == ']' || ch.is_whitespace())
                .unwrap_or(tail.len()),
    };
    Some(input[start..end].trim().to_string())
}

pub fn extract_json_bool(input: &str, key: &str) -> Option<bool> {
    let start = find_json_value_start(input, key)?;
    let tail = &input[start..];
    if tail.starts_with("true") {
        Some(true)
    } else if tail.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

pub fn extract_json_usize(input: &str, key: &str) -> Option<usize> {
    let start = find_json_value_start(input, key)?;
    let tail = &input[start..];
    let digits: String = tail.chars().take_while(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

pub(super) fn extract_last_json_string(input: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut offset = 0usize;
    let mut last = None;
    while let Some(rel) = input[offset..].find(&needle) {
        let absolute = offset + rel;
        let Some(start_rel) = find_json_value_start(&input[absolute..], key) else {
            offset = absolute + needle.len();
            continue;
        };
        let value_pos = absolute + start_rel;
        if let Some((value, _)) = parse_json_string_at(input, value_pos) {
            last = Some(value);
            offset = value_pos + 1;
        } else {
            offset = absolute + needle.len();
        }
    }
    last
}

pub(super) fn extract_header_value(raw: &str, name: &str) -> Option<String> {
    let header_end = raw.find("\r\n\r\n").unwrap_or(raw.len());
    for line in raw[..header_end].lines() {
        let Some((header_name, value)) = line.split_once(':') else {
            continue;
        };
        if header_name.eq_ignore_ascii_case(name) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

pub(super) fn extract_first_property_name(input: &str) -> Option<String> {
    extract_property_names(input).into_iter().next()
}

pub(super) fn extract_property_names(input: &str) -> Vec<String> {
    let parameters_start =
        match find_json_value_start(input, "parameters").or_else(|| find_json_value_start(input, "input_schema")) {
            Some(value) => value,
            None => return Vec::new(),
        };
    if !input[parameters_start..].starts_with('{') {
        return Vec::new();
    }
    let Some(parameters_end) = find_matching_delim(input, parameters_start, '{', '}') else {
        return Vec::new();
    };
    let parameters = &input[parameters_start..=parameters_end];
    let Some(properties_start) = find_json_value_start(parameters, "properties") else {
        return Vec::new();
    };
    if !parameters[properties_start..].starts_with('{') {
        return Vec::new();
    }
    let Some(properties_end) = find_matching_delim(parameters, properties_start, '{', '}') else {
        return Vec::new();
    };
    let properties = &parameters[properties_start + 1..properties_end];
    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < properties.len() {
        let Some(rel) = properties[idx..].find('"') else {
            break;
        };
        let key_start = idx + rel;
        let Some((name, next)) = parse_json_string_at(properties, key_start) else {
            break;
        };
        if !name.is_empty() {
            out.push(name);
        }
        idx = next;
    }
    out
}

pub(super) fn json_escape(input: &str) -> String {
    let mut out = String::new();
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "\\u{:04x}", ch as u32);
            }
            other => out.push(other),
        }
    }
    out
}
