use std::fs;
use std::path::{Path, PathBuf};

use ds4_rust::{ApiKind, RequestEnvelope};

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    value.get(key)?.as_str().map(|s| s.to_string())
}

fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    let value: serde_json::Value = serde_json::from_str(json).ok()?;
    value.get(key)?.as_u64().map(|n| n as usize)
}

fn vectors_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("test-vectors")
}

fn read_text(path: &Path) -> String {
    fs::read_to_string(path).expect("fixture should be readable")
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

fn extract_object<'a>(input: &'a str, key: &str) -> Option<&'a str> {
    let start = find_json_value_start(input, key)?;
    if !input[start..].starts_with('{') {
        return None;
    }
    let end = find_matching_delim(input, start, '{', '}')?;
    Some(&input[start..=end])
}

fn extract_top_level_objects<'a>(input: &'a str, key: &str) -> Option<Vec<&'a str>> {
    let start = find_json_value_start(input, key)?;
    if !input[start..].starts_with('[') {
        return None;
    }
    let end = find_matching_delim(input, start, '[', ']')?;
    Some(split_top_level_objects(&input[start..=end]))
}

#[test]
fn manifest_references_existing_prompt_and_official_files() {
    let root = vectors_root();
    let manifest = read_text(&root.join("manifest.json"));
    let prompts = extract_top_level_objects(&manifest, "prompts").expect("manifest prompts should exist");

    assert_eq!(
        extract_json_string(&manifest, "schema").as_deref(),
        Some("ds4-test-vector-manifest-v1")
    );
    assert_eq!(
        extract_json_string(&manifest, "model").as_deref(),
        Some("deepseek-v4-flash")
    );
    assert!(!prompts.is_empty());

    for prompt in prompts {
        let prompt_file = root.join(
            extract_json_string(prompt, "prompt_file").expect("prompt_file should be a string"),
        );
        let official_file = root.join(
            extract_json_string(prompt, "official_file").expect("official_file should be a string"),
        );
        assert!(prompt_file.is_file(), "missing prompt file: {}", prompt_file.display());
        assert!(
            official_file.is_file(),
            "missing official file: {}",
            official_file.display()
        );

        let prompt_text = read_text(&prompt_file);
        let official = read_text(&official_file);
        assert_eq!(
            extract_json_string(&official, "id"),
            extract_json_string(prompt, "id")
        );
        assert_eq!(
            extract_json_string(&official, "model"),
            extract_json_string(&manifest, "model")
        );
        assert_eq!(extract_json_string(&official, "prompt").as_deref(), Some(prompt_text.as_str()));
        assert_eq!(
            prompt_text.chars().count() as u64,
            extract_json_usize(prompt, "prompt_chars").expect("prompt_chars should be numeric") as u64
        );
        assert_eq!(
            extract_top_level_objects(&official, "steps")
                .expect("steps should be an array")
                .len() as u64,
            extract_json_usize(prompt, "steps").expect("steps should be numeric") as u64
        );
    }
}

#[test]
fn official_requests_round_trip_into_request_envelopes() {
    let root = vectors_root();
    let manifest = read_text(&root.join("manifest.json"));
    let prompts = extract_top_level_objects(&manifest, "prompts").expect("manifest prompts should exist");

    for prompt in prompts {
        let official = read_text(
            &root.join(extract_json_string(prompt, "official_file").expect("official_file should exist")),
        );
        let request_json = extract_object(&official, "request").expect("request should exist");
        let envelope = RequestEnvelope::from_http(ApiKind::ChatCompletions, request_json);
        let prompt_text = extract_json_string(&official, "prompt").expect("prompt should be a string");

        assert_eq!(envelope.api, ApiKind::ChatCompletions);
        assert_eq!(envelope.system, "You are a helpful assistant");
        assert_eq!(envelope.prompt, format!("User: {prompt_text}"));
        assert_eq!(
            envelope.max_output_tokens,
            extract_json_usize(request_json, "max_tokens").expect("max_tokens should be numeric")
        );
        assert!(!envelope.has_tool_results);
        assert!(envelope.last_tool_call_id.is_none());
        assert!(envelope.last_tool_result.is_none());
    }
}

#[test]
fn official_step_tokens_reconstruct_recorded_message_text() {
    let root = vectors_root();
    let manifest = read_text(&root.join("manifest.json"));
    let prompts = extract_top_level_objects(&manifest, "prompts").expect("manifest prompts should exist");

    for prompt in prompts {
        let vector_id = extract_json_string(prompt, "id").expect("id should exist");
        let official = read_text(
            &root.join(extract_json_string(prompt, "official_file").expect("official_file should exist")),
        );
        let steps = extract_top_level_objects(&official, "steps").expect("steps should be an array");
        let reconstructed = steps
            .iter()
            .map(|step| {
                let token = extract_object(step, "token").expect("token object should exist");
                let token_text = extract_json_string(token, "text").expect("token text should exist");
                let top = extract_top_level_objects(step, "top_logprobs")
                    .and_then(|items| items.into_iter().next())
                    .expect("top_logprobs should have at least one item");
                let top_token = extract_object(top, "token").expect("top token should exist");
                assert_eq!(
                    extract_json_string(top_token, "text").as_deref(),
                    Some(token_text.as_str())
                );
                token_text
            })
            .collect::<String>();

        let message = extract_object(&official, "message").expect("message object should exist");
        let message_content =
            extract_json_string(message, "content").expect("message content should be a string");
        assert_eq!(reconstructed, message_content, "vector {vector_id}");
    }
}
