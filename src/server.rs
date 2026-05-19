mod reply;
mod tools;

use std::io::Write;
use std::net::TcpStream;
use axum::{
    response::{IntoResponse, sse::Sse},
    extract::State,
    http::StatusCode,
    response::Response,
    Json,
};

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use crate::continuation::{ContinuationEntry, ContinuationStore};
use crate::engine::{Engine, ThinkMode};
use crate::error::{Ds4Error, Result};
use crate::kv::{unix_time_secs_now, KvCache, KvEntry, KvToolReplayEntry};
use crate::protocol::{ApiKind, RequestEnvelope, ResponseEnvelope, DEFAULT_MODEL_ID};
use crate::session::Session;
use crate::tokenizer::render_chat_prompt;
use crate::types::Tokens;
use reply::{latest_user_text, preview_or_model_reply};
use tools::{choose_tool_call, should_short_circuit_tool_planning};

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub ctx_size: usize,
    pub kv_disk_dir: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            ctx_size: 32_768,
            kv_disk_dir: None,
        }
    }
}

#[derive(Clone)]
pub struct Server {
    engine: Arc<Engine>,
    session: Arc<Mutex<Session>>,
    kv: Arc<Mutex<KvCache>>,
    continuations: Arc<Mutex<ContinuationStore>>,
    config: ServerConfig,
}

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
const DEBUG_ENV_PATH: &str = ".dbg/opencode-openai-compat.env";

impl Server {
    /// Creates a new `Server` instance bound to the given engine and config.
    pub fn new(engine: Arc<Engine>, config: ServerConfig) -> Result<Self> {
        let session = Session::create(engine.clone(), config.ctx_size)?;
        let kv = match &config.kv_disk_dir {
            Some(path) => KvCache::with_disk_dir(path)?,
            None => KvCache::new(),
        };
        Ok(Self {
            engine,
            session: Arc::new(Mutex::new(session)),
            kv: Arc::new(Mutex::new(kv)),
            continuations: Arc::new(Mutex::new(ContinuationStore::new())),
            config,
        })
    }

    pub async fn listen_and_serve(&self) -> Result<()> {
        let app = axum::Router::new()
            .route("/", axum::routing::get(Self::health_handler))
            .route("/v1", axum::routing::get(Self::health_handler))
            .route("/health", axum::routing::get(Self::health_handler))
            .route("/v1/models", axum::routing::get(Self::models_handler))
            .route("/models", axum::routing::get(Self::models_handler))
            .route("/v1/chat/completions", axum::routing::post(Self::chat_completions_handler))
            .route("/v1/completions", axum::routing::post(Self::completions_handler))
            .route("/v1/responses", axum::routing::post(Self::responses_handler))
            .route("/v1/messages", axum::routing::post(Self::messages_handler))
            .fallback(Self::fallback_handler)
            .with_state(self.clone());

        let addr = format!("{}:{}", self.config.host, self.config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        println!("ds4-rust server listening on http://{addr}");
        axum::serve(listener, app).await.map_err(|e| Ds4Error::Protocol(e.to_string()))?;
        Ok(())
    }




    async fn health_handler() -> Json<serde_json::Value> {
        Json(serde_json::json!({"ok": true, "service": "ds4-rust"}))
    }

    async fn models_handler(State(server): State<Server>) -> Json<serde_json::Value> {
        Json(serde_json::from_str(&server.models_json()).unwrap())
    }

    async fn chat_completions_handler(State(server): State<Server>, body: String) -> std::result::Result<Response, StatusCode> {
        let req_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        server.handle_api(req_id, &body, ApiKind::ChatCompletions).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }

    async fn completions_handler(State(server): State<Server>, body: String) -> std::result::Result<Response, StatusCode> {
        let req_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        server.handle_api(req_id, &body, ApiKind::Completions).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }

    async fn responses_handler(State(server): State<Server>, body: String) -> std::result::Result<Response, StatusCode> {
        let req_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        server.handle_api(req_id, &body, ApiKind::Responses).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }

    async fn messages_handler(State(server): State<Server>, body: String) -> std::result::Result<Response, StatusCode> {
        let req_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        server.handle_api(req_id, &body, ApiKind::Messages).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
    }

    async fn fallback_handler() -> (StatusCode, Json<serde_json::Value>) {
        (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "not found"})))
    }

    async fn handle_api(&self, req_id: u64, raw: &str, api: ApiKind) -> Result<Response> {
        let started = Instant::now();
        let mut request = RequestEnvelope::from_http(api, raw);
        let replay_hydrated = self.hydrate_tool_replay(&mut request)?;
        trace_request(
            req_id,
            "api.parsed",
            &started,
            &format!(
                "api={} tools={} tool_results={} max_tokens={}",
                match api {
                    ApiKind::ChatCompletions => "chat",
                    ApiKind::Responses => "responses",
                    ApiKind::Completions => "completions",
                    ApiKind::Messages => "messages",
                },
                request.has_tools,
                request.has_tool_results,
                request.max_output_tokens
            ),
        );
        // #region debug-point A:request-parse
        report_debug_event(
            "A",
            "server.rs:handle_api",
            "[DEBUG] parsed request envelope",
            &format!(
                concat!(
                    "{{",
                    "\"api\":\"{}\",",
                    "\"stream\":{},",
                    "\"has_tools\":{},",
                    "\"has_tool_results\":{},",
                    "\"has_real_model\":{},",
                    "\"last_tool_call_id\":{},",
                    "\"last_tool_result\":{},",
                    "\"prompt\":\"{}\"",
                    "}}"
                ),
                match api {
                    ApiKind::ChatCompletions => "chat.completions",
                    ApiKind::Responses => "responses",
                    ApiKind::Completions => "completions",
                    ApiKind::Messages => "messages",
                },
                request.stream,
                request.has_tools,
                request.has_tool_results,
                self.engine.has_real_model(),
                debug_json_opt(request.last_tool_call_id.as_deref()),
                debug_json_opt(request.last_tool_result.as_deref()),
                json_escape_min(&request.prompt)
            ),
        );
        // #endregion
        let continuation = self
            .continuations
            .lock()
            .map_err(|_| Ds4Error::Protocol("continuation mutex poisoned".to_string()))?
            .restore(
                request.previous_response_id.as_deref(),
                request.conversation.as_deref(),
                request.last_tool_call_id.as_deref(),
            );

        let continuation_hit = continuation.is_some();
        trace_request(
            req_id,
            "api.continuation",
            &started,
            &format!(
                "hit={} previous_response_id={} conversation={} tool_call_id={} replay_hydrated={}",
                continuation_hit,
                request.previous_response_id.is_some(),
                request.conversation.is_some(),
                request.last_tool_call_id.is_some(),
                replay_hydrated
            ),
        );
        let mut session = self
            .session
            .lock()
            .map_err(|_| Ds4Error::Protocol("session mutex poisoned".to_string()))?;

        if let Some(entry) = &continuation {
            session.load_snapshot(&entry.snapshot);
            trace_request(req_id, "api.snapshot", &started, "loaded continuation snapshot");
        }

        if should_short_circuit_tool_planning(&request) {
            let prompt = latest_user_text(&request.prompt);
            let (tool_name, tool_arguments) = choose_tool_call(&request, json_escape_min);
            let response = ResponseEnvelope::new_tool_call(
                api,
                0,
                prompt.chars().count(),
                continuation.is_some(),
                continuation_hit,
                tool_name.clone(),
                tool_arguments,
            );
            let snapshot = session.save_snapshot();
            self.store_visible_kv_entry(&session, api, &response, &snapshot, &Tokens::default())?;
            self.remember_continuation(&request, &response, snapshot)?;
            trace_request(
                req_id,
                "api.short_circuit_tool",
                &started,
                &format!("tool={} stream={}", tool_name, request.stream),
            );
            return if request.stream {
                Ok(Sse::new(tokio_stream::iter(vec![Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(response.to_sse()))])).into_response())
            } else {
                Ok(Json(serde_json::from_str::<serde_json::Value>(&response.to_json()).unwrap()).into_response())
            };
        }

        if let Some(reason) = untrusted_generation_reason(&self.engine) {
            trace_request(req_id, "api.gated", &started, reason);
            return Ok((StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": {
                    "message": reason,
                    "type": "service_unavailable",
                    "code": "untrusted_generation"
                }
            }))).into_response());
        }

        let think_mode = if request.stream {
            ThinkMode::High
        } else {
            ThinkMode::None
        };
        let rendered = render_chat_prompt(&self.engine, &request.system, &request.prompt, think_mode);
        trace_request(
            req_id,
            "api.rendered",
            &started,
            &format!("tokens={}", rendered.len()),
        );
        let stats = session.sync(&rendered)?;
        trace_request(
            req_id,
            "api.synced",
            &started,
            &format!(
                "cached_tokens={} replay_tokens={} rebuilt={}",
                stats.cached_tokens, stats.replay_tokens, stats.rebuilt
            ),
        );
        let (response, stream) = if request.stream && matches!(api, ApiKind::ChatCompletions) && self.engine.has_real_model() {
            let (response, stream) = stream_chat_completion_sse(
                &self.engine,
                &mut session,
                stats.cached_tokens,
                stats.replay_tokens,
                stats.rebuilt,
                continuation_hit,
                generation_budget(&self.engine, &request),
            )?;
            trace_request(
                req_id,
                "api.generated",
                &started,
                &format!("chars={}", response.message.chars().count()),
            );
            (response, Some(stream))
        } else {
            let generated_text = preview_or_model_reply(
                &self.engine,
                &request,
                &mut session,
                generation_budget(&self.engine, &request),
            )?;
            trace_request(
                req_id,
                "api.generated",
                &started,
                &format!("chars={}", generated_text.chars().count()),
            );
            (ResponseEnvelope::new_message(
                api,
                stats.cached_tokens,
                stats.replay_tokens,
                stats.rebuilt,
                continuation_hit,
                generated_text,
            ), None)
        };
        let snapshot = session.save_snapshot();

        self.store_visible_kv_entry(&session, api, &response, &snapshot, &rendered)?;

        self.remember_continuation(&request, &response, snapshot)?;
        trace_request(
            req_id,
            "api.response_ready",
            &started,
            &format!("continuation_hit={}",  continuation_hit),
        );

        if request.stream && !matches!(api, ApiKind::ChatCompletions) {
            Ok(Sse::new(tokio_stream::iter(vec![Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(response.to_sse()))])).into_response())
        } else if request.stream {
            if let Some(stream) = stream {
                Ok(Sse::new(stream).into_response())
            } else {
                Ok(Sse::new(tokio_stream::iter(vec![Ok::<_, std::convert::Infallible>(axum::response::sse::Event::default().data(response.to_sse()))])).into_response())
            }
        } else {
            Ok(Json(serde_json::from_str::<serde_json::Value>(&response.to_json()).unwrap()).into_response())
        }
    }

    fn store_visible_kv_entry(
        &self,
        session: &Session,
        api: ApiKind,
        response: &ResponseEnvelope,
        snapshot: &crate::types::SessionSnapshot,
        rendered: &Tokens,
    ) -> Result<()> {
        let kv_key = format!("visible:{}:{}", response.id, rendered.len());
        let now_unix = unix_time_secs_now();
        let tool_replay = match (response.tool_call_id(), response.tool_replay_block()) {
            (Some(tool_call_id), Some(sampled_block)) => vec![KvToolReplayEntry {
                tool_call_id: tool_call_id.to_string(),
                sampled_block,
            }],
            _ => Vec::new(),
        };
        self.kv
            .lock()
            .map_err(|_| Ds4Error::Protocol("kv mutex poisoned".to_string()))?
            .store(
                kv_key,
                KvEntry {
                    reason: match api {
                        ApiKind::ChatCompletions => "chat".to_string(),
                        ApiKind::Responses => "responses".to_string(),
                        ApiKind::Completions => "completions".to_string(),
                        ApiKind::Messages => "messages".to_string(),
                    },
                    rendered_text: session.render_tokens(rendered),
                    tokens: rendered.clone(),
                    snapshot: snapshot.clone(),
                    tool_replay,
                    ctx_size: session.ctx(),
                    hit_count: 0,
                    created_at_unix: now_unix,
                    last_used_at_unix: now_unix,
                },
            )
    }

    fn hydrate_tool_replay(&self, request: &mut RequestEnvelope) -> Result<bool> {
        if !request.has_tool_results {
            return Ok(false);
        }
        let Some(tool_call_id) = request.last_tool_call_id.as_deref() else {
            return Ok(false);
        };
        let Some(sampled_block) = self
            .kv
            .lock()
            .map_err(|_| Ds4Error::Protocol("kv mutex poisoned".to_string()))?
            .find_tool_replay(tool_call_id)?
        else {
            return Ok(false);
        };
        let rewritten = rewrite_prompt_with_tool_replay(&request.prompt, tool_call_id, &sampled_block);
        if rewritten == request.prompt {
            return Ok(false);
        }
        request.prompt = rewritten;
        Ok(true)
    }

    fn remember_continuation(
        &self,
        request: &RequestEnvelope,
        response: &ResponseEnvelope,
        snapshot: crate::types::SessionSnapshot,
    ) -> Result<()> {
        let visible_text = format!("system={}\nuser={}\n", request.system, request.prompt);
        self.continuations
            .lock()
            .map_err(|_| Ds4Error::Protocol("continuation mutex poisoned".to_string()))?
            .remember(ContinuationEntry::new(
                response.id.clone(),
                visible_text,
                request.conversation.clone(),
                response
                    .tool_call_id()
                    .map(str::to_string)
                    .or_else(|| request.last_tool_call_id.clone()),
                snapshot,
            ));
        Ok(())
    }

    fn models_json(&self) -> String {
        format!(
            concat!(
                "{{\"object\":\"list\",\"data\":[{{",
                "\"id\":\"{}\",\"object\":\"model\",\"created\":0,\"owned_by\":\"ds4-rust\"",
                "}}]}}"
            ),
            json_escape_min(DEFAULT_MODEL_ID)
        )
    }
}

fn trace_request(req_id: u64, stage: &str, started: &Instant, detail: &str) {
    if !trace_enabled() {
        return;
    }
    println!(
        "ds4-rust trace req={} stage={} elapsed_ms={} detail={}",
        req_id,
        stage,
        started.elapsed().as_millis(),
        detail
    );
}

fn trace_enabled() -> bool {
    std::env::var("DS4_TRACE")
        .or_else(|_| std::env::var("DS4_TRACE_HTTP"))
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}









fn stream_chat_completion_sse(
    engine: &Arc<Engine>,
    session: &mut Session,
    cached_tokens: usize,
    replay_tokens: usize,
    rebuilt: bool,
    continuation_hit: bool,
    generation_budget: usize,
) -> Result<(ResponseEnvelope, impl tokio_stream::Stream<Item = std::result::Result<axum::response::sse::Event, std::convert::Infallible>>)> {
    let response = ResponseEnvelope::new_message(
        ApiKind::ChatCompletions,
        cached_tokens,
        replay_tokens,
        rebuilt,
        continuation_hit,
        "",
    );
    let mut message = String::new();
    
    let engine = engine.clone();
    let id = response.id.clone();
    let created = response.created;
    
    // Actually we need to execute the generation blocking the current thread or spawn a blocking task.
    // For simplicity, we just generate all tokens and then return them as a stream.
    // In a real async environment we would yield.
    let mut events = Vec::new();
    events.push(Ok(axum::response::sse::Event::default().data(format!(
        "{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\"}},\"finish_reason\":null}}]}}",
        json_escape_min(&id), created, json_escape_min(crate::protocol::DEFAULT_MODEL_ID)
    ))));
    
    for _ in 0..generation_budget {
        let token = session.argmax();
        session.decode_next(token)?;
        let piece = engine.token_text(token);
        if piece.is_empty() || looks_like_terminal_token(&piece) {
            continue;
        }
        message.push_str(&piece);
        events.push(Ok(axum::response::sse::Event::default().data(format!(
            "{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}},\"finish_reason\":null}}]}}",
            json_escape_min(&id), created, json_escape_min(crate::protocol::DEFAULT_MODEL_ID), json_escape_min(&piece)
        ))));
    }
    
    events.push(Ok(axum::response::sse::Event::default().data(format!(
        "{{\"id\":\"{}\",\"object\":\"chat.completion.chunk\",\"created\":{},\"model\":\"{}\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}",
        json_escape_min(&id), created, json_escape_min(crate::protocol::DEFAULT_MODEL_ID)
    ))));
    events.push(Ok(axum::response::sse::Event::default().data("[DONE]")));

    let final_response = ResponseEnvelope::new_message(
        ApiKind::ChatCompletions,
        cached_tokens,
        replay_tokens,
        rebuilt,
        continuation_hit,
        message,
    );
    Ok((final_response, tokio_stream::iter(events)))
}








fn looks_like_terminal_token(piece: &str) -> bool {
    let trimmed = piece.trim();
    trimmed.is_empty()
        || matches!(
            trimmed,
            "<｜end▁of▁sentence｜>" | "</think>" | "<think>" | "<think max>"
        )
}

fn report_debug_event(hypothesis_id: &str, location: &str, msg: &str, data_json: &str) {
    let Ok(env_text) = std::fs::read_to_string(DEBUG_ENV_PATH) else {
        return;
    };
    let mut url = "http://127.0.0.1:7777/event".to_string();
    let mut session_id = "opencode-openai-compat".to_string();
    for line in env_text.lines() {
        if let Some(value) = line.strip_prefix("DEBUG_SERVER_URL=") {
            url = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("DEBUG_SESSION_ID=") {
            session_id = value.trim().to_string();
        }
    }
    let Some(authority) = url.strip_prefix("http://") else {
        return;
    };
    let (host_port, path) = authority.split_once('/').unwrap_or((authority, "event"));
    let payload = format!(
        concat!(
            "{{",
            "\"sessionId\":\"{}\",",
            "\"runId\":\"pre-fix\",",
            "\"hypothesisId\":\"{}\",",
            "\"location\":\"{}\",",
            "\"msg\":\"{}\",",
            "\"data\":{}",
            "}}"
        ),
        json_escape_min(&session_id),
        json_escape_min(hypothesis_id),
        json_escape_min(location),
        json_escape_min(msg),
        data_json
    );
    let request = format!(
        "POST /{} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path,
        host_port,
        payload.len(),
        payload
    );
    if let Ok(mut socket) = TcpStream::connect(host_port) {
        let _ = socket.write_all(request.as_bytes());
        let _ = socket.flush();
    }
}







fn json_escape_min(input: &str) -> String {
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

fn debug_json_opt(input: Option<&str>) -> String {
    input
        .map(|value| format!("\"{}\"", json_escape_min(value)))
        .unwrap_or_else(|| "null".to_string())
}

fn untrusted_generation_reason(engine: &Engine) -> Option<&'static str> {
    generation_gate_reason(engine.has_real_model(), engine.supports_trustworthy_generation())
}

fn generation_gate_reason(
    has_real_model: bool,
    supports_trustworthy_generation: bool,
) -> Option<&'static str> {
    if !has_real_model {
        return Some("ds4-rs has no loaded GGUF model");
    }
    if !supports_trustworthy_generation {
        return Some(
            "ds4-rs inference path still lacks required bound blocks or FFN weights for this model; continue implementing full inference",
        );
    }
    None
}

fn generation_budget(engine: &Engine, request: &RequestEnvelope) -> usize {
    if request.max_output_tokens > 0 {
        return request.max_output_tokens.min(32);
    }
    default_generation_budget(engine.has_real_model(), request.has_tool_results)
}

fn default_generation_budget(has_real_model: bool, has_tool_results: bool) -> usize {
    if has_real_model {
        if has_tool_results {
            // Tool-result follow-ups need enough room to summarize or continue.
            12
        } else {
            8
        }
    } else {
        24
    }
}

fn rewrite_prompt_with_tool_replay(prompt: &str, tool_call_id: &str, sampled_block: &str) -> String {
    let tool_line_prefix = format!("Tool[{tool_call_id}]:");
    let assistant_line_prefix = format!("AssistantToolCall[{tool_call_id}]");
    let tool_result_prefix = "<tool_result>";
    let mut out = Vec::new();
    let mut replaced = false;
    let mut inserted = false;
    for line in prompt.lines() {
        if line.starts_with(&assistant_line_prefix) {
            if !replaced {
                out.push(sampled_block.to_string());
                replaced = true;
            }
            continue;
        }
        if (line.starts_with(&tool_line_prefix) || line.starts_with(tool_result_prefix))
            && !replaced
            && !inserted
        {
            out.push(sampled_block.to_string());
            inserted = true;
        }
        out.push(line.to_string());
    }
    if !replaced && !inserted {
        return prompt.to_string();
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::RequestTool;

    #[test]
    fn uses_fixed_openai_model_id() {
        let engine = Engine::open(crate::engine::EngineOptions {
            model_path: "/definitely/missing.gguf".into(),
            ..crate::engine::EngineOptions::default()
        })
        .unwrap();
        let server = Server::new(engine, ServerConfig::default()).unwrap();
        let json = server.models_json();
        assert!(json.contains(&format!("\"id\":\"{}\"", DEFAULT_MODEL_ID)));
    }

    #[test]
    fn generation_gate_reason_matches_model_trust_state() {
        assert_eq!(
            generation_gate_reason(false, false),
            Some("ds4-rs has no loaded GGUF model")
        );
        assert_eq!(
            generation_gate_reason(true, false),
            Some(
                "ds4-rs inference path is not trustworthy for this model yet; restart with --quality or continue implementing full inference"
            )
        );
        assert_eq!(generation_gate_reason(true, true), None);
    }

    #[test]
    fn prefers_request_token_budget_when_present() {
        let request = RequestEnvelope {
            api: ApiKind::Responses,
            system: String::new(),
            prompt: String::new(),
            previous_response_id: None,
            conversation: None,
            available_tools: Vec::new(),
            has_tools: false,
            has_tool_results: false,
            primary_tool_name: None,
            primary_tool_arg_name: None,
            last_tool_call_id: None,
            last_tool_result: None,
            stream: false,
            max_output_tokens: 3,
        };
        let engine = Engine::open(crate::engine::EngineOptions {
            model_path: "/definitely/missing.gguf".into(),
            ..crate::engine::EngineOptions::default()
        })
        .unwrap();
        assert_eq!(generation_budget(&engine, &request), 3);
    }

    #[test]
    fn gives_tool_result_followups_more_generation_budget() {
        assert_eq!(default_generation_budget(true, false), 8);
        assert_eq!(default_generation_budget(true, true), 12);
        assert_eq!(default_generation_budget(false, false), 24);
    }

    #[test]
    fn falls_back_for_empty_or_nearly_empty_generated_replies() {
        assert!(reply::should_fallback_generated_reply(""));
        assert!(reply::should_fallback_generated_reply(" \n\t "));
        assert!(reply::should_fallback_generated_reply("..."));
        assert!(reply::should_fallback_generated_reply("?"));
        assert!(reply::should_fallback_generated_reply("a"));
        assert!(reply::should_fallback_generated_reply("好"));
    }

    #[test]
    fn keeps_brief_but_meaningful_generated_replies() {
        assert!(!reply::should_fallback_generated_reply("ok"));
        assert!(!reply::should_fallback_generated_reply("好的"));
        assert!(!reply::should_fallback_generated_reply("yes."));
    }

    #[test]
    fn falls_back_for_think_markers_garbled_text_and_repetition() {
        assert!(reply::should_fallback_generated_reply("<think>still thinking"));
        assert!(reply::should_fallback_generated_reply("abcabcabc"));
        assert!(reply::should_fallback_generated_reply("好的 好的 好的"));
        assert!(reply::should_fallback_generated_reply("��a"));
        assert!(reply::should_fallback_generated_reply("� � � ok"));
    }

    #[test]
    fn keeps_non_repetitive_non_garbled_replies() {
        assert!(!reply::should_fallback_generated_reply("第一行\n第二行"));
        assert!(!reply::should_fallback_generated_reply("ok, let me continue."));
        assert!(!reply::should_fallback_generated_reply("好的，我继续处理。"));
    }

    #[test]
    fn short_circuits_tool_planning_only_for_explicit_tool_tasks() {
        let request = RequestEnvelope {
            api: ApiKind::Responses,
            system: String::new(),
            prompt: "User: list files in current directory".to_string(),
            previous_response_id: None,
            conversation: None,
            available_tools: vec![RequestTool {
                name: "read_file".to_string(),
                first_arg_name: Some("file_path".to_string()),
                property_names: vec!["file_path".to_string()],
            }],
            has_tools: true,
            has_tool_results: false,
            primary_tool_name: Some("read_file".to_string()),
            primary_tool_arg_name: Some("file_path".to_string()),
            last_tool_call_id: None,
            last_tool_result: None,
            stream: false,
            max_output_tokens: 0,
        };
        assert!(should_short_circuit_tool_planning(&request));

        let mut follow_up = request.clone();
        follow_up.has_tool_results = true;
        assert!(!should_short_circuit_tool_planning(&follow_up));

        let mut plain_chat = request;
        plain_chat.prompt = "User: say hello".to_string();
        plain_chat.has_tool_results = false;
        assert!(!should_short_circuit_tool_planning(&plain_chat));
    }

    #[test]
    fn bash_tool_arguments_include_description_when_schema_requires_it() {
        let tool = RequestTool {
            name: "bash".to_string(),
            first_arg_name: Some("command".to_string()),
            property_names: vec!["command".to_string(), "description".to_string()],
        };
        let args =
            tools::build_tool_arguments_for_tool(&tool, "list files in current directory", json_escape_min);
        assert!(args.contains("\"command\":\"ls -la\""));
        assert!(args.contains("\"description\":\"Runs requested shell command\""));
    }

    #[test]
    fn preview_reply_is_coherent_for_ping_messages() {
        let request = RequestEnvelope {
            api: ApiKind::ChatCompletions,
            system: String::new(),
            prompt: "User: 你要回答我啊".to_string(),
            previous_response_id: None,
            conversation: None,
            available_tools: Vec::new(),
            has_tools: false,
            has_tool_results: false,
            primary_tool_name: None,
            primary_tool_arg_name: None,
            last_tool_call_id: None,
            last_tool_result: None,
            stream: true,
            max_output_tokens: 0,
        };
        let reply = reply::preview_text_reply(&request);
        assert!(reply.contains("我在"));
    }

    #[test]
    fn preview_reply_summarizes_tool_results() {
        let request = RequestEnvelope {
            api: ApiKind::Responses,
            system: String::new(),
            prompt: "User: 继续".to_string(),
            previous_response_id: None,
            conversation: None,
            available_tools: Vec::new(),
            has_tools: false,
            has_tool_results: true,
            primary_tool_name: None,
            primary_tool_arg_name: None,
            last_tool_call_id: Some("call_1".to_string()),
            last_tool_result: Some("第一行\n第二行".to_string()),
            stream: false,
            max_output_tokens: 0,
        };
        let reply = reply::preview_text_reply(&request);
        assert!(reply.contains("我已经拿到结果了"));
        assert!(reply.contains("第一行"));
    }

    #[test]
    fn rewrites_canonical_tool_call_line_with_exact_replay_block() {
        let prompt = concat!(
            "User: list files\n",
            "AssistantToolCall[call_1] function bash({\"command\":\"ls -la\"})\n",
            "<tool_result>file1\nfile2</tool_result>"
        );
        let rewritten = rewrite_prompt_with_tool_replay(
            prompt,
            "call_1",
            "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n<｜DSML｜parameter name=\"description\" string=\"true\">Runs requested shell command</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        assert!(rewritten.contains("<｜DSML｜tool_calls>"));
        assert!(rewritten.contains("Runs requested shell command"));
        assert!(!rewritten.contains("AssistantToolCall[call_1] function bash({\"command\":\"ls -la\"})"));
    }

    #[test]
    fn inserts_exact_replay_block_before_tool_result_when_missing() {
        let prompt = "User: list files\n<tool_result>file1\nfile2</tool_result>";
        let rewritten = rewrite_prompt_with_tool_replay(
            prompt,
            "call_1",
            "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        );
        let expected = concat!(
            "User: list files\n",
            "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">pwd</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n",
            "<tool_result>file1\nfile2</tool_result>"
        );
        assert_eq!(rewritten, expected);
    }
}
