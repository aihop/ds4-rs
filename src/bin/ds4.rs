use std::io::{self, Write};
use std::time::Instant;

use ds4_rust::{render_chat_prompt, Backend, Engine, EngineOptions, Session, ThinkMode};

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    if let Err(err) = run() {
        tracing::error!("ds4-rust: {err}");
        std::process::exit(1);
    }
}

fn run() -> ds4_rust::Result<()> {
    let mut opts = EngineOptions::default();
    let mut ctx_size = 32_768usize;
    let mut max_tokens = 24usize;
    let mut prompt: Option<String> = None;
    let mut system = "You are a helpful assistant".to_string();
    let mut think_mode = ThinkMode::High;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                if let Some(value) = args.next() {
                    opts.model_path = value.into();
                }
            }
            "-p" | "--prompt" => prompt = args.next(),
            "-c" | "--ctx" => {
                if let Some(value) = args.next() {
                    ctx_size = value.parse().unwrap_or(ctx_size);
                }
            }
            "--max-tokens" => {
                if let Some(value) = args.next() {
                    max_tokens = value.parse().unwrap_or(max_tokens);
                }
            }
            "--quality" => opts.quality = true,
            "--metal" => opts.backend = Backend::Metal,
            "--cuda" => opts.backend = Backend::Cuda,
            "--cpu" => opts.backend = Backend::Cpu,
            "--nothink" => think_mode = ThinkMode::None,
            "--think-max" => think_mode = ThinkMode::Max,
            "--think" => think_mode = ThinkMode::High,
            "--system" => {
                if let Some(value) = args.next() {
                    system = value;
                }
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
    }

    let engine = Engine::open(opts)?;
    println!("{}", engine.summary());
    let mut session = Session::create(engine.clone(), ctx_size)?;

    if let Some(prompt) = prompt {
        run_one_shot(&mut session, &system, &prompt, think_mode, max_tokens)?;
    } else {
        run_interactive(&mut session, &system, think_mode, max_tokens)?;
    }
    Ok(())
}

fn run_one_shot(
    session: &mut Session,
    system: &str,
    prompt: &str,
    think_mode: ThinkMode,
    max_tokens: usize,
) -> ds4_rust::Result<()> {
    ensure_trustworthy_generation(session.engine())?;
    // #region debug-point D:prompt-render
    debug_cli_event(
        "D",
        "src/bin/ds4.rs:run_one_shot:render-chat-prompt-start",
        "[DEBUG] render chat prompt start",
        format!(
            "{{\"system_chars\":{},\"prompt_chars\":{},\"think_mode\":{}}}",
            system.chars().count(),
            prompt.chars().count(),
            debug_json_string(&format!("{think_mode:?}"))
        ),
    );
    // #endregion
    let render_prompt_started = Instant::now();
    let tokens = render_chat_prompt(session.engine(), system, prompt, think_mode);
    // #region debug-point D:prompt-render
    debug_cli_event(
        "D",
        "src/bin/ds4.rs:run_one_shot:render-chat-prompt-done",
        "[DEBUG] render chat prompt done",
        format!(
            "{{\"prompt_tokens\":{},\"elapsed_ms\":{}}}",
            tokens.len(),
            render_prompt_started.elapsed().as_millis()
        ),
    );
    // #endregion
    // #region debug-point A:cli-prefill
    debug_cli_event(
        "A",
        "src/bin/ds4.rs:run_one_shot:prefill-start",
        "[DEBUG] cli prefill start",
        format!(
            "{{\"prompt_tokens\":{},\"max_tokens\":{}}}",
            tokens.len(),
            max_tokens
        ),
    );
    // #endregion
    let prefill_started = Instant::now();
    let stats = session.sync(&tokens)?;
    // #region debug-point A:cli-prefill
    debug_cli_event(
        "A",
        "src/bin/ds4.rs:run_one_shot:prefill-done",
        "[DEBUG] cli prefill done",
        format!(
            "{{\"cached_tokens\":{},\"replay_tokens\":{},\"rebuilt\":{},\"elapsed_ms\":{}}}",
            stats.cached_tokens,
            stats.replay_tokens,
            stats.rebuilt,
            prefill_started.elapsed().as_millis()
        ),
    );
    // #endregion
    println!(
        "cached={} replay={} rebuilt={}",
        stats.cached_tokens, stats.replay_tokens, stats.rebuilt
    );
    // #region debug-point B:cli-generate
    debug_cli_event(
        "B",
        "src/bin/ds4.rs:run_one_shot:generate-start",
        "[DEBUG] cli generate start",
        format!("{{\"max_tokens\":{},\"checkpoint_len\":{}}}", max_tokens, session.pos()),
    );
    // #endregion
    let generate_started = Instant::now();
    let generated = session.generate_argmax_tokens(max_tokens)?;
    // #region debug-point B:cli-generate
    debug_cli_event(
        "B",
        "src/bin/ds4.rs:run_one_shot:generate-done",
        "[DEBUG] cli generate done",
        format!(
            "{{\"generated_tokens\":{},\"elapsed_ms\":{},\"checkpoint_len\":{}}}",
            generated.len(),
            generate_started.elapsed().as_millis(),
            session.pos()
        ),
    );
    // #endregion
    // #region debug-point C:cli-render
    debug_cli_event(
        "C",
        "src/bin/ds4.rs:run_one_shot:render-start",
        "[DEBUG] cli render start",
        format!("{{\"generated_tokens\":{}}}", generated.len()),
    );
    // #endregion
    let rendered = session.render_tokens(&generated);
    // #region debug-point C:cli-render
    debug_cli_event(
        "C",
        "src/bin/ds4.rs:run_one_shot:render-done",
        "[DEBUG] cli render done",
        format!(
            "{{\"chars\":{},\"preview\":{}}}",
            rendered.chars().count(),
            debug_json_string(&rendered.chars().take(80).collect::<String>())
        ),
    );
    // #endregion
    println!("assistant> {}", rendered);
    Ok(())
}

fn run_interactive(
    session: &mut Session,
    system: &str,
    think_mode: ThinkMode,
    max_tokens: usize,
) -> ds4_rust::Result<()> {
    println!("ds4-rust interactive preview. Type /exit to quit.");
    let stdin = io::stdin();
    loop {
        print!("ds4> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "/exit" | "/quit") {
            break;
        }
        ensure_trustworthy_generation(session.engine())?;
        let tokens = render_chat_prompt(session.engine(), system, line, think_mode);
        let stats = session.sync(&tokens)?;
        println!(
            "cached={} replay={} rebuilt={}",
            stats.cached_tokens, stats.replay_tokens, stats.rebuilt
        );
        let generated = session.generate_argmax_tokens(max_tokens)?;
        println!("assistant> {}", session.render_tokens(&generated));
    }
    Ok(())
}

fn ensure_trustworthy_generation(engine: &Engine) -> ds4_rust::Result<()> {
    if !engine.has_real_model() {
        return Err(ds4_rust::Ds4Error::Unavailable(
            "ds4-rs has no loaded GGUF model".to_string(),
        ));
    }
    if !engine.supports_trustworthy_generation() {
        return Err(ds4_rust::Ds4Error::Unavailable(
            "ds4-rs inference path still lacks required bound blocks or FFN weights for this model; continue implementing full inference".to_string(),
        ));
    }
    Ok(())
}

fn print_help() {
    println!(
        "Usage: ds4 [-p PROMPT] [--model FILE] [--ctx N] [--max-tokens N] [--quality] [--metal|--cuda|--cpu] [--think|--think-max|--nothink]"
    );
}

fn debug_cli_event(hypothesis_id: &str, location: &str, msg: &str, data_json: String) {
    // #region debug-point A:network-report
    let event = format!(
        "{{\"sessionId\":\"slow-prefill-startup\",\"runId\":\"pre-fix\",\"hypothesisId\":\"{}\",\"location\":{},\"msg\":{},\"data\":{},\"ts\":{}}}",
        hypothesis_id,
        debug_json_string(location),
        debug_json_string(msg),
        data_json,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default()
    );
    let _ = std::process::Command::new("python3")
        .arg("-c")
        .arg(
            "import pathlib, urllib.request, sys; p=pathlib.Path('.dbg/slow-prefill-startup.env'); u='http://127.0.0.1:7777/event';\n\
try:\n\
 c=p.read_text();\n\
 u=next((line.split('=',1)[1].strip() for line in c.splitlines() if line.startswith('DEBUG_SERVER_URL=')), u)\n\
except Exception:\n\
 pass\n\
urllib.request.urlopen(urllib.request.Request(u, data=sys.argv[1].encode(), headers={'Content-Type':'application/json'}), timeout=1).read()",
        )
        .arg(event)
        .output();
    // #endregion
}

fn debug_json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}
