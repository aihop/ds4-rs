use ds4_rust::{Engine, EngineOptions, Server, ServerConfig};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    if let Err(err) = run().await {
        tracing::error!("ds4-server-rust: {err}");
        std::process::exit(1);
    }
}

async fn run() -> ds4_rust::Result<()> {
    let mut engine_opts = EngineOptions::default();
    let mut cfg = ServerConfig::default();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                if let Some(value) = args.next() {
                    engine_opts.model_path = value.into();
                }
            }
            "--quality" => engine_opts.quality = true,
            "--host" => {
                if let Some(value) = args.next() {
                    cfg.host = value;
                }
            }
            "--port" => {
                if let Some(value) = args.next() {
                    cfg.port = value.parse().unwrap_or(cfg.port);
                }
            }
            "-c" | "--ctx" => {
                if let Some(value) = args.next() {
                    cfg.ctx_size = value.parse().unwrap_or(cfg.ctx_size);
                }
            }
            "--kv-disk-dir" => {
                if let Some(value) = args.next() {
                    cfg.kv_disk_dir = Some(value);
                }
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            _ => {}
        }
    }

    let engine = Engine::open(engine_opts)?;
    println!("{}", engine.summary());
    let server = Server::new(engine, cfg)?;
    server.listen_and_serve().await
}

fn print_help() {
    println!(
        "Usage: ds4-server [--host HOST] [--port PORT] [--ctx N] [--model FILE] [--quality] [--kv-disk-dir DIR]"
    );
}
