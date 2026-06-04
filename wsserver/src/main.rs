#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let port: u16 = std::env::var("WSDICT_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8765);

    let static_dir = std::env::var("WSDICT_STATIC").unwrap_or_else(|_| "static".into());

    let state_file = std::env::var("WSDICT_STATE").ok();
    wsserver::start(port, &static_dir, state_file.as_deref()).expect("failed to start wsserver");

    tokio::signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    tracing::info!("received shutdown signal");

    wsserver::stop();
}
