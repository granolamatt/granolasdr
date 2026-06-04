pub mod dict;
pub mod protocol;
pub mod server;
pub mod subs;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

use axum::routing::get;
use tower_http::services::ServeDir;

use crate::dict::Dict;
use crate::server::{ws_handler, AppState};
use crate::subs::Subscriptions;

// ── Rust API ─────────────────────────────────────────────────────────────────

struct ServerHandle {
    thread: JoinHandle<()>,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

static SERVER: OnceLock<Mutex<Option<ServerHandle>>> = OnceLock::new();

/// Start the wsdict server in a background thread on the given port.
/// Blocks until the server is bound and listening, then returns.
/// state_file: optional path to persist uhd:config / uhd:channels across restarts.
pub fn start(port: u16, static_dir: &str, state_file: Option<&str>) -> Result<(), String> {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let static_dir = static_dir.to_string();
    let state_path = state_file.map(std::path::PathBuf::from);

    let thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("wsserver: failed to build tokio runtime");
        rt.block_on(async move {
            let state = Arc::new(AppState {
                dict: Dict::new_with_state(state_path),
                subs: Subscriptions::new(),
            });
            let app = axum::Router::new()
                .route("/ws", get(ws_handler))
                .nest_service("/", ServeDir::new(&static_dir))
                .with_state(state);
            let addr = SocketAddr::from(([127, 0, 0, 1], port));
            let listener = match tokio::net::TcpListener::bind(addr).await {
                Ok(l) => {
                    ready_tx.send(Ok(())).ok();
                    l
                }
                Err(e) => {
                    ready_tx.send(Err(e.to_string())).ok();
                    return;
                }
            };
            eprintln!("[wsserver] listening on http://{addr}");
            axum::serve(listener, app)
                .with_graceful_shutdown(async { shutdown_rx.await.ok(); })
                .await
                .ok();
            eprintln!("[wsserver] stopped");
        });
    });

    let result = ready_rx
        .recv()
        .map_err(|_| "wsserver thread exited before binding".to_string())
        .and_then(|r| r);

    if result.is_ok() {
        let mutex = SERVER.get_or_init(|| Mutex::new(None));
        *mutex.lock().unwrap() = Some(ServerHandle { thread, shutdown_tx });
    }
    result
}

/// Signal the server to shut down and wait for the background thread to exit.
pub fn stop() {
    if let Some(mutex) = SERVER.get() {
        if let Ok(mut guard) = mutex.lock() {
            if let Some(handle) = guard.take() {
                let _ = handle.shutdown_tx.send(());
                let _ = handle.thread.join();
            }
        }
    }
}

// ── C FFI ────────────────────────────────────────────────────────────────────

/// Start the wsdict server. Blocks until listening. Returns 0 on success, -1 on failure.
/// static_dir: path to static files (NULL → "static").
/// state_file: path for persistent dict state (NULL → no persistence).
#[no_mangle]
pub extern "C" fn wsdict_server_start(
    port: u16,
    static_dir: *const std::os::raw::c_char,
    state_file: *const std::os::raw::c_char,
) -> std::os::raw::c_int {
    let dir = if static_dir.is_null() {
        "static".to_string()
    } else {
        unsafe { std::ffi::CStr::from_ptr(static_dir) }
            .to_string_lossy()
            .into_owned()
    };
    let sf = if state_file.is_null() {
        None
    } else {
        Some(
            unsafe { std::ffi::CStr::from_ptr(state_file) }
                .to_string_lossy()
                .into_owned(),
        )
    };
    match start(port, &dir, sf.as_deref()) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[wsserver] start failed: {e}");
            -1
        }
    }
}

/// Stop the wsdict server and wait for it to exit.
#[no_mangle]
pub extern "C" fn wsdict_server_stop() {
    stop();
}
