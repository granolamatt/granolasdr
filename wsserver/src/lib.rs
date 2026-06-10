pub mod dict;
pub mod protocol;
pub mod server;
pub mod subs;
pub mod tci;

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;

use axum::routing::get;
use tower_http::services::ServeDir;

use crate::dict::Dict;
use crate::server::{run_ttl_sweep, ws_handler, AppState};
use crate::subs::Subscriptions;

// ── Rust API ─────────────────────────────────────────────────────────────────

struct ServerHandle {
    thread: JoinHandle<()>,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

static SERVER: OnceLock<Mutex<Option<ServerHandle>>> = OnceLock::new();

/// Tokio runtime handle stored before block_on so TCI server can spawn tasks
/// into the same runtime (D1: set before rt.block_on, not inside async block).
static RUNTIME_HANDLE: OnceLock<tokio::runtime::Handle> = OnceLock::new();

/// AbortHandle for the TCI server task, stored so tci_server_stop() can cancel it.
static TCI_TASK: OnceLock<Mutex<Option<tokio::task::AbortHandle>>> = OnceLock::new();

/// Start the wsdict server in a background thread on the given port.
/// Blocks until the server is bound and listening, then returns.
pub fn start(port: u16, static_dir: &str, _state_file: Option<&str>) -> Result<(), String> {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let static_dir = static_dir.to_string();

    let thread = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("wsserver: failed to build tokio runtime");
        // D1: store handle BEFORE block_on so tci_server_start() can spawn
        // into this runtime without needing a second rt.
        let _ = RUNTIME_HANDLE.set(rt.handle().clone());
        rt.block_on(async move {
            let state = Arc::new(AppState {
                dict: Dict::new(),
                subs: Subscriptions::new(),
            });

            // Spawn TTL sweep with panic-restart loop.
            {
                let sweep_state = state.clone();
                tokio::spawn(async move {
                    loop {
                        let result =
                            tokio::spawn(run_ttl_sweep(sweep_state.clone())).await;
                        if let Err(e) = result {
                            tracing::warn!(error = ?e, "TTL sweep panicked, restarting in 1s");
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        }
                    }
                });
            }

            let app = axum::Router::new()
                .route("/ws", get(ws_handler))
                .nest_service("/", ServeDir::new(&static_dir))
                .with_state(state);
            let addr = SocketAddr::from(([0, 0, 0, 0], port));
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
/// state_file: reserved, pass NULL.
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

// ── TCI C FFI ────────────────────────────────────────────────────────────────

/// Start the TCI WebSocket server on `port` with `gain` for INT16 scaling.
/// Requires wsdict_server_start() to have been called first (sets RUNTIME_HANDLE).
/// Blocks until the server is bound. Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn tci_server_start(port: u16, gain: f32) -> std::os::raw::c_int {
    let handle = match RUNTIME_HANDLE.get() {
        Some(h) => h,
        None => {
            eprintln!("[TCI] wsdict_server_start must be called before tci_server_start");
            return -1;
        }
    };
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let join_handle = handle.spawn(crate::tci::run(port, gain, ready_tx));
    TCI_TASK
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .replace(join_handle.abort_handle());
    match ready_rx.recv() {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            eprintln!("[TCI] {e}");
            -1
        }
        Err(_) => {
            eprintln!("[TCI] server task exited before bind");
            -1
        }
    }
}

/// Abort the TCI server task.
#[no_mangle]
pub extern "C" fn tci_server_stop() {
    if let Some(mutex) = TCI_TASK.get() {
        if let Some(handle) = mutex.lock().unwrap().take() {
            handle.abort();
        }
    }
}

/// Push a block of float32 PCM samples for `trx` into the TCI audio accumulator.
/// When 480 samples accumulate, a binary audio frame is broadcast to subscribers.
/// Called from C++ audioWorker; never blocks.
/// Safety: `pcm` must point to `count` valid f32 values for the duration of the call.
#[no_mangle]
pub extern "C" fn tci_push_audio(trx: u32, pcm: *const f32, count: u32) {
    let slice = unsafe { std::slice::from_raw_parts(pcm, count as usize) };
    crate::tci::push_audio(trx, slice);
}

/// Broadcast S-meter level `dbfs` for `trx` to all connected TCI clients.
/// Called from C++ audioWorker every 100 frames (~1 Hz).
/// Source should be norm_ema_h_[sink_bins[trx]] (RF signal level, not audio RMS).
#[no_mangle]
pub extern "C" fn tci_push_smeter(trx: u32, dbfs: f32) {
    crate::tci::push_smeter(trx, dbfs);
}

/// Drain one VFO command from the queue into `*trx_out` and `*freq_out`.
/// Returns 1 if a command was available, 0 if the queue is empty.
/// Call in a loop to drain all pending commands.
/// Safety: trx_out and freq_out must be valid writable pointers.
#[no_mangle]
pub extern "C" fn tci_poll_vfo(
    trx_out: *mut u32,
    freq_out: *mut u64,
) -> std::os::raw::c_int {
    match crate::tci::poll_vfo() {
        Some((trx, freq)) => {
            unsafe {
                *trx_out = trx;
                *freq_out = freq;
            }
            1
        }
        None => 0,
    }
}
