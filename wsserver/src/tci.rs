//! TCI (Transceiver Control Interface) WebSocket server.
//!
//! Spec: Expert Electronics TCI v2.x — https://github.com/ExpertSDR3/TCI
//!
//! Binary audio frame layout matches WSJT-X 3.0.1 TCITransceiver.hpp Data_Stream:
//!   Bytes 0–63:  header, 16 × u32 little-endian (AudioHeaderSize = 64)
//!     [0]  receiver    = trx index
//!     [1]  sampleRate  = 48000
//!     [2]  format      = 3 (float32)
//!     [3]  codec       = 0
//!     [4]  crc         = 0
//!     [5]  length      = float count (2 × sample count, stereo-interleaved)
//!     [6]  type        = 1 (RX_AUDIO_STREAM)
//!     [7–15] reserv[9] = 0
//!   Bytes 64+:  float32 LE stereo-interleaved (L,R,L,R,...; mono: L=R)
//!
//! WSJT-X reads pStream->data as float* regardless of format field, and
//! uses pStream->length as float count.  store() takes source[i*2] (left
//! channel only), so stereo-interleaved mono produces the correct output.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;

// ── Constants ─────────────────────────────────────────────────────────────

pub const NUM_SINKS: usize = 4;
pub const AUDIO_VALID: usize = 240;   // C++ AUDIO_VALID: samples per push from audioWorker
pub const AUDIO_FRAME: usize = 480;   // 2 × AUDIO_VALID = 10 ms at 48 kHz

const CHAN_DEPTH: usize = 64;

/// Default VFO frequencies in Hz — must match HFChannelizer C++ defaults.
const DEFAULT_VFO_HZ: [u64; NUM_SINKS] = [14_074_000, 7_074_000, 3_573_000, 28_074_000];

// ── Message type ──────────────────────────────────────────────────────────

enum WsMsg {
    Binary(Vec<u8>),
    Text(String),
}

// ── Shared state ──────────────────────────────────────────────────────────

pub struct TciState {
    /// Per-sink float PCM accumulator; flushed at AUDIO_FRAME samples.
    audio_accum: [Mutex<Vec<f32>>; NUM_SINKS],
    /// Per-sink audio subscribers: (conn_id, Sender).
    audio_subs: [Mutex<Vec<(u64, mpsc::Sender<WsMsg>)>>; NUM_SINKS],
    /// All-connection senders for S-meter text broadcasts.
    all_subs: Mutex<Vec<(u64, mpsc::Sender<WsMsg>)>>,
    /// VFO commands from TCI client; consumed by C++ tciVfoWorker.
    vfo_queue: Mutex<VecDeque<(u32, u64)>>,
    /// Current VFO frequency per sink (Hz); updated on each VFO command.
    pub vfo_freqs: [AtomicU64; NUM_SINKS],
    /// Audio gain applied before float32 payload (from --tci-gain CLI flag).
    pub gain: f32,
    /// Total push_audio calls across all sinks (monotonic; for rate measurement).
    pub audio_push_count: AtomicU64,
}

static NEXT_CONN_ID: AtomicU64 = AtomicU64::new(0);
pub static TCI_STATE: OnceLock<Arc<TciState>> = OnceLock::new();

impl TciState {
    pub fn new(gain: f32) -> Arc<Self> {
        Arc::new(Self {
            audio_accum: std::array::from_fn(|_| {
                Mutex::new(Vec::with_capacity(AUDIO_FRAME * 2))
            }),
            audio_subs: std::array::from_fn(|_| Mutex::new(Vec::new())),
            all_subs: Mutex::new(Vec::new()),
            vfo_queue: Mutex::new(VecDeque::new()),
            vfo_freqs: std::array::from_fn(|i| AtomicU64::new(DEFAULT_VFO_HZ[i])),
            gain,
            audio_push_count: AtomicU64::new(0),
        })
    }
}

// ── Server entry point ────────────────────────────────────────────────────

/// Start the TCI WebSocket server on `port`.
/// Sends `ready_tx` after binding, then runs until the task is aborted.
pub async fn run(
    port: u16,
    gain: f32,
    ready_tx: std::sync::mpsc::Sender<Result<(), String>>,
) {
    let state = TCI_STATE.get_or_init(|| TciState::new(gain)).clone();

    let app = Router::new()
        .fallback(axum::routing::get(ws_handler))
        .with_state(state.clone());

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => {
            let _ = ready_tx.send(Ok(()));
            l
        }
        Err(e) => {
            let _ = ready_tx.send(Err(format!("bind :{port} failed: {e}")));
            return;
        }
    };

    let state_tick = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        interval.tick().await; // consume initial immediate tick
        let mut last_pushes = 0u64;
        let mut last_tick = tokio::time::Instant::now();
        loop {
            interval.tick().await;
            let s = &*state_tick;
            let now = tokio::time::Instant::now();
            let elapsed = now.duration_since(last_tick).as_secs_f64();
            let pushes = s.audio_push_count.load(Ordering::Relaxed);
            let delta = pushes.saturating_sub(last_pushes);
            // Each push delivers AUDIO_VALID samples to one sink; divide by NUM_SINKS
            // to get per-sink block rate, then multiply by samples/block.
            let sps = if elapsed > 0.0 {
                (delta as f64 * AUDIO_VALID as f64) / (NUM_SINKS as f64 * elapsed)
            } else {
                0.0
            };
            last_pushes = pushes;
            last_tick = now;

            let conns = s.all_subs.lock().unwrap().len();
            let freqs: Vec<String> = (0..NUM_SINKS)
                .map(|trx| format!("trx{}={}", trx, s.vfo_freqs[trx].load(Ordering::Acquire)))
                .collect();
            println!("[TCI] clients={conns} rate={sps:.0}sps {}", freqs.join(" "));
        }
    });

    axum::serve(listener, app).await.ok();
}

// ── WebSocket handler ──────────────────────────────────────────────────────

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<TciState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_connection(socket, state))
}

async fn handle_connection(socket: WebSocket, state: Arc<TciState>) {
    let conn_id = NEXT_CONN_ID.fetch_add(1, Ordering::Relaxed);
    let (mut ws_sink, mut ws_stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<WsMsg>(CHAN_DEPTH);

    state.all_subs.lock().unwrap().push((conn_id, tx.clone()));

    let init = build_init_sequence(&state);
    if ws_sink.send(Message::Text(init)).await.is_err() {
        cleanup_conn(&state, conn_id);
        return;
    }

    let mut write_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let ws_msg = match msg {
                WsMsg::Binary(data) => Message::Binary(data),
                WsMsg::Text(text) => Message::Text(text),
            };
            if ws_sink.send(ws_msg).await.is_err() {
                break;
            }
        }
    });

    // No auto-start: clients must send audio_start:{trx}; explicitly.
    // Auto-subscribing all sinks would flood the channel and cause try_send
    // drops on control echoes (VFO, modulation) that WSJT-X depends on.

    loop {
        tokio::select! {
            result = ws_stream.next() => {
                match result {
                    Some(Ok(Message::Text(text))) => {
                        for cmd in text.split(';').map(str::trim).filter(|s| !s.is_empty()) {
                            dispatch_cmd(cmd, &state, conn_id, &tx);
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(_))) | Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::debug!("[TCI] conn={conn_id} read error: {e}");
                        break;
                    }
                }
            }
            _ = &mut write_task => break,
        }
    }

    cleanup_conn(&state, conn_id);
    write_task.abort();
}

fn dispatch_cmd(
    cmd: &str,
    state: &Arc<TciState>,
    conn_id: u64,
    tx: &mpsc::Sender<WsMsg>,
) {
    // TCI commands are case-insensitive; WSJT-X sends lowercase.
    let cmd_lower = cmd.to_lowercase();
    let cl = cmd_lower.as_str();

    if let Some(rest) = cl.strip_prefix("audio_start:") {
        if let Ok(trx) = rest.trim().parse::<u32>() {
            subscribe_audio(state, conn_id, tx, trx);
            // Echo back so WSJT-X Cmd_AudioStart sets stream_audio_=true.
            reliable_send(tx, format!("audio_start:{trx};"));
        }
    } else if let Some(rest) = cl.strip_prefix("audio_stop:") {
        if let Ok(trx) = rest.trim().parse::<u32>() {
            unsubscribe_audio(state, conn_id, trx);
        }
    } else if let Some(rest) = cl.strip_prefix("vfo:") {
        parse_vfo_cmd(state, rest);
        // Echo back so WSJT-X Cmd_VFO updates rx_frequency_.  For same-band
        // changes (!band_change) this also calls tci_done7(), letting
        // do_frequency() exit mysleep7(2000) early.  Use reliable_send so the
        // echo is never lost to channel backpressure; a dropped echo would leave
        // rx_frequency_ stale → band_change2=true → "TCI failed set rxfreq".
        reliable_send(tx, format!("vfo:{rest};"));
    } else if cl.starts_with("split_enable:") || cl.starts_with("modulation:") {
        // Echo back so rig_split() mysleep5(500) and do_frequency mode
        // mysleep7(1000) exit immediately rather than timing out.
        reliable_send(tx, format!("{cl};"));
    }
    // All other commands (smeter polls, rx_sensors_enable, etc.) silently accepted.
}

/// Send a control message reliably: spawns an async task that waits for channel
/// space rather than dropping.  Only used for small control echoes (not audio).
fn reliable_send(tx: &mpsc::Sender<WsMsg>, msg: String) {
    let tx = tx.clone();
    tokio::spawn(async move {
        let _ = tx.send(WsMsg::Text(msg)).await;
    });
}

// ── Subscriber management ──────────────────────────────────────────────────

fn subscribe_audio(state: &TciState, conn_id: u64, tx: &mpsc::Sender<WsMsg>, trx: u32) {
    if trx as usize >= NUM_SINKS {
        tracing::warn!("[TCI] AUDIO_START: trx={trx} out of range");
        return;
    }
    let mut subs = state.audio_subs[trx as usize].lock().unwrap();
    if !subs.iter().any(|(id, _)| *id == conn_id) {
        subs.push((conn_id, tx.clone()));
    }
}

fn unsubscribe_audio(state: &TciState, conn_id: u64, trx: u32) {
    if trx as usize >= NUM_SINKS {
        return;
    }
    state.audio_subs[trx as usize].lock().unwrap().retain(|(id, _)| *id != conn_id);
}

fn cleanup_conn(state: &TciState, conn_id: u64) {
    state.all_subs.lock().unwrap().retain(|(id, _)| *id != conn_id);
    for subs in &state.audio_subs {
        subs.lock().unwrap().retain(|(id, _)| *id != conn_id);
    }
}

// ── VFO command parser ─────────────────────────────────────────────────────

fn parse_vfo_cmd(state: &TciState, rest: &str) {
    // Format: "trx,channel,freq_hz"
    let mut parts = rest.splitn(3, ',');
    let trx_str = parts.next().unwrap_or("").trim();
    let _chan_str = parts.next().unwrap_or("").trim();
    let freq_str = parts.next().unwrap_or("").trim();

    let trx = match trx_str.parse::<u32>() {
        Ok(v) => v,
        Err(_) => return,
    };
    if trx as usize >= NUM_SINKS {
        tracing::warn!("[TCI] VFO: trx={trx} out of range");
        return;
    }
    let freq_raw = match freq_str.parse::<u64>() {
        Ok(v) => v,
        Err(_) => return,
    };
    let freq_hz = freq_raw.clamp(1_000_000, 69_952_000);
    if freq_raw != freq_hz {
        tracing::warn!("[TCI] VFO:{trx} freq {freq_raw} Hz clamped to {freq_hz}");
    }
    state.vfo_freqs[trx as usize].store(freq_hz, Ordering::Release);
    state.vfo_queue.lock().unwrap().push_back((trx, freq_hz));
    println!("[TCI] VFO trx={trx} rx_freq={freq_hz} Hz");
}

// ── Init sequence ──────────────────────────────────────────────────────────

fn build_init_sequence(state: &TciState) -> String {
    let mut s = String::with_capacity(512);
    // All commands must be lowercase — WSJT-X's command map uses lowercase keys.
    s.push_str("vfo_limits:1000000,70000000;");
    s.push_str("if_limits:-24000,24000;");
    s.push_str("trx_count:4;");
    s.push_str("channels_count:1;");
    s.push_str("device:GranolaSdr;");
    s.push_str("receive_only:true;");
    s.push_str("modulations_list:usb,lsb,cw,am,fm;");
    s.push_str("protocol:GranolaSdr,2.0;");
    for trx in 0..NUM_SINKS {
        let freq = state.vfo_freqs[trx].load(Ordering::Acquire);
        s.push_str(&format!("vfo:{trx},0,{freq};"));
    }
    for trx in 0..NUM_SINKS {
        s.push_str(&format!("modulation:{trx},usb;"));
    }
    for trx in 0..NUM_SINKS {
        s.push_str(&format!("rx_enable:{trx},true;"));
    }
    // start; sets _power_=true in WSJT-X TCITransceiver::do_start() during its
    // 2s mysleep6 wait.  Do NOT send ready; here — Cmd_Ready calls tci_done7()
    // which poisons tci_loop7_ so the first mysleep7() in do_frequency() returns
    // immediately before the VFO echo arrives, breaking cross-band tuning.
    s.push_str("start;");
    s
}

// ── Audio frame builder ────────────────────────────────────────────────────

/// Build a TCI binary audio frame: 64-byte header + float32 stereo-interleaved payload.
fn build_audio_frame(trx: u32, samples: &[f32], gain: f32) -> Vec<u8> {
    let n = samples.len();
    let float_count = n * 2; // stereo-interleaved L,R pairs
    let mut buf = Vec::with_capacity(64 + float_count * 4);

    // 64-byte header: 16 × u32 LE, matching WSJT-X Data_Stream layout exactly.
    let header: [u32; 16] = [
        trx,                // [0]  receiver
        48_000,             // [1]  sampleRate
        3,                  // [2]  format: 3 = float32
        0,                  // [3]  codec
        0,                  // [4]  crc
        float_count as u32, // [5]  length: float count (2 × sample count)
        1,                  // [6]  type: 1 = RX_AUDIO_STREAM
        0, 0, 0, 0, 0, 0, 0, 0, 0, // [7–15] reserv[9]
    ];
    for word in &header {
        buf.extend_from_slice(&word.to_le_bytes());
    }

    // Float32 stereo-interleaved payload.  WSJT-X store() reads source[i*2]
    // (left channel only), so mono audio duplicated as L=R gives full fidelity.
    for &s in samples {
        let v = s * gain;
        buf.extend_from_slice(&v.to_le_bytes()); // L
        buf.extend_from_slice(&v.to_le_bytes()); // R
    }

    buf
}

// ── Broadcast helpers ──────────────────────────────────────────────────────

fn broadcast_audio(state: &TciState, trx: usize, frame: Vec<u8>) {
    let mut subs = state.audio_subs[trx].lock().unwrap();
    subs.retain(|(_, tx)| match tx.try_send(WsMsg::Binary(frame.clone())) {
        Ok(()) => true,
        Err(TrySendError::Full(_)) => {
            tracing::debug!("[TCI] trx={trx} audio frame dropped: subscriber slow");
            true
        }
        Err(TrySendError::Closed(_)) => false,
    });
}

fn broadcast_text(state: &TciState, text: String) {
    let mut subs = state.all_subs.lock().unwrap();
    subs.retain(|(_, tx)| match tx.try_send(WsMsg::Text(text.clone())) {
        Ok(()) => true,
        Err(TrySendError::Full(_)) => true,
        Err(TrySendError::Closed(_)) => false,
    });
}

// ── FFI-callable functions (called from lib.rs C exports) ─────────────────

/// Append `pcm` to the per-sink accumulator; broadcast a frame when 480 samples accumulate.
/// Called from C++ audioWorker for each 240-sample block.
pub fn push_audio(trx: u32, pcm: &[f32]) {
    let Some(state) = TCI_STATE.get() else { return };
    push_audio_impl(state, trx, pcm);
}

fn push_audio_impl(state: &TciState, trx: u32, pcm: &[f32]) {
    let trx_idx = trx as usize;
    if trx_idx >= NUM_SINKS {
        tracing::warn!("[TCI] push_audio: trx={trx} out of range");
        return;
    }
    state.audio_push_count.fetch_add(1, Ordering::Relaxed);
    let flushed = {
        let mut accum = state.audio_accum[trx_idx].lock().unwrap();
        accum.extend_from_slice(pcm);
        if accum.len() >= AUDIO_FRAME {
            let samples: Vec<f32> = accum.drain(..AUDIO_FRAME).collect();
            Some(samples)
        } else {
            None
        }
    };
    if let Some(samples) = flushed {
        let frame = build_audio_frame(trx, &samples, state.gain);
        broadcast_audio(state, trx_idx, frame);
    }
}

/// Broadcast S-meter level to all connected clients.
/// Called from C++ audioWorker every 100 frames (~1 Hz).
/// Source: norm_ema_h_[sink_bins[trx]] (RF signal level, not audio RMS).
/// WSJT-X Cmd_Smeter: level_ = args.at(2).toInt() + 73; needs integer dBFS.
pub fn push_smeter(trx: u32, dbfs: f32) {
    let Some(state) = TCI_STATE.get() else { return };
    if trx as usize >= NUM_SINKS { return; }
    let level = dbfs as i32;
    let text = format!("rx_smeter:{trx},0,{level};");
    broadcast_text(state, text);
}

/// Drain one VFO command from the queue.
/// Returns `Some((trx, freq_hz))` if pending, `None` if empty.
/// Called from C++ tciVfoWorker; drains to empty per tick.
pub fn poll_vfo() -> Option<(u32, u64)> {
    TCI_STATE.get()?.vfo_queue.lock().unwrap().pop_front()
}

// ── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state() -> Arc<TciState> {
        TciState::new(1.0)
    }

    #[test]
    fn audio_accumulator_flushes_at_480() {
        let state = make_state();
        let block1 = vec![0.5f32; AUDIO_VALID];
        let block2 = vec![-0.5f32; AUDIO_VALID];

        push_audio_impl(&state, 0, &block1);
        assert_eq!(state.audio_accum[0].lock().unwrap().len(), AUDIO_VALID);

        push_audio_impl(&state, 0, &block2);
        assert_eq!(
            state.audio_accum[0].lock().unwrap().len(),
            0,
            "accumulator should drain after flush"
        );
    }

    #[test]
    fn audio_accumulator_no_partial_flush() {
        let state = make_state();
        let partial = vec![0.0f32; 100];
        push_audio_impl(&state, 0, &partial);
        assert_eq!(state.audio_accum[0].lock().unwrap().len(), 100);
    }

    #[test]
    fn audio_frame_header_layout() {
        let frame = build_audio_frame(2, &vec![0.5f32; AUDIO_FRAME], 1.0);
        // 64-byte header + AUDIO_FRAME stereo float32 pairs
        assert_eq!(frame.len(), 64 + AUDIO_FRAME * 2 * 4, "total frame size");

        let header: Vec<u32> = (0..16)
            .map(|i| u32::from_le_bytes(frame[i * 4..(i + 1) * 4].try_into().unwrap()))
            .collect();

        assert_eq!(header[0], 2, "receiver = trx");
        assert_eq!(header[1], 48_000, "sample_rate");
        assert_eq!(header[2], 3, "format: float32");
        assert_eq!(header[3], 0, "codec");
        assert_eq!(header[5], (AUDIO_FRAME * 2) as u32, "length = stereo float count");
        assert_eq!(header[6], 1, "type: RX_AUDIO_STREAM");
        assert_eq!(header[7], 0, "reserv[0]");
    }

    #[test]
    fn audio_frame_payload_float32_stereo() {
        // Each mono sample emitted as an L,R float32 pair.
        let frame = build_audio_frame(0, &[0.5f32], 2.0);
        let l = f32::from_le_bytes(frame[64..68].try_into().unwrap());
        let r = f32::from_le_bytes(frame[68..72].try_into().unwrap());
        assert!((l - 1.0f32).abs() < 1e-6, "L = sample * gain = 0.5 * 2.0");
        assert_eq!(l, r, "stereo: L == R for mono source");
    }

    #[test]
    fn vfo_parser_happy_path() {
        let state = make_state();
        parse_vfo_cmd(&state, "0,0,14074000");
        let entry = state.vfo_queue.lock().unwrap().pop_front();
        assert_eq!(entry, Some((0, 14_074_000)));
        assert_eq!(state.vfo_freqs[0].load(Ordering::Relaxed), 14_074_000);
    }

    #[test]
    fn vfo_parser_malformed_too_few_parts() {
        let state = make_state();
        parse_vfo_cmd(&state, "not,valid");
        assert!(state.vfo_queue.lock().unwrap().is_empty());
    }

    #[test]
    fn vfo_parser_malformed_freq() {
        let state = make_state();
        parse_vfo_cmd(&state, "0,0,notanumber");
        assert!(state.vfo_queue.lock().unwrap().is_empty());
    }

    #[test]
    fn vfo_parser_oob_trx() {
        let state = make_state();
        parse_vfo_cmd(&state, "99,0,14074000");
        assert!(state.vfo_queue.lock().unwrap().is_empty());
    }

    #[test]
    fn vfo_parser_freq_clamped_high() {
        let state = make_state();
        parse_vfo_cmd(&state, "0,0,999999999");
        let (_, freq) = state.vfo_queue.lock().unwrap().pop_front().unwrap();
        assert_eq!(freq, 69_952_000, "should clamp to IF_LIMITS max");
    }

    #[test]
    fn vfo_parser_freq_clamped_low() {
        let state = make_state();
        parse_vfo_cmd(&state, "0,0,1");
        let (_, freq) = state.vfo_queue.lock().unwrap().pop_front().unwrap();
        assert_eq!(freq, 1_000_000, "should clamp to VFO_LIMITS min");
    }

    #[test]
    fn audio_start_idempotent() {
        let state = make_state();
        let (tx, _rx) = mpsc::channel::<WsMsg>(8);
        subscribe_audio(&state, 42, &tx, 0);
        subscribe_audio(&state, 42, &tx, 0); // duplicate
        let count = state.audio_subs[0].lock().unwrap().len();
        assert_eq!(count, 1, "idempotent: only one subscriber despite two calls");
    }

    #[test]
    fn audio_start_oob_trx() {
        let state = make_state();
        let (tx, _rx) = mpsc::channel::<WsMsg>(8);
        subscribe_audio(&state, 42, &tx, 99); // out of range
        // Should not panic; no subscriber added anywhere
        for subs in &state.audio_subs {
            assert!(subs.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn init_sequence_structure() {
        let state = make_state();
        let init = build_init_sequence(&state);
        // All commands must be lowercase for WSJT-X's case-sensitive map lookup.
        assert!(init.contains("trx_count:4;"), "trx count");
        assert!(init.contains("receive_only:true;"), "receive only");
        assert!(init.contains("vfo:0,0,14074000;"), "default 20m freq");
        assert!(init.contains("vfo:1,0,7074000;"), "default 40m freq");
        assert!(init.contains("rx_enable:0,true;"), "rx_enable trx 0");
        assert!(init.contains("rx_enable:3,true;"), "rx_enable trx 3");
        assert!(init.contains("start;"), "start; sets _power_=true in WSJT-X");
        assert!(init.ends_with("start;"), "start; must be last — ready; poisons tci_loop7_");
        assert!(!init.contains("ready;"), "ready; must NOT be sent: Cmd_Ready pre-quits loop7");
    }

    #[test]
    fn poll_vfo_returns_none_when_empty() {
        let state = make_state();
        assert!(state.vfo_queue.lock().unwrap().pop_front().is_none());
    }

    #[test]
    fn push_audio_oob_trx_no_panic() {
        let state = make_state();
        let pcm = vec![0.5f32; AUDIO_VALID];
        push_audio_impl(&state, 99, &pcm); // should not panic
    }
}
