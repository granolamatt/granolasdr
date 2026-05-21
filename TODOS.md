# TODOS

Deferred items from /plan-ceo-review (2026-05-21). None are blocked — each waits on a natural predecessor.

## Live browser monitoring dashboard

SSE endpoint in `controlWorker` (HFChannelizer.cc) + minimal HTML page served from the same port.
- `GET /events` → chunked SSE stream: one JSON line per decode, plus periodic blocks/s heartbeat
- Single static HTML file served from `GET /` with a plain JS EventSource listener and a scrolling decode table
- ~80 lines C++, ~60 lines HTML/JS, zero new dependencies (cpp-httplib already supports chunked streaming)
- **Prerequisite**: none. Can be built any time on a stable branch.

## Wideband waterfall display

WebSocket stream of decimated FFT magnitude from HFChannelizer → canvas renderer in browser.
- Decimate 140k bins → ~2k bins per block (log-scale or band-slice); 50 kB/s at 6.25 blocks/s
- GPU decimation kernel or simple CPU strided sum; WebSocket server (cpp-httplib or uWebSockets)
- Canvas-based waterfall renderer in JS (~150 lines)
- **Prerequisite**: dashboard SSE infra first — proves the HTTP server can handle streaming before adding WebSocket.

## GPU LDPC post-ship items (gpu-ldpc branch)

From /plan-eng-review (2026-05-21), deferred until after the branch is merged:

- **QP-ADMM vs BP convergence baseline**: measure decode counts per epoch on D8 corpus WAV files
  with both decoders at equal SNR. Target: QP-ADMM ≥ BP at FT8_GPU_CAND_MAX=500.
  Gate: if QP-ADMM loses >2% decodes vs BP on real traffic, investigate rho/max_iter tuning first.

- **FT8_GPU_CAND_MAX reduction**: current value (500000) pre-dates the GPU scan path; typical
  epoch returns ~73 candidates. Reduce to 2048 to cut log174 device/pinned alloc by 99%.
  Prerequisite: one week of production data with --gpu-ldpc to confirm count distribution.

- **Cascade timeout detection**: count consecutive ldpc_done 500ms timeouts in the snapshot thread;
  if >2 consecutive, print a loud warning (GPU may be overloaded). Reset counter on success.

## FT4 decode pipeline

Second decode consumer alongside FT8, using the same Costas scan / LDPC family.
- 7.5-second epoch (half of FT8); separate GPU scan parameters and epoch trigger
- Reuse most of FT8Cuda machinery; add second ZMQ PUB on a different port; PSKReporter accepts FT4
- L effort: ~400 lines across FT8Cuda.cc, a new FT4.cc, CMakeLists
- **Prerequisite**: FT8 pipeline fully stable (oversample branch merged, no slip issues in production).
