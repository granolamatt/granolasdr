# TODOS

Updated 2026-07-10.

## TCI Server (WSJT-X audio interface) — ✅ COMPLETE (active 2026-06-27)

Shipped and active. WSJT-X audio interface live. Original plan retained below for reference.
(IQ stream item TC-IQ remains deferred — see P3.)

**P1 — core:**
- **TC1** `wsserver/src/tci.rs` (new): TCI server module — init sequence with RX_ENABLE (×4), AUDIO_START/STOP subscription, audio accumulator (480 samples/10ms per sink), INT16 conversion (u32::to_le_bytes serialization), auto-start 1s after READY, VFO command queue draining-to-empty per tick, RX_CHANNEL_SENSORS at 1 Hz (dBFS from norm_ema_h_ via tci_push_smeter). Runs in existing tokio runtime via OnceLock<Handle>. **VERIFY Stream struct binary format against ExpertSDR3/TCI repo spec before implementing serialization.**
- **TC-UNIT** `wsserver/src/tci.rs` (`#[cfg(test)]` block): Unit tests — accumulator flush at 480 samples, INT16 saturation (gain=10), VFO parser happy+malformed+OOB, AUDIO_START idempotency, dBFS calculation. Run with `cargo test`.
- **TC2** `wsserver/src/lib.rs`: OnceLock<tokio::runtime::Handle> set via `rt.handle().clone()` BEFORE `rt.block_on()`. C FFI exports: `tci_server_start`, `tci_server_stop`, `tci_push_audio(trx, pcm, count)`, `tci_push_smeter(trx, dbfs)`, `tci_poll_vfo(trx*, freq*)`.
- **TC3** `tci_server.h` (new): C header for all 5 FFI declarations.
- **TC4** `gm/cuda/HFChannelizer.cc/.h`: audioWorker calls `tci_push_audio(sink, pcm, AUDIO_VALID)` after ZMQ send, and `tci_push_smeter(sink, dbfs)` every 100 frames using `norm_ema_h_[sink_bins[sink]]`. Add `tciVfoWorker()` as joinable thread (stored as `tci_vfo_thread_` in header, joined in `~HFChannelizer()`). Checks `isRunning()`. Drains `tci_poll_vfo()` to empty per tick. Does NOT update `sink_labels[]`.
- **TC5** `gm/HFRx.cc`: Add `--tci-port=40001` (0=disabled) and `--tci-gain=1.0` CLI args; call `tci_server_start(tci_port)` at startup. **Shutdown order: `channelizer.stop()` → `channelizer.join()` → `tci_server_stop()`** (prevents tciVfoWorker from polling destroyed Rust global).
- **TC6** `gm/HFRx.cc` (cleanup): Add `channelizer.join()` call to ensure tci_vfo_thread_ exits before tci_server_stop.
- **TC7** `docker-compose.yml`: Add `- "40001:40001"` to ports.

**Error handling (required, P1):**
- `tci_push_audio()`: bounds-check `trx < NUM_SINKS`; drop + log warn if out of range.
- VFO handler: clamp incoming freq to `[1_000_000, 69_952_000]` Hz before `sink_bins` write.
- Duplicate `AUDIO_START`: idempotent subscribe (no double-push).

**Key invariants:**
- TCI uses existing tokio runtime via OnceLock<Handle> cloned BEFORE block_on.
- Audio accumulator is per-sink (global), not per-client. All clients get same 10ms frames.
- `tci_push_audio()` never blocks audioWorker (try_send drop on full).
- S-meter uses `norm_ema_h_[sink_bins[sink]]` → dBFS (RF signal level, not audio RMS).
- S-meter rate: 1 Hz (every 100 audio frames).
- `--tci-gain` tunes INT16 scale; calibrate at first WSJT-X connection.
- tciVfoWorker does NOT update sink_labels[] (avoids data race on non-atomic std::string).
- Shutdown order: channelizer.stop() → join() → tci_server_stop().

**P2 — polish:**
- **TC8** `test/test_tci_client.py` (new): Python WebSocket client — connect, send VFO:0,0,14074000; + AUDIO_START:0;, verify binary Stream frame headers (receiver=0, type=1, sample_rate=48000, channels=1). Follows debug_js8_fast.py pattern.
- **TC9** `ARCHITECTURE.md`: Add TCI server block to pipeline diagram.

**P3 — deferred:**
- **TC-IQ** IQ stream (IQ_START/IQ_STOP): streams channelizer composite at 409.6 kHz as TCI IQ frames. Enables RBN Skimmer, SDR Console, broad-band WSJT-X scan. Implement after audio path is validated on-air.
  Context: TCI IQ frame uses StreamType::IQ_STREAM=0, same Stream struct. Would need `tci_push_iq(const complex<float>* buf, count)` FFI and a subscriber path from HFChannelizer output.
  Effort: M (human ~1 day / CC ~30 min). No deps on audio path.

## Phase 15: Turbo + Ultra JS8 modes — ✅ COMPLETE (shipped 2026-06-27)

Turbo and Ultra modes shipped. granolasdr now decodes all five JS8 speed modes
simultaneously. Original plan retained below for reference.

**P1 — bug fixes (do first):**
- **T1** `gm/HFRx.cc`: Fix cap_blocks — kNormalCapBlks 106→108, kFastCapBlks 100→108, kSlowCapBlks 94→108
- **T7** `gm/HFRx.cc`: Fix runProxy() hardcoded ports 5599/5600 → use kProxyXSubPort/kProxyXPubPort
- **ER1** `debug_js8_fast.py:9`: Fix mode filter 'JS8-FAST' → 'JS8 Fast' (bug — never matched)

**P1 — new modes:**
- **T2** `gm/HFRx.cc`: Add Turbo constants (rfft=20480, time_osr=2, freq_osr=2, cap=108, sym_per=0.050, cycle=6) + Ultra (rfft=13107, time_osr=2, freq_osr=2, cap=108, sym_per≈0.032, cycle=4)
- **T3** `gm/HFRx.cc`: Migrate Fast+Slow from MagBlock\<100\>/JS8Cuda\<100\> → \<128\>
- **T4** `gm/HFRx.cc`: Add Turbo+Ultra pipeline blocks, flags, CLI args `--js8-turbo`, `--js8-ultra`; mode_name_="JS8 Turbo"/"JS8 Ultra"; update both call sites (lines 248, 259)
- **T5** `gm/cuda/MagBlock.h/.cc`: Add extern template \<128\>, remove \<100\>
- **T6** `gm/cuda/JS8Cuda.h/.cc`: Add extern template \<128\>, remove \<100\>
- **T9** `test/test_js8_fast_gpu_scan.cc`: Recreate GPU scan test for N=128 + MODIFIED Costas {0,6,2,3,5,4,1}/{1,5,0,2,3,6,4}/{2,5,0,6,4,1,3}

**P2 — polish:**
- **T8** `gm/cuda/MagBlock.h:19-20`: Fix stale comment Normal rfft 32768→65536, Fast 20480→40960; Fast ring 100→128
- **T10** `debug_js8_turbo.py`, `debug_js8_ultra.py`: Create from debug_js8_fast.py pattern; mode filter = 'JS8 Turbo'/'JS8 Ultra' (NOT 'JS8-40'/'JS8-60')
- **T11** `docker-compose.yml:5`: Add --js8-turbo --js8-ultra to command
- **T12** `ARCHITECTURE.md`: MagBlock\<100\>→\<128\>, cap_blocks table 100/94→108, add Turbo+Ultra rows
- **T13** `docs/pipeline.html:445`: ring sizes 100→128, 94→128, add Turbo+Ultra ring rows

**Key invariants:**
- cap_blocks ≥ 108 for ALL JS8 modes (formula: max(block_abs) = 29+72+6 = 107)
- N=128 for all non-Normal rings (≥ cap_blocks + 20 spare slots)
- mode_name_ strings: "JS8", "JS8 Fast", "JS8 Slow", "JS8 Turbo", "JS8 Ultra"
- All non-Normal modes use js8_fast_gpu_scan + legacy_costas flag (matches HFRx.cc pattern)
- Ultra rfft=13107 forces cuFFT Bluestein — accepted tradeoff (D10)



## Remaining Phase 12 work — ✅ COMPLETE

Phase 11 and Phase 12 complete, including the previously-open items (wideband
waterfall resolution, CUDA error checking on FT8Cuda/JS8Cuda launches, corpus
re-enable).

## Docker deployment — ✅ COMPLETE

Docker smoke test and layer caching complete.

## LDPC decode → moved to ../qp-admm

LDPC decode (QP-ADMM, BP, OSD fallback) has moved out of granolasdr to the
`../qp-admm` repo. granolasdr's role is now **LLR capture only**: it produces
labeled LLRs (pass/osd/fail) and hands them off. Decoder convergence work,
the QP-ADMM vs BP baseline, and any decoder tuning now live in `../qp-admm`.

## CW Skimmer — DEFERRED (disabled by default 2026-07-10, opt-in via --cw)

Phase-1 only and off by default (`enable_cw=false`, HFRx.cc). Two blocking problems
keep it there. Do NOT re-enable by default until CW1 and CW2 are fixed.

**CW1 — Crash on CW-like input — ✅ FIXED (3924a15).**
Root cause was a concrete OOB write: an all-zero/degenerate envelope → cw_morse
AGC 0/0=NaN → otsu `hist[(int)NaN]` out of bounds (SIGSEGV). Fixed in depth
(NaN-safe otsu + clamped index, all-silence bail in cw_morse decode, all-zero
skip in CWSkimmerCuda decodeActive, strtof/clamped CW_SNR/CW_DRIFT/CW_CONFIRM).
test_cw_morse now has a crash-safety block (negative control segfaulted pre-fix).

**CW2 — Detection quality — IN PROGRESS (adaptive gate + validation shipped).**
Metric is ALREADY percentile (p80-p50), not peak-mean (that's long fixed). Built
`cw_offline` (test/cw_offline.cc): CPU MagBlock front end + real CwTracker/CwMorse,
replays any window of a --record-cw capture for tuning without CUDA/radio. Against
cwtest.dat (11h overnight capture):
- Adaptive gate (3132221): gate = max(floor, k×median-of-metric), default k=2.7
  floor=6, on by default. Median is the stable noise floor so the gate floats.
  Fixed 12→49 unique / sweet-spot ~10→63 / default→53 evening. CW_ADAPT_K/CW_SNR tune.
- Validation quality (3fe5d3f): strip CQ/QRZ/TEST/DX prefix (CQK4RO→K4RO), and
  relative-dominance suppression of sparse QSB misreads (one carrier's dominant
  call wins; real alternating QSO still spots both). 74→64 unique.
REMAINING: decode-quality misreads that validation can't touch (odd-prefix D3WU/
R5HMT, merged-call M9MR≈W9MR on adjacent bins, 1×1 partials). Needs cw_morse timing
work or CW3. Also: still verify all this on-air / wire ZMQ+RBN publish (Phase 3).

**CW3 — ML CW reader (real fix; deferred behind the de-chirp track).**
CNN/RNN over the CW composite spectrogram: detect carriers as dashed-line objects,
read dot/dash timing, emit callsign. Train on `cwtest*.dat` + confirmed CW decodes
via the `--record-cw`/`--playback-cw` harness; crash inputs become labeled
robustness cases. This is one class of the larger "waterfall track detector" vision
(see the FT8/JS8 de-chirp exploration — same YOLO/CNN detector, CW is one label).

**Sequencing:** CW1 (crash) anytime — it's a bug. CW2/CW3 wait until the de-chirp
track (active) proves the free-label training loop on FT8/JS8.
