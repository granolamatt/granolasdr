# JS8 Normal Decode — Phase 10 CEO Plan

**Status:** ✅ COMPLETE (shipped 2026-06-27)  
**Mode:** SELECTIVE EXPANSION  
**Authored:** 2026-05-27  
**Review:** /plan-ceo-review

> **Done.** JS8 Normal decode shipped via the ring-sharing architecture as planned.
> Note: LDPC decode has since moved out of granolasdr to `../qp-admm`; granolasdr now
> captures LLRs only and hands them off. The plan below is retained as a historical record.

---

## 1. Problem Statement / Vision

granolasdr decodes FT8 on every HF band simultaneously using a GPU pipeline.
JS8 Normal (Mode A) uses the same 79-symbol, 6.25 baud, 8-FSK frame as FT8, differing only in
three constants: Costas sync pattern, LDPC parity matrix, and message format. The hardware
already hears every JS8 frequency; we are discarding the signals at decode time.

Goal: add JS8 Normal decode as a second consumer of the existing mag ring buffer, producing
decoded JS8 conversations on a ZMQ bus, visible in the dashboard, and spotted on PSKReporter.
No new hardware. No new SDR sample pipeline. Continuous scan only — epoch-path not needed
because JS8 epoch timing is advisory and can drift.

---

## 2. What We Are Building

### In scope

| Item | Description |
|------|-------------|
| JS8ScanCuda | GPU Costas scan with `{4,2,5,6,1,3,0}` array |
| JS8LdpcCuda | GPU BP decoder with JS8's M=87 H-matrix |
| JS8Cuda | Lightweight scan orchestrator; reads FT8Cuda's shared ring |
| gm/hf/js8 | CPU decode: CRC-12 (poly 0xC06, XOR 42), 12×6-bit message, ZMQ PUB 5581 |
| FT8Cuda delta | Public ring accessor + make `ring_write_idx` atomic |
| HFRx.cc delta | Wire JS8Cuda + JS8 decoder with callbacks |
| Dashboard | JS8 conversation log panel (cherry-pick) |
| PSKReporter | JS8 spots via psk_uploader.py (cherry-pick) |

### Out of scope

- JS8 Fast / Slow / Turbo / Ultra submodes (different baud rates; Phase 11 if needed)
- TX / transmit path (RX-only hardware)
- JS8 Varicode/protocol layer — we emit the raw 12-char message + type bits; application
  semantics (heartbeat, APRS relay, etc.) are not decoded
- JS8 soft combining across cycles (JS8Call does this; adds significant state; defer)
- Epoch scan path for JS8 (continuous scan handles arbitrary epoch offsets; epoch path unused)

---

## 3. Success Criteria

1. **Decode parity**: JS8 Normal decodes on 40m/20m with candidate counts comparable to running
   JS8Call on the same band. Not measuring SNR floor; measuring "do we hear traffic."
2. **FT8 zero regression**: FT8 decode rate and timing are unchanged. Verified by running both
   protocols simultaneously during a 15-minute on-air window.
3. **Dashboard**: JS8 conversation panel shows callsign + raw 12-char message + frequency + SNR
   updating in real time.
4. **PSKReporter**: JS8 spots appear on pskreporter.info within 10 minutes of a decode.
5. **Ring atomicity**: no data races (TSAN clean, or at minimum: `ring_write_idx` made atomic).

---

## 4. Architecture

### Ring buffer sharing (key decision)

FT8Cuda owns the mag ring (`magFT8_ring_d` — device memory, 200 slots × block_bytes).
JS8Cuda does NOT allocate its own ring or RFFT; it reads from FT8Cuda's ring via two new
public accessors:

```cpp
// FT8Cuda.h additions:
const uint8_t* getRingPtr() const { return magFT8_ring_d; }
uint64_t       getRingWriteIdx() const { return ring_write_idx.load(std::memory_order_acquire); }
int            getRingBlocks() const { return RING_BLOCKS; }
```

`ring_write_idx` must be changed from `uint64_t` to `std::atomic<uint64_t>`.
This is both a new-feature requirement AND a latent bug fix: the existing contWorker thread
already reads `ring_write_idx` from FT8Cuda's main run thread without synchronization.

JS8Cuda has its own `cont_scan_stream` CUDA stream and polls `getRingWriteIdx()` at stride=6
(same cadence as FT8's continuous scan, ~1 scan/sec). Because JS8 transmissions can start at
any offset within the 15s window, continuous scan is the correct and sufficient path.

### JS8Cuda as a lighter class

JS8Cuda constructor takes `FT8Cuda*` for ring access. It does NOT replicate:
- RFFT or block ingestion (ring is filled by FT8Cuda)
- Waterfall path
- Epoch trigger logic
- Audio corpus capture

JS8Cuda only owns:
- Its own scan buffers (candidate fo/to/ts/fs/score, log174, x_hat, parity) on device
- Its own `cont_scan_stream` and `cont_ring_ready` event
- Its own `contWorker` thread
- A `decode_callback` wired to the JS8 CPU decoder

### Protocol flow

```
HFChannelizer → FT8Cuda (ring write)
                    │
                    ├─ FT8 cont scan → FT8 decode → ZMQ 5580 → psk_uploader
                    │
                    └─ JS8Cuda reads same ring
                           └─ JS8 cont scan → JS8 decode → ZMQ 5581 → psk_uploader
```

### Dashboard

HFChannelizer SSE already supports `broadcastDecode()`. We add `broadcastJS8()` — same
signature, different event type `"js8"`. The dashboard panel listens for `"js8"` events and
appends to a scrolling conversation log (callsign + message text + freq + SNR + time).

### PSKReporter

`psk_uploader.py` hardcodes `_str("FT8")` on line 98. Change to `_str(r.get("mode", "FT8"))`.
JS8 decoder's ZMQ JSON adds `"mode":"JS8"`. psk_uploader subscribes to both ports 5580 and 5581
using two ZMQ SUB sockets polled with `zmq.Poller`. JS8 spots tagged with mode "JS8" — PSKReporter
accepts this as a valid mode identifier.

---

## 5. Implementation Phases

### Phase 1 — Core JS8 GPU decode pipeline (~1,800 lines)

Files created:

| File | Lines | Notes |
|------|-------|-------|
| `gm/cuda/JS8ScanCuda.cu` | ~165 | Clone of FT8ScanCuda.cu; `kCostas[7] = {4,2,5,6,1,3,0}` |
| `gm/cuda/JS8ScanCuda.h` | ~30 | Clone of FT8ScanCuda.h; rename symbols JS8_* |
| `gm/cuda/JS8LdpcCuda.cu` | ~680 | Same BP kernel structure; new M=87 H-matrix from JS8Call Nm[] |
| `gm/cuda/JS8LdpcCuda.h` | ~45 | `JS8_LDPC_BATCH = JS8_GPU_CAND_MAX` |
| `gm/cuda/JS8Cuda.h` | ~120 | Lighter than FT8Cuda; takes FT8Cuda* for ring access |
| `gm/cuda/JS8Cuda.cc` | ~380 | Cont scan loop + LDPC + decode callback dispatch |
| `gm/hf/js8.h` | ~65 | Clone of ft8.h; ZMQ port 5581 |
| `gm/hf/js8.cc` | ~315 | CRC-12, 12×6-bit message decode, ZMQ publish, broadcastCallback |

Files modified:

| File | Delta | Notes |
|------|-------|-------|
| `gm/cuda/FT8Cuda.h` | +15 | `ring_write_idx` → `std::atomic<uint64_t>`; 3 public accessors |
| `gm/cuda/FT8Cuda.cc` | +5 | `.store()` / `.load()` on ring_write_idx; 0 behavior change |
| `gm/HFRx.cc` | +28 | Construct JS8Cuda(ft8channel), JS8, wire callbacks; 2nd ZMQ port |
| `CMakeLists.txt` | +12 | Add JS8ScanCuda, JS8LdpcCuda, JS8Cuda, js8 to target |

### Phase 2 — Dashboard JS8 conversation panel (~160 lines, cherry-pick)

| File | Delta | Notes |
|------|-------|-------|
| `gm/cuda/HFChannelizer.h` | +8 | `broadcastJS8()` declaration |
| `gm/cuda/HFChannelizer.cc` | +30 | SSE event type `"js8"`, same queue path as FT8 |
| `gm/HFRx.cc` | +8 | Wire js8.setBroadcastCallback → epochbuffer.broadcastJS8() |
| Dashboard HTML/JS | +110 | JS8 panel: EventSource "js8" events, scrolling table |

### Phase 3 — PSKReporter JS8 spots (~70 lines, cherry-pick)

| File | Delta | Notes |
|------|-------|-------|
| `psk_uploader.py` | +35 | `_str(r.get("mode","FT8"))` on line 98; add port 5581 subscriber; zmq.Poller |
| `gm/hf/ft8.cc` | +5 | Add `"mode":"FT8"` to ZMQ JSON for consistency |
| `gm/hf/js8.cc` | — | Already includes `"mode":"JS8"` from Phase 1 |

---

## 6. JS8 Technical Constants

### Costas array (GPU scan kernel)
```c
// JS8ScanCuda.cu — the only change from FT8ScanCuda.cu
static __constant__ uint8_t kCostas[7] = {4, 2, 5, 6, 1, 3, 0};  // JS8 ORIGINAL
// FT8 uses:                              {3, 1, 4, 0, 6, 5, 2}
```

### LDPC parameters
```
N = 174   (codeword bits — same as FT8)
K = 87    (message bits — FT8 uses 91)
M = 87    (check nodes — FT8 uses 83)
Rate = 0.5

Source: JS8Call JS8.cpp — Nm[87] check node list
Transcription target: k_check_var_j[87][D_JS8_MAX]
D_JS8_MAX: each variable node participates in 3 checks (BP_MAX_CHECKS=3);
  total connections = 174×3 = 522; avg per check node = 6.0;
  D_JS8_MAX expected = 6 or 7 (verify from Nm table)
```

### CRC
```
CRC-12, polynomial 0xC06, initial value 0, XOR key 42
Input: decoded[0..74] (75 message bits, zero-padded to 12 bytes)
Check: computed CRC == decoded[75..86]
Source: JS8Call JS8.cpp — crc12() function
```

### Message decode
```cpp
const char* alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+";
// decoded[0..71]: 12 × 6-bit words → 12 chars
// decoded[72..74]: 3-bit type field
```

### Operating frequencies (all within existing HFChannelizer bands)
```
40m  7.078 MHz    20m  14.078 MHz   80m  3.578 MHz
30m  10.130 MHz   17m  18.104 MHz   15m  21.078 MHz
```

---

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| H-matrix transcription error | Medium | Unit-test CRC-12 against a known good JS8 frame; compare BP output to JS8Call reference decoder on same LLRs |
| `ring_write_idx` atomic change breaks FT8 | Low | Change is load/store — no behavior difference; CI build + on-air test before merging |
| JS8 candidates exceed GPU_CAND_MAX | Low | JS8 traffic density is lower than FT8; use same JS8_GPU_CAND_MAX=2048 initially |
| JS8Cuda scan adds latency to FT8 path | Negligible | JS8Cuda runs on its own CUDA stream; no synchronization with FT8 stream |
| PSKReporter rejects "JS8" mode tag | Low | PSKReporter mode field is free-form string; "JS8" is widely used by JS8Call submitters |
| Costas doc table inconsistency | Verified | docs/js8_decode_algorithm.md comparison table shows FT8 Costas as [4,2,5,6,1,3,0] — this is WRONG. Actual FT8ScanCuda.cu kCostas = {3,1,4,0,6,5,2}. JS8 Normal ORIGINAL = {4,2,5,6,1,3,0}. Arrays ARE different. |

---

## 8. Dependencies

- **JS8Call source** (`~/workarea/js8/`): already present. Source of H-matrix (Nm[87]),
  CRC-12 function, Costas ORIGINAL array. Read-only reference.
- **ft8_lib**: NOT used for JS8 decode (different message format). JS8 decode is self-contained.
- **HFChannelizer ring**: no new dependency; JS8Cuda reads from FT8Cuda's existing ring.
- **zmq**: already linked; second socket on 5581 trivial.
- Phase 2 and Phase 3 are independent of each other; both require Phase 1.

---

## 9. Effort Estimate

| Phase | New files | Modified files | Lines | Calendar |
|-------|-----------|----------------|-------|----------|
| Phase 1 (core) | 8 | 4 | ~1,800 | 1 session |
| Phase 2 (dashboard) | 0 | 4 | ~160 | 0.5 session |
| Phase 3 (PSKReporter) | 0 | 3 | ~70 | 0.25 session |
| **Total** | **8** | **11** | **~2,030** | **~2 sessions** |

Primary risk multiplier: H-matrix transcription + CRC-12 verification (1–2 hours of careful
work with reference decoder cross-check).

---

## 10. Open Questions

1. **D_JS8_MAX**: What is the actual max check-node degree in JS8's Nm[87]? Grep JS8.cpp for the
   largest `CheckNode.count` to set `D_JS8_MAX` before writing the BP kernel.
2. **JS8 min_score threshold**: FT8 uses min_score=3.0 (CLI default). JS8 traffic is lower
   density; same threshold should work but may need tuning after first on-air run.
3. **ZMQ port conflict**: Port 5581 is currently unused in the project. Confirm no other tool
   uses it before hardcoding.
4. **Dashboard layout**: Does the JS8 conversation panel replace or augment the FT8 decode panel?
   Recommendation: add a second tab/section — JS8 conversations have different utility
   (ongoing QSOs) vs FT8 spots (callsign grid exchanges).

---

## 11. Decision

**Ship Phase 1 → Phase 2 → Phase 3 in order.**

Phase 1 is prerequisite for all. Phase 2 and Phase 3 are independent cherry-picks — either
can be deferred without blocking the other.

Start with Phase 1. Validate on-air JS8 decodes on 40m and 20m. If decode counts look correct
(non-zero, plausible given band conditions), proceed to Phase 2 dashboard and Phase 3 spots.

The ring-sharing architecture is the key decision: JS8Cuda is a scan+LDPC consumer only, not
a full pipeline clone. This halves the new code volume vs a full FT8Cuda clone, eliminates
duplicate GPU memory for RFFT and ring, and means future protocol additions (FT4, WSPR) can
follow the same pattern: one ring, N consumers.

---

*Plan written by /plan-ceo-review, 2026-05-27*
