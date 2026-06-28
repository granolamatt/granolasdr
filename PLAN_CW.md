# CW Skimmer — Plan

**Status:** DRAFT (eng review in progress)
**Authored:** 2026-06-27
**Mode:** SELECTIVE EXPANSION → full-band skimmer, local output, RBN deferred
**Review:** /plan-ceo-review (scope) → /plan-eng-review (this doc)

---

## 1. Problem Statement

granolasdr decodes FT8 and all five JS8 speed modes simultaneously across every HF band
from one RX888 + GPU pipeline. CW (Morse) is the oldest and still one of the most active
HF modes, and the hardware already hears every CW signal in the capture bandwidth. We are
discarding all of it at decode time.

Goal: decode CW across all 11 HF bands at once, like a wideband CW skimmer, surface decoded
text + extracted callsigns on the dashboard and the ZMQ bus, and report spots to the Reverse
Beacon Network (RBN). No new hardware. No new sample pipeline. RBN reporting reuses the existing
ZMQ XPUB/XSUB proxy: the decoder publishes `cw/decode`, and `rbn_uploader.py` subscribes and
relays to RBN, exactly as `psk_uploader.py` does for PSKReporter. The uploader stays OFF (or
log-only) until the Phase 0 copy-% bar is met on-air, so we never pollute RBN with bad copy.

### Why CW is not "JS8 with new constants"

JS8 reused the entire FT8 GPU decode core because it shares the frame: 79 symbols, Costas
sync, LDPC. CW shares none of that. No frame, no sync word, no FEC, no fixed timing. CW is
on-off keying of a bare carrier, hand-sent at 5–40+ WPM with irregular operator timing
("fist"). We reuse the front end (channelizer + mag ring machinery) and the output plumbing
(ZMQ, wsdict, dashboard). The decode core is entirely new, and a good CW decoder on real
band noise is the hard part of this plan.

---

## 2. Architecture

CW needs its **own** time-frequency front end. Two numbers govern it, and they are NOT the
same thing:

- **Window (time resolution)** = `rfft / sample_rate`. The FFT integration length. This is
  what limits how fast a keying edge you can resolve. Must be ≤ a dit (`1200/WPM` ms: 30 WPM
  = 40 ms, 40 WPM = 30 ms).
- **Hop (sample spacing)** = `window / time_osr`. How often a magnitude row is emitted. Set
  to ~5 ms for a smooth envelope. The hop does NOT improve resolution; only the window does.

Existing rings fail on the window axis (Normal 160 ms, Ultra ~32 ms windows — far longer
than a dit). The CW front end is sized so the window matches CW keying.

**Default: entire CW band, all 9 bands.** CWChannelizer composite = ~4,700 selected bins →
**8192-pt IFFT → 819.2 kHz** CW stream. CW MagBlock over that: **`rfft=16384, time_osr=4,
freq_osr=1` → 50 Hz/bin, 20 ms window, 5 ms hop.** The 819.2 kHz rate is what makes a 16384
FFT a 20 ms window (at the half-rate it would be 40 ms). ~8 MB ring VRAM at N=128.
Window-vs-bin-width is the CW tradeoff and a Phase 0 knob: if fast CW (>30 WPM) clips, drop
to `rfft=8192` (100 Hz/bin, 10 ms window, good past 40 WPM).

| dit (window must beat) | 20 WPM | 26 WPM (median) | 36 WPM (max seen) | 40 WPM |
|---|---|---|---|---|
| dit length | 60 ms | 46 ms | 33 ms | 30 ms |
| samples/dit @ 5 ms hop | 12 | 9 | 7 | 6 |
| 20 ms window resolves? | yes | yes | marginal | no → use rfft=8192 |

Forward construction per ARCHITECTURE.md (own stream, const-ref upstream, no callbacks):

```
HFChannelizer
  └─ wideband FFT (fftData_d, 1.4M-pt, 100 Hz/bin)  ──exposed as BufferPosition──┐
     (FT8/JS8 composite + audio sinks unchanged)                                 │
                                                                                 ▼
                                        CWChannelizer   own cudaStream_t
                                          CW bin-select (~4,700 bins) → 8192-pt IFFT → 819.2 kHz
                                          └─ MagBlock<128> CW ring  rfft=16384, osr 4×1
                                             │                       → 50 Hz/bin, 20 ms window, 5 ms hop
                                             └─ CWSkimmerCuda   own cudaStream_t + contWorker
                                                │   1. peak-detect active CW bins (SNR across mag cols)
                                                │   2. extract per-bin magnitude envelope
                                                └─ gm::hf::CW   CPU: envelope → Morse → text
                                                     threshold + WPM estimate + element segment
                                                     + Morse table + callsign validate + dedup/TTL
                                                     → ZMQ "cw/decode" + wsdict granolasdr:cw:heard:CALL
```

GPU/CPU split mirrors `FT8Cuda → gm::hf::FT8`: GPU does the parallel signal work (peak
pick + envelope gather across all bins), CPU runs the branchy sequential Morse state
machine. The Morse decoder does not belong on the GPU.

### Decoder state machine (per active bin)

```
 envelope[t] (5 ms, 200 Hz)
     │
     ▼
 [noise floor EMA] ──► threshold (hysteresis: T_hi, T_lo)
     │
     ▼
 key-down / key-up  ──► run-length encode  ──► (mark_ms, space_ms) stream
     │
     ▼
 [adaptive dit estimator]  (running estimate of unit length from short-mark cluster)
     │
     ▼
 classify:  mark <2·dit → DIT, else DAH
            space <2·dit → elem gap, 2–5·dit → CHAR gap, >5·dit → WORD gap
     │
     ▼
 Morse table → chars → words
     │
     ▼
 callsign-shape validation ──► confident CALL → wsdict cw:heard:CALL
 raw text ──────────────────► ZMQ cw/decode + dashboard panel
```

---

## 3. Success Criteria

1. **Phase 0 gate (the real bar):** copy a multi-signal contest recording with two signals
   <80 Hz apart and a ≥30 WPM signal under QSB, to a defined copy-% target, vs ground truth.
   Clean-signal copy alone does not pass.
2. **Live multi-signal:** with `--cw`, decode multiple simultaneous CW signals on 40m
   during an active evening, callsigns plausibly matching the band.
3. **Zero regression:** FT8 and JS8 decode rate and timing unchanged with `--cw` on. CW waits
   on the wideband-FFT ring event but never backpressures HFChannelizer; FT8/JS8 composite
   byte-identical with `--cw` on vs off.
4. **Dashboard:** CW panel shows decoded text + freq + WPM + SNR + time, live.
5. **Garbage control:** callsign validation suppresses obvious noise; `cw:heard:CALL`
   only fires on callsign-shaped tokens with a confidence gate.

---

## 4. Implementation Phases

### Phase -1 — CW coverage via a dedicated CWChannelizer (PREREQUISITE, P0 BLOCKER)

**The channelizer does not currently capture any CW sub-band.** Every `kHFBands` window
sits on the FT8/JS8 dial frequencies, above the CW portion of each band (40m window
7.068–7.082 vs CW 7.000–7.040; 20m window 14.070–14.100 vs CW 14.000–14.070). CW capture
today is ~zero. Nothing downstream (not even the Phase 0 spike) can see CW until coverage exists.

**Chosen approach: a separate `CWChannelizer` block, NOT expansion of the shared composite.**
`HFChannelizer`'s 1.4M-pt wideband FFT (`fftData_d`, 100 Hz/bin, 0–70 MHz) already contains
every CW frequency. `CWChannelizer` taps that same wideband FFT, bin-selects CW windows into
its OWN composite + IFFT, and feeds its own CW MagBlock ring. The expensive 1.4M FFT is not
duplicated. This decouples CW completely from the FT8/JS8 composite: no contention for the
4096-bin budget, and zero blast radius on SpectrumNorm, the waterfall, the audio sinks, or
the FT8/JS8/audio paths that already work. CW gets its own composite budget sized for CW alone.

```
HFChannelizer
  ├─ wideband FFT (fftData_d, 1.4M-pt, 100 Hz/bin)  ── exposed as BufferPosition ──┐
  ├─ FT8/JS8 composite (2110 bins → 409.6 kHz) → MagBlock rings (unchanged)        │
  └─ audio sinks (unchanged)                                                       │
                                                                                   ▼
                                                        CWChannelizer (own stream, waits on wideband-FFT ring event)
                                                          CW bin-select → 8192-pt IFFT → 819.2 kHz
                                                          → CW MagBlock<128> (rfft=16384, osr 4×1)
```

### RBN coverage data (skimmed 2026-06-27, reversebeacon.net)

WPM distribution: min 8, p10 14, median 26, p90 31, max 36. First decoder target 14–31 WPM
(easy-timing machine/contest range). 5 ms CW ring resolves 40 WPM (30 ms dit = 6 samples).

CW window sizing (RBN clustering + band-plan extents for bands quiet at sample time):

| Band | CW range (MHz) | Window kHz | Bins @100 Hz |
|------|----------------|-----------|--------------|
| 160m | 1.800–1.840 | 40 | 400 |
| 80m  | 3.500–3.560 | 60 | 600 |
| 40m  | 7.000–7.050 | 50 | 500 |
| 30m  | 10.100–10.130 | 30 | 300 |
| 20m  | 14.000–14.070 | 70 | 700 |
| 17m  | 18.068–18.098 | 30 | 300 |
| 15m  | 21.000–21.090 | 90 | 900 |
| 12m  | 24.890–24.920 | 30 | 300 |
| 10m  | 28.000–28.070 | 70 | 700 |

**CW composite sizing (the load-bearing number):** entire CW band, all 9 ≈ 4,700 bins →
**8192-pt CW IFFT → 819.2 kHz**. CW MagBlock over that: rfft=16384, osr 4 → 50 Hz/bin, 20 ms
window, 5 ms hop. The higher 819.2 kHz rate is a feature: it makes a 16384 FFT a 20 ms window
(short enough for CW keying) instead of 40 ms. All of this is contained to the CW path; FT8/JS8
untouched. (Active-bands-only at 4096-pt/409.6 kHz is the fallback if VRAM ever matters — it
does not here, ~8 MB ring.)

**`fftData_d` is a single in-place buffer (`HFChannelizer.h:56`), overwritten every block.**
"Expose as a BufferPosition" is not enough — a second stream reading it races the writer.
Resolution: HFChannelizer keeps a **small ring (depth 2–3) of the wideband FFT** (R2C → ~700k
complex bins × 8 B × depth ≈ 11–17 MB VRAM, budgeted), with a per-slot event. CWChannelizer
`cudaStreamWaitEvent`s on the committed slot's event. Honest cost: this is a **one-directional
event dependency** (CW waits on the wideband FFT being ready) but NEVER backpressure onto
HFChannelizer — CW cannot stall FT8/JS8. The earlier "no shared-stream sync" wording was wrong;
the accurate claim is "CW waits on the wideband-FFT event; HFChannelizer never waits on CW."

| Item | File | Notes |
|------|------|-------|
| CW-1.1 | (done) | RBN skim complete — see table above. CW windows sized from real activity |
| CW-1.2 | `gm/cuda/HFChannelizer.h/.cc` | Wideband FFT as a **depth-2/3 ring** of `complex<float>` + per-slot event (NOT a single shared ptr). Budget ~11–17 MB VRAM |
| CW-1.3 | composite-builder helper | Extract bin-select + composite-assembly + IFFT shared by HFChannelizer and CWChannelizer (DRY) |
| CW-1.4 | `gm/cuda/CWChannelizer.h/.cc` (new) | Own stream; `cudaStreamWaitEvent` on wideband-FFT ring slot; CW bin-select → 8192-pt IFFT → 819.2 kHz → CW `MagBlock<128>(rfft=16384, time_osr=4, freq_osr=1)` |
| CW-1.5 | `gm/hf/cw_bands.h` (new) | CW band windows (separate from `kHFBands`), sized from RBN activity |
| CW-1.6 | `gm/cuda/MagBlock.cc` | Plumb `sample_rate` into MagBlock: `bin_hz_` and `freqShift` hard-code 409600 (`MagBlock.cc:27,102`); at 819.2 kHz `bin_hz_` reports 25 Hz (wrong; is 50 Hz) and freq_osr>1 would miscompute |
| CW-1.7 | composite-bin → RF demap | Reverse map from a CW composite bin index to real RF Hz + band (composite concatenates 9 disjoint sub-bands; bin index is not linear in frequency). Needed per detection for spots |
| CW-1.8 | verify | FT8/JS8/audio/waterfall byte-identical with `--cw` on vs off (decoupling proof) |

**Gate:** CW signals visible in the CW composite (waterfall/record). This gates **live** decode
(Phase 1+), NOT Phase 0 — Phase 0 runs on canned audio independent of this block.

### Phase 0 — Decoder spike (de-risk, GATE) — runs FIRST, zero pipeline dependency

The whole risk is the decoder. Prove it on **canned/synthetic CW audio that does NOT come
through the granolasdr pipeline** — fldigi-keyed WAVs, internet contest recordings, or a
synthetic envelope generator. This is the fix for the inverted-gate trap: the live capture
path (Phase -1) is the heaviest new block, so the decoder must be de-risked WITHOUT it.
`cw_morse` takes a magnitude/envelope array; any WAV or synthetic source feeds it.

| Item | File | Notes |
|------|------|-------|
| CW0.1 | `gm/hf/cw_morse.h/.cc` (new) | Pure decoder core: envelope→text. No CUDA, no ZMQ, no pipeline. Host-only |
| CW0.2 | synth + canned corpus | Synthetic envelope generator (known text @ WPM×SNR×QSB) + real fldigi-keyed/contest WAVs with ground truth |
| CW0.3 | `test/cw_decode_offline.cc` (new) | Standalone harness: WAV/synth → STFT (50 Hz/20 ms) → cw_morse → text, vs ground truth |

**Gate (hardened — the clean-signal test is a strawman):** copy a **multi-signal contest
recording** with (a) two signals <80 Hz apart and (b) a ≥30 WPM signal under QSB, hitting a
defined copy-% target. If the single-STFT (50 Hz/20 ms) front end or the adaptive-threshold
decoder fails this, the architecture is wrong, not just the threshold — escalate to per-signal
narrowband Viterbi (the CW Skimmer approach) BEFORE building Phase 1–3. Clean-signal copy is
necessary but NOT sufficient to pass.

### Phase 1 — Peak detection on the CW ring (CW MagBlock built in Phase -1, CW-1.4)

| Item | File | Notes |
|------|------|-------|
| CW1.1 | `gm/cuda/CWSkimmerCuda.h/.cc` (new) | Reads CW ring, own stream + contWorker. Peak-detect active bins per scan. Structural clone of `JS8Cuda` |
| CW1.2 | `gm/cuda/CWKernel.cu` (new) | GPU peak-pick + per-bin envelope-gather kernels |
| CW1.3 | `gm/HFRx.cc` | Wire CWSkimmerCuda; log detected bins + SNR + RF Hz (via CW-1.7 demap). No decode yet |
| CW1.4 | `CMakeLists.txt` | Add CWSkimmerCuda, CWKernel sources |

### Phase 2 — Decoder core, live

**The per-signal tracker is the hard part here, not a structural clone of JS8Cuda's batch
model.** The CW ring holds only ~640 ms (N=128 × 5 ms); a QSO is multiple seconds. Each
decoder instance must persist envelope/timing state across many ring wraps AND keep its
identity as the carrier drifts across 50 Hz bins. JS8 decodes a fixed 15 s frame in one shot;
CW is open-ended streaming per signal. Spec the tracker explicitly: signal lifetime,
bin-association/drift, ring-drain cadence, teardown on signal loss.

| Item | File | Notes |
|------|------|-------|
| CW2.1 | `gm/hf/cw_track.h/.cc` (new) | **Per-signal tracker**: associate detections→signals across ring wraps + bin drift; lifetime + teardown. The actual hard component |
| CW2.2 | `gm/hf/cw.h/.cc` (new) | Drive one `cw_morse` core per tracked signal; accumulate envelope across ring drains; batched |
| CW2.3 | `gm/HFRx.cc` | Forward-wire CWSkimmerCuda → gm::hf::CW |

### Phase 3 — Validation + integration

| Item | File | Notes |
|------|------|-------|
| CW3.1 | `gm/hf/cw.cc` | Callsign-shape validation + dedup with TTL (mirror `js8:heard` 900 s) |
| CW3.2 | `gm/hf/cw.cc` | ZMQ PUB `cw/decode` JSON (`"mode":"CW"`); wsdict `granolasdr:cw:heard:CALL` |
| CW3.3 | Dashboard HTML/JS | CW panel: EventSource/wsdict `cw` events, scrolling table. Clone JS8 panel |
| CW3.4 | `rbn_uploader.py` (new) | ZMQ subscriber on `cw/decode` (proxy XPUB :5600) → RBN, mirroring `psk_uploader.py`. Decoupled; no C++ changes. **Run gated on Phase 0 copy-% bar (off/log-only until proven).** |
| CW3.5 | `ARCHITECTURE.md` | Add CWChannelizer + CW ring + CWSkimmerCuda to topology + ms/block table; add `cw/decode` producer + rbn_uploader consumer to the ZMQ bus diagram |
| CW3.6 | `debug_cw.py` (new) | Offline inspect, clone of `debug_js8_fast.py` |

### Phase 4 — RBN ingestion spike (the one non-trivial part of reporting)

| Item | File | Notes |
|------|------|-------|
| CW4.1 | `rbn_uploader.py` (spike) | **RBN ingestion is NOT an open POST like PSKReporter** — spots flow via the RBN Aggregator / node-registration path. Spike the submission protocol + node registration; the ZMQ-subscriber half (CW3.4) is trivial, this half is the unknown |

---

## 5. Key Invariants

- CW path is independent: CWChannelizer (8192-pt IFFT → 819.2 kHz) + `MagBlock<128>` at
  rfft=16384/osr 4×1 → 50 Hz/bin, 20 ms window, 5 ms hop; never shares FT8/JS8 rings.
- Window (`rfft/rate`) must stay ≤ a dit; hop (`window/osr`) is just envelope smoothness.
- CWChannelizer/CWSkimmerCuda own their streams. CW waits on the wideband-FFT ring event but
  NEVER backpressures HFChannelizer (one-directional dependency); FT8/JS8 stay byte-identical.
- The Morse decoder (`cw_morse`) is pure CPU and CUDA-free so it is unit-testable offline.
- dit estimate is adaptive and per-bin; never a fixed WPM assumption.
- `cw:heard:CALL` only on callsign-shaped tokens past a confidence gate; raw text always streams.
- ZMQ JSON carries `"mode":"CW"` for downstream consumers.

---

## 6. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| **Single-STFT (50 Hz/20 ms) can't both separate <80 Hz signals AND resolve 30+ WPM keying** | High | Phase 0 gate tests exactly this; fallback = per-signal narrowband Viterbi (CW Skimmer approach). This is the architectural bet |
| **Adaptive-threshold decoder degrades on RBN-grade QSB/QRN/weak/bad-fist** | High | Phase 0 gate uses a real contest recording, not a clean signal; escalate to Viterbi if it fails |
| **Channelizer captures no CW** (P0 blocker for live) | Certain | Phase -1 CWChannelizer + RBN-sized `cw_bands.h`; verify CW visible before live decode |
| `fftData_d` read races the writer | Certain if unhandled | Depth-2/3 wideband-FFT ring + per-slot event; CW waits on committed slot (CW-1.2) |
| WPM estimation fails on short bursts (5-char CQ) | High | Seed dit estimate from band-typical default; refine as elements arrive; Phase 0 tuning |
| QSB breaks threshold mid-character | Medium | Hysteresis + noise-floor EMA; offline-tune on real fading captures |
| Adjacent-signal QRM bleeds across 50 Hz bins | Medium | Per-bin SNR gate; per-signal tracker keeps one decoder per signal |
| Drifting carrier walks across bins | Medium | Per-signal tracker (CW2.1) follows the carrier; bin-association across drift |
| Decoder emits callsign garbage to dashboard | Medium | Callsign-shape validation as the filter; confidence gate on `cw:heard` |
| CW ring VRAM/timing regresses FT8/JS8 | Low | Own stream; measure FT8 decode rate with `--cw` on vs off |

---

## 7. NOT in scope

- **Single-frequency-only decoder** — rejected in CEO review; we want the full-band skimmer.
- **TX / keying** — RX-only hardware.
- **Decoder ML/HMM approach (CW Skimmer-style Bayesian)** — start with adaptive threshold +
  timing; revisit only if the threshold decoder underperforms after Phase 0 tuning.
- **CW from the 48 kHz audio sinks** — the wideband mag-ring path decodes all signals at
  once; per-sink audio decode would throw away the skimmer advantage.

---

## 8. What already exists (reused, not rebuilt)

- `MagBlock<N>` — parameterized FFT→mag ring; CW reuses it with a new config, no new ring class.
- `DeviceRingBuffer<uint8_t,N>` multi-reader — CW skimmer is just another reader pattern.
- `BufferFile` `--record`/`--playback` — the Phase 0 offline iteration loop, already built.
- `JS8Cuda` structure — `CWSkimmerCuda` is a structural clone (own stream, contWorker, ring read).
- `gm::hf::JS8` / `gm::hf::FT8` — `gm::hf::CW` clones the ZMQ + wsdict + dedup/TTL plumbing.
- Dashboard JS8 conversation panel — CW panel is a clone.
- Host-only test pattern (`test_osd` compiles `OsdCore.cc`, no CUDA) — `cw_morse` gets the
  same: a CUDA-free `test_cw_morse` executable. GPU blocks follow `test_js8_fast_gpu_scan`.

---

## 9. Test Coverage

The decoder is the risk, and it is pure CPU (`cw_morse`), so it is fully unit-testable
host-only with NO CUDA, exactly like `test_osd`. The two eval harnesses (synthetic grid +
real corpus) ARE the Phase 0 gate.

```
CODE PATH                                          TEST (planned)                              kind
[+] gm/hf/cw_morse.cc   (pure CPU — host-only test, no CUDA link)
  ├ classifyElement(mark,dit)                      test_cw_morse: dit/dah @ 15/26/36 WPM        ★★★ unit
  ├ classifyGap(space,dit)                         test_cw_morse: elem/char/word boundaries     ★★★ unit
  ├ updateDitEstimate(marks)                       test_cw_morse: converge; WPM step; 5-char    ★★★ unit
  │                                                  burst; straight-key jitter; 8 & 40 WPM
  ├ morseToChar(sym)                               test_cw_morse: full table round-trip         ★★★ unit
  ├ threshold + hysteresis + noiseEMA              test_cw_morse: QSB mid-char; all-noise;      ★★★ unit
  │                                                  single dit; empty envelope
  └ decodeEnvelope(env[]) → text                   synth golden msg @ WPM×SNR grid              [→EVAL]
                                                    Phase 0 real recording vs fldigi/known       [→EVAL]
[+] gm/cuda/CWChannelizer.cc   (CUDA test)
  ├ bin-select CW windows                          test_cw_channelizer: tone@freq → exp bin     ★★★ unit
  ├ composite IFFT (8192)                          test_cw_channelizer: tone → right offset     ★★  unit
  └ DECOUPLING                                     FT8/JS8 composite byte-identical --cw on/off  ★★★ [→E2E] CRITICAL regression
[+] gm/cuda/HFChannelizer.cc   (modified: expose fftData_d)
  └ wideband-FFT BufferPosition accessor           accessor data == internal path               ★★★ unit
                                                    write_idx race (TSAN/stress)                 [→E2E]
[+] gm/cuda/CWSkimmerCuda.cc   (CUDA test)
  ├ peak-detect active bins                        test_cw_skimmer: N tones → exactly N, 0 in noise ★★★ unit
  ├ envelope extraction                            test_cw_skimmer: per-bin env == synth keying ★★★ unit
  └ carrier-drift tracking                         test_cw_skimmer: walking tone stays tracked  ★★  unit
[+] gm/hf/cw.cc   (CPU orchestration)
  ├ callsign validation                            test_cw: real calls pass, garbage rejected   ★★★ unit
  ├ dedup + TTL                                    test_cw: same call+freq in window → 1 spot   ★★  unit
  ├ batched per-bin decoders                       test_cw: 3 simultaneous signals independent  ★★  unit
  └ ZMQ "cw/decode" JSON                           test_cw: mode:"CW", fields present           ★   smoke

COVERAGE TARGET: every cw_morse branch unit-tested host-only; GPU blocks synthetic-tone tested;
1 CRITICAL decoupling regression; 2 eval harnesses. CMake: add test_cw_morse (host-only, like
test_osd), test_cw_channelizer / test_cw_skimmer (CUDA-linked, like test_js8_fast_gpu_scan).
```

## 10. Failure Modes

| New codepath | Realistic production failure | Test? | Error handling? | Silent? |
|---|---|---|---|---|
| `fftData_d` accessor | CW reads while channelizer writes (data race) | TSAN/stress | atomic write_idx + event (mag-ring pattern) | would be silent → **must test** |
| CW composite IFFT | total_bw > 8192 after a window edit | unit | bounds-check + log, clamp | guarded |
| Peak detector | contest pileup: candidates exceed cap | unit | cap + **log dropped count** (no silent truncation) | guarded |
| dit estimator | straight-key fist breaks WPM lock | unit | bounded estimate; degraded copy, never crash | visible (bad text) |
| Carrier drift | signal walks out of tracked bins | unit | per-decoder timeout, clean teardown | visible (decode ends) |
| Callsign validation | decoder emits garbage call to dashboard | unit | shape-validate + confidence gate before `cw:heard` | guarded |
| CW path GPU cost | extra 8192 IFFT + 32768-wait regresses FT8/JS8 | A/B timing | own stream, no cross-sync | **measure FT8 rate --cw on/off** |

**Critical gap watch:** the `fftData_d` accessor race is the one failure that would be both
silent and corrupting. It MUST ship with the atomic write_idx + event and a stress test, not
as a follow-up. Everything else is either guarded or visibly degrades.

## 11. Implementation Tasks

Synthesized from this review. P1 blocks ship, P2 same branch, P3 follow-up. Effort: human / CC.

- [ ] **T1 (P1, human ~1h / CC ~10m)** — Phase -1 — `gm/hf/cw_bands.h`: CW windows from the RBN table (§4), total_bw ≈ 4,700.
- [ ] **T2 (P1, human ~4h / CC ~20m)** — Phase -1 — `HFChannelizer.h/.cc`: wideband FFT as a depth-2/3 `complex<float>` **ring** + per-slot event (NOT a single shared ptr); budget ~11–17 MB. **Ship with stress test.** _(outside voice #1)_
- [ ] **T2b (P1, human ~2h / CC ~15m)** — Phase -1 — `MagBlock.cc`: plumb `sample_rate` (kill hard-coded 409600 at `:27,:102`) so `bin_hz_`/`freqShift` are correct at 819.2 kHz. _(outside voice #8)_
- [ ] **T2c (P1, human ~2h / CC ~15m)** — Phase 1 — composite-bin → RF-Hz + band reverse map for the packed CW composite; per detection, for spots. _(outside voice #7)_
- [ ] **T2d (P1, human ~1d / CC ~45m)** — Phase 2 — `gm/hf/cw_track.h/.cc`: per-signal tracker (lifetime, bin-association across drift, ring-drain cadence, teardown). The actual hard part of live decode. _(outside voice #6)_
- [ ] **T3 (P1, human ~3h / CC ~20m)** — Phase -1 — extract shared composite-builder (bin-select + assemble + IFFT) from HFChannelizer for DRY reuse by CWChannelizer.
- [ ] **T4 (P1, human ~1d / CC ~40m)** — Phase -1 — `gm/cuda/CWChannelizer.h/.cc`: 8192-pt CW composite IFFT → 819.2 kHz; own stream; const-ref wideband FFT; forward-construct in HFRx.cc behind `--cw`.
- [ ] **T5 (P1, human ~2h / CC ~15m)** — Phase -1 — `test_cw_channelizer` (CUDA) + the **CRITICAL** FT8/JS8 byte-identical decoupling regression.
- [x] **T6 (P1) DONE** — Phase 0 — `gm/hf/cw_morse.h/.cc`: pure-CPU decoder core (smoothing + AGC + Otsu + two-pass timing + Morse table). Commit 042f5a4.
- [x] **T7 (P1) DONE** — Phase 0 — `test_cw_morse` host-only gate: clean 18/26/36 WPM exact, adaptive lock, 12 dB noise, workable QSB. Commit 042f5a4.
- [ ] **T8 (P1, human ~2h / CC ~15m)** — Phase 0 — `test/cw_decode_offline.cc`: `--record` capture → cw_morse → text; validate vs fldigi/known. **GATE.**
- [ ] **T9 (P1, human ~1d / CC ~40m)** — Phase 1 — `gm/cuda/CWSkimmerCuda.h/.cc` + `CWKernel.cu`: peak-detect + envelope gather; own stream; `test_cw_skimmer`.
- [ ] **T10 (P1, human ~1d / CC ~45m)** — Phase 2 — `gm/hf/cw.h/.cc`: drive cw_morse per active bin, batched; carrier-drift tracking; wire CWSkimmerCuda → gm::hf::CW.
- [ ] **T11 (P1, human ~4h / CC ~30m)** — Phase 3 — callsign validation + dedup/TTL + ZMQ `cw/decode` + wsdict `cw:heard`; `test_cw`.
- [ ] **T11b (P2, human ~2h / CC ~15m)** — Phase 3 — `rbn_uploader.py`: ZMQ subscriber on `cw/decode` → RBN, mirroring `psk_uploader.py`. Off/log-only until Phase 0 copy-% bar met.
- [ ] **T11c (P2, human ~4h / CC ~30m)** — Phase 4 — RBN ingestion spike: confirm the Aggregator/node-registration submission protocol (not an open POST like PSKReporter).
- [ ] **T12 (P2, human ~3h / CC ~20m)** — Phase 3 — dashboard CW panel (clone JS8); `debug_cw.py`.
- [ ] **T13 (P2, human ~1h / CC ~10m)** — Phase 3 — `ARCHITECTURE.md`: add CWChannelizer + CW ring to topology + ms/block table.
- [ ] **T14 (P2, human ~1h / CC ~10m)** — measure FT8/JS8 decode rate with `--cw` on vs off (zero-regression proof).

## 12. Parallelization

| Lane | Steps | Modules | Depends on |
|------|-------|---------|------------|
| A | T6 → T7 → T8 (decoder spike) | `gm/hf/cw_morse`, `test/` | — (pure CPU, fully independent) |
| B | T1 → T2 → T3 → T4 → T5 (channelizer) | `gm/cuda/`, `gm/hf/cw_bands.h` | — |
| C | T9 → T10 → T11 → T12 (skimmer + decode + UI) | `gm/cuda/CWSkimmerCuda`, `gm/hf/cw` | B (needs CW composite), A (needs cw_morse) |

**Launch A and B in parallel** (worktrees — A is pure CPU, B is GPU/channelizer, zero shared
files). Both gate before C: A must pass the Phase 0 decoder gate, B must pass the decoupling
regression. Then C. T13/T14 trail C.

---

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 (informal) | ACCEPTED | Scope set: full-band skimmer, local output, RBN deferred |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | ISSUES_OPEN | 10 findings; 1 critical (fftData_d race, mitigation specced); 1 strategic unresolved |
| Outside Voice | Claude subagent | Independent 2nd opinion | 1 | ISSUES_FOUND | 8 findings; 7 folded into plan, 1 strategic escalated to user |

**OUTSIDE VOICE:** 8 findings. Folded: fftData_d single-buffer race (#1 → depth-2/3 ring),
inverted Phase 0 gate (#2 → canned-audio, pipeline-independent), weak gate (#3/#4 → hardened
to multi-signal contest copy + Viterbi fallback named), per-signal tracker (#6 → first-class
item T2d), bin→RF demap (#7 → T2c), config contradictions + MagBlock hard-coded 409600
(#8 → T2b, osr pinned at 4). Escalated to user: #5 (RBN-deferred strategy).

**CROSS-MODEL TENSION (resolved):**
- **RBN scope (#5):** Outside voice argued local-only CW duplicates fldigi/CW Skimmer and
  defers the differentiated value. **Resolved by the user:** RBN reporting reuses the existing
  ZMQ XPUB/XSUB proxy — `cw/decode` is already published, so `rbn_uploader.py` is just another
  subscriber (like `psk_uploader.py`), zero C++ cost, fully decoupled. RBN is now IN scope
  (Phase 3, T11b), enabled gated on the Phase 0 copy-% bar so quality is still proven first.
  Both perspectives satisfied. One residual unknown moved to a spike (RBN ingestion protocol).

**VERDICT:** CEO + ENG review CLEARED — build-ready. Decoupled CWChannelizer (wideband-FFT
ring tap), entire-band 8192/819.2 kHz, 50 Hz/20 ms front end with Viterbi fallback, decoder
gated on canned-audio Phase 0, RBN as a ZMQ-subscriber Python relay gated on copy quality.
The hardened Phase 0 gate (multi-signal contest copy) is the go/no-go before Phase 1–3.

NO UNRESOLVED DECISIONS
