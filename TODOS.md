# TODOS

Updated 2026-05-27. Deferred items from prior plan reviews. None are blocked — each waits on a natural predecessor.

## JS8 Normal decode (Phase 10)

Plan: PLAN_JS8_PHASE10.md (2026-05-27). Architecture: JS8Cuda reads from FT8Cuda's shared mag
ring; no second RFFT or ring allocation needed. Three phases:

- **Phase 1 (core)**: JS8ScanCuda (Costas {4,2,5,6,1,3,0}), JS8LdpcCuda (M=87 H-matrix),
  JS8Cuda (ring consumer), gm/hf/js8 (CRC-12 + message decode + ZMQ 5581). ~1,800 lines.
  **Primary risk**: H-matrix transcription from JS8Call Nm[87] — cross-check with reference decoder.
  **Fix needed**: change `ring_write_idx` from `uint64_t` to `std::atomic<uint64_t>` in FT8Cuda
  (latent race that JS8Cuda would expose; existing contWorker also has this race).

- **Phase 2 (cherry-pick)**: Dashboard JS8 conversation panel — `broadcastJS8()` SSE + HTML panel.
  ~160 lines. Independent of Phase 3.

- **Phase 3 (cherry-pick)**: PSKReporter JS8 spots — psk_uploader.py subscribes to port 5581,
  mode field parameterized. ~70 lines. Independent of Phase 2.

**Note on FT4**: FT4 previously slated as Phase 10; superseded by JS8 (more traffic, larger
user base, conversational protocol adds dashboard value). FT4 becomes Phase 11 or later.

## GPU LDPC post-ship items

From /plan-ceo-review (2026-05-27), Phases 7–9 CEO plan — **FT8_GPU_CAND_MAX reduction and cascade timeout detection are in active Phase 7 plan**. Items below are the remaining deferred sub-items.

- **QP-ADMM vs BP convergence baseline**: measure decode counts per epoch on D8 corpus WAV files
  with both decoders at equal SNR. Target: QP-ADMM ≥ BP at FT8_GPU_CAND_MAX=500.
  Gate: if QP-ADMM loses >2% decodes vs BP on real traffic, investigate rho/max_iter tuning first.
  **Prerequisite**: --jtdx corpus WAV files. Add when WAV corpus exists.

## Wideband waterfall resolution

Shipped in Phase 9 (2026-05-27) but resolution is too low to be useful.
The 2048-bin output covers the full 0–70 MHz composite range, giving ~34 kHz/bin — bands
are dots, not features. Options to consider:

- **Zoom to HF window only** (1–30 MHz): restrict the output bin mapping to the HF sub-range
  of the composite spectrum, discarding everything above 30 MHz. Gives ~7 kHz/bin across 2048 bins.
- **Per-band zoom windows**: add a selector to the dashboard that streams a 2048-bin slice centred
  on a chosen 500 kHz window (e.g. 14.0–14.5 MHz). Would need a frequency parameter on the
  WebSocket URL or a separate endpoint per zoom level.
- **Increase output bins**: change WATERFALL_BINS from 2048 to a larger value (e.g. 8192) to
  spread the quadratic mapping over more pixels. Canvas width would need to match.

Easiest first step: restrict the quadratic mapping to [0, 30 MHz] only by clamping
`rfft_bin_max = round(30000000 / 6.25)` and mapping `b → round(rfft_bin_max * (b/2047)²)`.

## FT4 decode pipeline

Second decode consumer alongside FT8, using the same Costas scan / LDPC family.
- 7.5-second epoch (half of FT8); separate GPU scan parameters and epoch trigger
- Reuse most of FT8Cuda machinery; add second ZMQ PUB on a different port; PSKReporter accepts FT4
- L effort: ~400 lines across FT8Cuda.cc, a new FT4.cc, CMakeLists
- **Prerequisite**: wideband waterfall (Phase 9) shipped. Planned as Phase 10.
