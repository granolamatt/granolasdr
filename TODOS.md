# TODOS

Updated 2026-05-31.

## Remaining Phase 12 work

Phase 11 (MagBlock, DeviceRingBuffer, JS8 integration) and Phase 12 (epoch
scan removal, 15s window accumulation, BufferFile, flow graph cleanup) are
complete. The following items remain from the original Phase 12 list:

- **Wideband waterfall resolution fix**: 2048 bins over 0–70 MHz gives ~34 kHz/bin — too coarse.
  Easiest fix: clamp the quadratic mapping to `[0, 30 MHz]` only by setting
  `rfft_bin_max = round(30e6 / 6.25)` before mapping to 2048 output bins.
  Gives ~7 kHz/bin across the amateur HF window with no canvas changes.

- **CUDA error checking**: add `cudaGetLastError()` checks to FT8Cuda + JS8Cuda
  kernel launches (MagBlock already has this from Phase 11). Phase 13 adds
  JS8FastScanCuda and extends JS8Cuda to a template — all three new kernel
  launch sites need the same treatment. T6 in Phase 13 implementation tasks.

- **Corpus re-enable**: MagBlock exposes `demodFT8_d` callback so a
  `--record`-based corpus capture path can be used for JTDX/WSJT-X comparison.

## Phase 13+ ZMQ port configurability

ZMQ ports for Fast/Slow/Turbo/Ultra modes (5591-5594) are hardcoded in HFRx.cc.
If any downstream tooling already uses these ports, there's a silent conflict.

- Add `--js8-fast-port`, `--js8-slow-port`, `--js8-turbo-port` CLI args to HFRx.cc
  with defaults 5591/5592/5593. Same pattern as existing `--min-score` arg.
- Three lines of argparse per mode; trivially done when the mode is wired in.
- Deferred from Phase 13 eng review (outside voice Point 5).

## QP-ADMM vs BP convergence baseline

Measure decode counts per epoch on corpus recordings with both decoders
at equal SNR. Target: QP-ADMM ≥ BP at FT8_GPU_CAND_MAX=500.
Gate: if QP-ADMM loses >2% decodes vs BP on real traffic, investigate
rho/max_iter tuning before enabling by default.
Prerequisite: corpus session recorded with `--record`.
