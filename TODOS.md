# TODOS

Updated 2026-05-30.

## Remaining Phase 12 work

Phase 11 (MagBlock, DeviceRingBuffer, JS8 integration) and Phase 12 (epoch
scan removal, 15s window accumulation, BufferFile, flow graph cleanup) are
complete. The following items remain from the original Phase 12 list:

- **Wideband waterfall resolution fix**: 2048 bins over 0–70 MHz gives ~34 kHz/bin — too coarse.
  Easiest fix: clamp the quadratic mapping to `[0, 30 MHz]` only by setting
  `rfft_bin_max = round(30e6 / 6.25)` before mapping to 2048 output bins.
  Gives ~7 kHz/bin across the amateur HF window with no canvas changes.

- **CUDA error checking**: add `cudaGetLastError()` checks to FT8Cuda + JS8Cuda
  kernel launches (MagBlock already has this from Phase 11).

- **Corpus re-enable**: MagBlock exposes `demodFT8_d` callback so a
  `--record`-based corpus capture path can be used for JTDX/WSJT-X comparison.

## Docker deployment

- **Docker smoke test** (P2, S effort): Add a compose `healthcheck` or standalone script that
  verifies `hf_rx` started cleanly — e.g. `curl http://localhost:8765/` returns 200 within 30s.
  Context: Docker image has no automated test; currently users must manually check `docker logs`.
  Start in `docker-compose.yml` healthcheck block using curl.

- **Docker layer caching** (P3, S effort): Split cmake configure from make into separate `RUN`
  layers so source code changes don't invalidate the dependency install layers and trigger a
  full 30-min CUDA rebuild. Currently cmake configure + nvcc compile are in one `RUN` layer.

## QP-ADMM vs BP convergence baseline

Measure decode counts per epoch on corpus recordings with both decoders
at equal SNR. Target: QP-ADMM ≥ BP at FT8_GPU_CAND_MAX=500.
Gate: if QP-ADMM loses >2% decodes vs BP on real traffic, investigate
rho/max_iter tuning before enabling by default.
Prerequisite: corpus session recorded with `--record`.
