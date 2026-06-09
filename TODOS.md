# TODOS

Updated 2026-06-09.

## Phase 15: Turbo + Ultra JS8 modes

CEO review complete (2026-06-09). Eng review complete (2026-06-09). Ready to implement.

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
