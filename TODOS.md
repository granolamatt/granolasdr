# TODOS

Updated 2026-05-30.

## Flow graph refactor (next major phase)

The target architecture (see ARCHITECTURE.md) requires two new primitives.
These should be built before adding new decoders so future work lands on
clean foundations.

### DeviceRingBuffer<T, N>

Replace the five ad-hoc ring accessors on `FT8Cuda` (`getRingPtr`,
`getRingWriteIdx`, `getRingBlocks`, `getRingReadyEvent`, `getRingNumBins`)
with a single typed struct:

```cpp
template<typename T, int N>
struct DeviceRingBuffer {
    T*                    base_d;      // device allocation: N * slot_elems
    size_t                slot_elems;
    std::atomic<uint64_t> write_idx{0};
    cudaEvent_t           ready;

    T* slot(uint64_t idx) const { return base_d + (idx % N) * slot_elems; }
};
```

- Lives in `gm/buffer/DeviceRingBuffer.h`
- `FT8Cuda` constructs one and exposes `const DeviceRingBuffer<uint8_t, 200>& getRing()`
- `JS8Cuda` takes `const DeviceRingBuffer<uint8_t, 200>&` instead of `FT8Cuda*`
- Breaks the JS8Cuda → FT8Cuda compile-time dependency

### MagBlock

Extract the magnitude computation out of `FT8Cuda` into a standalone block:

```
input:  DeviceBuffer<complex<float>>     (HFChannelizer output, rfft_length bins)
output: DeviceRingBuffer<uint8_t, 200>   (shared read-only by all scanners)
```

Moves these responsibilities out of `FT8Cuda`:
- Per-block RFFT on channelizer output
- `|·|²` magnitude + uint8 decimation kernel
- Ring slot write + `cudaEventRecord(ready)`
- Waterfall decimation kernel → ZMQ "waterfall" publish

`FT8Cuda` becomes a pure scanner: takes a `DeviceRingBuffer` input, runs the
Costas scan, emits `ContScanResult` for the CPU decode stage.

Build order: `DeviceRingBuffer` first (header only, no GPU code), then
`MagBlock` extraction, then update `FT8Cuda` and `JS8Cuda` constructors.

## Wideband waterfall resolution

Shipped in Phase 9 but 2048 bins over 0–70 MHz gives ~34 kHz/bin — too coarse.
Easiest fix: clamp the quadratic mapping to `[0, 30 MHz]` only by setting
`rfft_bin_max = round(30e6 / 6.25)` before mapping to 2048 output bins.
Gives ~7 kHz/bin across the amateur HF window with no canvas changes.

## QP-ADMM vs BP convergence baseline

Measure decode counts per epoch on D8 corpus WAV files with both decoders
at equal SNR. Target: QP-ADMM ≥ BP at FT8_GPU_CAND_MAX=500.
Gate: if QP-ADMM loses >2% decodes vs BP on real traffic, investigate
rho/max_iter tuning before enabling by default.
Prerequisite: corpus WAV files captured with `--record`.
