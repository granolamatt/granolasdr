# TODOS

Updated 2026-05-30.

## Phase 11: Flow graph refactor (ready to implement)

See CEO plan: `~/.gstack/projects/granolamatt-granolasdr/ceo-plans/2026-05-30-phase11-magblock.md`

### DeviceRingBuffer<T, N>

Replace the five ad-hoc ring accessors on `FT8Cuda` with a single typed struct.
Lives in `gm/buffer/DeviceRingBuffer.h`.

**Confirmed slot size** (from FT8Cuda.cc line 154):
`slot_bytes = FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length * sizeof(T)`

```cpp
template<typename T, int N>
struct DeviceRingBuffer {
    T*                    base_d;       // device allocation: N * slot_bytes
    size_t                slot_bytes;   // FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length
    size_t                num_bins;     // rfft_length (for scan functions)
    std::atomic<uint64_t> write_idx{0};
    cudaEvent_t           ready;

    T* slot(uint64_t idx) const {
        return base_d + (idx % N) * (slot_bytes / sizeof(T));
    }
};
```

### MagBlock

Extract magnitude computation out of `FT8Cuda` into a standalone block.
Owns: RFFT, |·|² kernel, ring write, waterfall → ZMQ.
Constructor takes: `BufferPosition<complex<float>>*` + zmq_port.
Exposes: `const DeviceRingBuffer<uint8_t, 200>& getRing()`.

`FT8Cuda` becomes a pure scanner: takes `const DeviceRingBuffer<uint8_t,200>&`.
`JS8Cuda` also takes `const DeviceRingBuffer<uint8_t,200>&` (drops FT8Cuda* dep).

**Note:** `enable_corpus` / `--jtdx` path removed in Phase 11. See Phase 12.

### Phase 11 test gate
Run `--playback /tmp/js8test.dat --js8` before and after.
FT8 + JS8 decode counts must match exactly (deterministic input).

## Phase 12: Follow-on work (after Phase 11 ships)

- **Corpus re-enable**: MagBlock exposes `demodFT8_d` callback so FT8Cuda corpus
  path (`--jtdx`) can be restored without a second RFFT
- **CUDA error checking**: add `cudaGetLastError()` checks to FT8Cuda + JS8Cuda
  kernel launches (MagBlock already has this from Phase 11)
- **Wideband waterfall resolution fix** (see below)
- **QP-ADMM vs BP baseline** (see below)

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
