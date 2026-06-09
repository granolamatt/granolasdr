# granolasdr Architecture

## Core concept: typed flow graph over a GPU ring

granolasdr is a **typed flow graph** where each processing block connects to the
next through an explicitly-typed buffer. The runtime is the data itself — when
an upstream block writes a buffer and signals its completion event, the downstream
block's CUDA stream begins work without any CPU thread wakeup in between.

This mirrors the structure of offline SDR tools (GNU Radio, liquid-dsp) but
targets online GPU execution: the "scheduler" is `cudaStreamWaitEvent`, not a
thread pool.

## Buffer kinds

```
BufferPosition<T>        pinned or device memory  CPU semaphore / condition_variable
DeviceRingBuffer<T,N>    device circular ring      cudaEvent_t + atomic write_idx
```

`BufferPosition<T>` is the universal buffer interface — it holds a typed pointer,
a shape, and a monotonic position counter. Any producer calls `setPosition()` to
signal a new block; any consumer calls `getPosition()` to block until one arrives.

`DeviceRingBuffer<T,N>` supports **multiple independent readers** — each reader
tracks its own read cursor against a shared atomic `write_idx`. This is how one
magnitude ring feeds FT8Cuda, JS8Cuda, and WaterfallCuda simultaneously.

## BufferFile<T>

`BufferFile<T>` is a `Thread` that attaches to any `BufferPosition<T>` in either
direction:

- **Playback** (`BufferFile(path)`) — allocates an internal GPU ring, reads the
  file block-by-block, and feeds `BufferPosition<T>` at the recorded pace. Drop-in
  source for any pipeline stage.
- **Record** (`BufferFile(src, path, params)`) — drains `BufferPosition<T>` via
  `getPosition()`, D2H-copies each block, and writes it to file. Drop-in sink for
  any pipeline stage.

The same binary format (`BufferFileHeader` magic `0x474E4C48`) is used for both
directions. This means **any edge in the pipeline can be saved and replayed** without
changing the surrounding blocks.

```
Playback:   BufferFile<complex<float>>(path)  →  BufferPosition  →  MagBlock  → …
Record:     HFChannelizer  →  BufferPosition  →  BufferFile<complex<float>>(path)
```

## Pipeline topology (current)

```
rx888 (USB, 140 MS/s int16_t)
  └─ BufferPosition<int16_t>              pinned host, CPU semaphore
       └─ HFChannelizer                  CUDA thread — owns one cudaStream_t
            │  1,400,000-pt R2C FFT → 100 Hz/bin (0–70 MHz)
            │  11-band bin-select → 2110-bin composite
            │  SpectrumNorm (every 64 frames):
            │    asymmetric EMA (α_down=0.10, α_up=0.005) per composite bin
            │    Legendre degree-3 poly fit → per-bin equalization gains
            │    cross-band leveling: geometric mean of 11 band-center EMAs
            │  4096-pt C2C IFFT → 2048 valid samples @ 409.6 kHz
            │  4 tunable audio sinks: 480-bin gather → batched C2C IFFT → 48 kHz ZMQ
            │    cmdWorker: publishes granolasdr:audio:status:N via wsdict
            │               subscribes granolasdr:audio:cmd for retune commands
            └─ BufferPosition<complex<float>>   409.6 kHz, 2048 samples/block
                 │
                 ├─ [optional] BufferFile<complex<float>>   (--record / --playback)
                 │
                 └─ MagBlock<200>               CUDA thread — normal ring
                      │  65536-pt C2C FFT, 4 time × 4 freq OSR → 6.25 Hz/bin
                      │  magKernel: complex → uint8_t dB-compressed
                      │  DeviceRingBuffer<uint8_t, 200>   ~200 MB VRAM
                      ├─ WaterfallCuda          CUDA thread
                      │    2048-col RGBA colormap → wsdict granolasdr:waterfall:N
                      ├─ FT8Cuda                CUDA thread
                      │    Costas scan + LLRs → callback
                      │    └─ gm::hf::FT8       CPU thread
                      │         15s best-SNR window → ZMQ ft8/decode
                      │         wsdict granolasdr:ft8:heard:CALL  (TTL 900s)
                      └─ JS8Cuda<200> Normal    CUDA thread (--js8)
                           Costas scan + LLRs → callback
                           └─ gm::hf::JS8       CPU thread
                                per-epoch dedup → ZMQ js8/decode
                                wsdict granolasdr:js8:heard:CALL

Optional fast ring (--js8-fast):
  MagBlock<128>  40960-pt FFT, 2×2 OSR, 10 Hz/bin, 100 ms/block
    └─ JS8Cuda<128> Fast → gm::hf::JS8 (10s cycle)

Optional slow ring (--js8-slow):
  MagBlock<128>  131072-pt FFT, 2×2 OSR, 3.125 Hz/bin, 320 ms/block
    └─ JS8Cuda<128> Slow → gm::hf::JS8 (30s cycle)

Optional turbo ring (--js8-turbo):
  MagBlock<128>  20480-pt FFT, 2×2 OSR, 20 Hz/bin, 50 ms/block
    └─ JS8Cuda<128> Turbo → gm::hf::JS8 (6s cycle)

Optional ultra ring (--js8-ultra):
  MagBlock<128>  13107-pt FFT (Bluestein), 2×2 OSR, ~31 Hz/bin, ~32 ms/block
    └─ JS8Cuda<128> Ultra → gm::hf::JS8 (4s cycle)

Playback path:
  BufferFile<complex<float>>(path) → BufferPosition<complex<float>> → MagBlock → …
```

## Block contract

Each processing block:

1. **Owns one `cudaStream_t`** — all GPU work for that block runs on this stream.
2. **Declares typed inputs** — constructor takes a const reference to the upstream buffer.
3. **Declares typed outputs** — exposes a buffer reference for downstream blocks.
4. **Signals completion** — records a `cudaEvent_t` on its stream after each write.
5. **Waits on upstream** — calls `cudaStreamWaitEvent(myStream, input.ready, 0)` before reading.
6. **Does not block the CPU** — GPU-to-GPU handoffs never stall a CPU thread.

CPU blocking (`getPosition()`, `usleep`) is reserved for I/O boundaries (USB/PCIe
input, file playback/record) where the CPU must pace the data rate.

## Decode pipeline: flow not callbacks

Downstream nodes self-wire to their upstream in their own constructor. The
top-level `runPipeline` reads as a pure forward construction sequence — no lambdas
or callback registration at the orchestration level.

```cpp
// HFRx.cc runPipeline — each line constructs the next stage downstream
MagBlock<200> magblock(&buf, kNormalRfftLen, kNormalTimeOsr, kNormalFreqOsr, kProxyXSubPort);
FT8Cuda ft8channel(magblock.getRing(), min_score, "EPOCH", kProxyXSubPort, legacy_costas);
FT8     ft8(&ft8channel, kProxyXSubPort, kWsDictPort);
WaterfallCuda waterfall(magblock.getRing(), wf_bin_start, wf_bin_end,
                        WaterfallCuda::DEFAULT_OUT_BINS, kWsDictPort, wf_floor, wf_ceil);

// JS8 Normal (--js8): same normal ring
JS8Cuda<200> js8channel(magblock.getRing(), min_score, kProxyXSubPort,
                        js8_gpu_scan, kNormalTimeOsr, kNormalFreqOsr, kNormalCapBlks,
                        "JS8", legacy_costas);
JS8 js8(&js8channel, kProxyXSubPort, kNormalSymPer, kNormalCycleSec, …);

// JS8 Fast (--js8-fast): dedicated fast ring
MagBlock<128> magblock_fast(&buf, kFastRfftLen, kFastTimeOsr, kFastFreqOsr, 0);
JS8Cuda<128>  js8fast_channel(magblock_fast.getRing(), …);
JS8           js8fast(&js8fast_channel, …);

// JS8 Turbo (--js8-turbo): dedicated turbo ring
MagBlock<128> magblock_turbo(&buf, kTurboRfftLen, kTurboTimeOsr, kTurboFreqOsr, 0);
JS8Cuda<128>  js8turbo_channel(magblock_turbo.getRing(), …);
JS8           js8turbo(&js8turbo_channel, …);

// JS8 Ultra (--js8-ultra): dedicated ultra ring
MagBlock<128> magblock_ultra(&buf, kUltraRfftLen, kUltraTimeOsr, kUltraFreqOsr, 0);
JS8Cuda<128>  js8ultra_channel(magblock_ultra.getRing(), …);
JS8           js8ultra(&js8ultra_channel, …);
```

## Spectral noise-floor normalization (SpectrumNorm)

Applied every 64 frames in `HFChannelizer::doCopy()` — after audio extraction, before
composite assembly → IFFT. Operates on per-band slices of `fftData_d`.

**Asymmetric EMA** tracks the noise floor per composite bin:
- `α_down = 0.10` (warm) — signals fade quickly, track the drop
- `α_up = 0.005` (warm) — genuine noise floor rises are rare; ~30 min time constant
- 5-frame warm-up uses `α_down=0.30, α_up=0.10` to seed the estimate

**Legendre polynomial fit** (degree 3) on `log10(EMA)` per band → per-bin gains
that flatten intra-band tilt and RFI bumps. Legendre basis (vs. monomial Vandermonde)
gives well-conditioned normal equations across the band.

**Cross-band leveling** prevents active bands from appearing brighter than quiet ones:
geometric mean of the 11 band-center EMA values provides a global reference; a scalar
multiplied into each band's gain array brings all bands to the same power level.

## Continuous scan and epoch windows

FT8 and JS8 use **continuous scanning** — the GPU Costas scanner fires every 6 ring
blocks. Each signal is seen by ~15 overlapping windows per 15-second epoch, so every
transmission is decoded regardless of epoch phase alignment.

JS8 modes use independent MagBlock rings with different FFT sizes:

| Mode | FFT size | Hz/bin | ms/block | cap_blocks | Cycle |
|------|----------|--------|----------|------------|-------|
| FT8 / JS8 Normal | 65536 | 6.25 | 160 | 108 blks | 15 s |
| JS8 Fast | 40960 | 10.0 | 100 | 108 blks | 10 s |
| JS8 Slow | 131072 | 3.125 | 320 | 108 blks | 30 s |
| JS8 Turbo | 20480 | 20.0 | 50 | 108 blks | 6 s |
| JS8 Ultra | 13107 | ~31.3 | ~32 | 108 blks | 4 s |

## ZMQ bus

All external decode outputs publish to a ZMQ XPUB/XSUB proxy:

```
Producers (connect to XSUB :5599)    Consumers (connect to XPUB :5600)
  FT8         → "ft8/decode"  JSON     psk_uploader.py   (ft8, js8 → PSKreporter)
  JS8         → "js8/decode"  JSON     any subscriber     (custom logging, APRS, etc.)
```

Each producer owns its ZMQ PUB socket. No callbacks from consumers back to upstream blocks.

## wsdict — WebSocket key-value bus

The browser dashboard and audio control use a Rust WebSocket dict server (`wsserver/`)
on port 8765. Any connected client (browser or C++) can `set`, `get`, `subscribe`, or
`delete` keys. C++ uses `WsDictClient` (header-only, `wsdict.h`).

```
Key                              Producer            Consumer
granolasdr:waterfall:N           WaterfallCuda       browser (canvas)
granolasdr:ft8:heard:CALL        gm::hf::FT8         browser (decode list, band table)
granolasdr:js8:heard:CALL        gm::hf::JS8         browser (decode list, band table)
granolasdr:audio:status:N        HFChannelizer       browser (audio panel display)
granolasdr:audio:cmd             browser             HFChannelizer (retune sink)
```

The browser subscribes with pattern `granolasdr:*` on connect and receives all live
keys immediately (server pushes current values). TTL keys (`ft8:heard:*`, `js8:heard:*`)
expire after 900 s; the browser removes them from the decode list on `key_expired`.
