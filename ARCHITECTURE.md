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
magnitude ring feeds both FT8 and JS8 scanners simultaneously.

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
directions. This means **any edge in the pipeline can be saved and replayed** — IQ
samples today, magnitude ring tomorrow — without changing the surrounding blocks.

```
Playback:   BufferFile<complex<float>>(path)  →  BufferPosition  →  MagBlock  → …
Record:     HFChannelizer  →  BufferPosition  →  BufferFile<complex<float>>(path)
```

## Pipeline topology (current)

```
rx888 (USB/PCIe)
  └─ BufferPosition<int16_t>          (pinned, CPU semaphore)
       └─ HFChannelizer               RFFT + polyphase channelise → HF band
            └─ BufferPosition<complex<float>>
                 │
                 ├─ [optional] BufferFile<complex<float>>  (record to file)
                 │
                 └─ MagBlock          RFFT + |·|² + uint8 decimation
                      └─ DeviceRingBuffer<uint8_t, 200>    (shared mag ring)
                           ├─ FT8Cuda  continuous Costas scan + LLR → FT8
                           │    └─ gm::hf::FT8   CPU BP-LDPC + 15s window + ZMQ
                           └─ JS8Cuda  continuous Costas scan + LLR → JS8
                                └─ gm::hf::JS8   CPU BP-LDPC + ZMQ

Playback path (substitutes for the hardware source):
  BufferFile<complex<float>>(path)
    └─ BufferPosition<complex<float>>  →  MagBlock  →  (same as above)
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
or callback registration are visible at the orchestration level.

```cpp
// HFRx.cc runPipeline — each line constructs the next stage downstream
MagBlock    magblock(&buf,             kProxyXSubPort);
FT8Cuda     ft8channel(magblock.getRing(), min_score, kProxyXSubPort);
FT8         ft8(&ft8channel,           kProxyXSubPort);   // wires callback internally
JS8Cuda     js8channel(magblock.getRing(), min_score, kProxyXSubPort);
JS8         js8(&js8channel,           kProxyXSubPort);   // wires callback internally
```

Callbacks on `this` (a node registering itself as its upstream's consumer) are
fine and hidden inside constructors. Passing callbacks through the orchestrator
layer destroys the graph view and is avoided.

## Continuous scan and 15-second window

FT8 and JS8 use **continuous scanning** only — the GPU Costas scanner fires every
6 ring blocks (~1 s) with a 106-block window. Each signal is seen by ~5 overlapping
windows, so every transmission is decoded regardless of epoch phase alignment.

On the CPU side, `gm::hf::FT8` accumulates decoded callsigns into a 15-second
window buffer, keeping the best SNR per callsign. At each window boundary it:
1. Prints `[FT8] 15s window: N unique callsigns`
2. Publishes each best-SNR entry to ZMQ once

This gives PSKreporter a clean, deduped batch every 15 s. The PSK uploader
(`psk_uploader.py`) further deduplicates by (callsign, band) before uploading
every 5 minutes.

The FT8 protocol callsign hashtable (22-bit hash resolution) is separate from
the dedup window — it is a protocol necessity for decoding compressed messages
and is maintained by the continuous decode path.

## ZMQ bus

All external outputs publish to a ZMQ XPUB/XSUB proxy:

```
Producers (connect to XSUB :5599)          Consumers (connect to XPUB :5600)
  FT8         → "ft8/decode"  JSON           psk_uploader.py   (ft8, js8 → PSKreporter)
  JS8         → "js8/decode"  JSON           aprs_uploader.py  (ft8, js8 → APRS-IS)
  FT8Cuda     → "ft8/timing"  JSON           ws_bridge.py      (all → WebSocket :8765)
  MagBlock    → "waterfall"   binary 2048B   browser dashboard
```

Each producer owns its ZMQ PUB socket. There are no callbacks from consumers
back to upstream blocks.
