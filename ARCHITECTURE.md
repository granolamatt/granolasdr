# granolasdr Architecture

## Core concept: GPU flow graph over epochs

granolasdr is built around a **typed flow graph** where signal processing blocks
connect through explicitly-typed buffers. Each block owns a CUDA stream and
declares its input and output buffer types. The runtime is the data itself —
when an upstream block writes a buffer and signals its completion event, the
downstream block's CUDA stream begins work without any CPU thread wakeup in
between.

This mirrors the structure of offline SDR tools (GNU Radio, liquid-dsp) but
targets online GPU execution: the "scheduler" is `cudaStreamWaitEvent`, not a
thread pool.

## Buffer kinds

Two memory kinds, two synchronization primitives:

```
HostBuffer<T>          pinned host memory   CPU semaphore / condition_variable
DeviceBuffer<T>        device memory        cudaEvent_t
DeviceRingBuffer<T,N>  device circular ring cudaEvent_t + atomic write_idx
```

`HostBuffer` is appropriate for I/O boundaries (USB, file) where the CPU must
participate. `DeviceBuffer` and `DeviceRingBuffer` keep data on the GPU across
processing stages, avoiding PCIe round-trips.

A `DeviceRingBuffer<T, N>` supports **multiple independent readers** — each
reader tracks its own read cursor against the shared atomic `write_idx`. This
is how one magnitude ring feeds both FT8 and JS8 scanners.

## Pipeline topology (current → target)

```
                    ┌─────────────────────────────────────────┐
                    │  current (mixed abstractions)           │
                    └─────────────────────────────────────────┘

rx888 (USB/PCIe)
  └─ HostBuffer<int16_t>          (pinned, CPU semaphore)
       └─ HFChannelizer            RFFT + polyphase channelise → 11 HF bands
            └─ HostBuffer<complex<float>>   (pinned, CPU semaphore)
                 └─ FT8Cuda        RFFT + |·|² + mag ring + Costas scan
                      │            (magnitude ring is ad-hoc inside FT8Cuda)
                      ├─ JS8Cuda   reads FT8Cuda's ring via raw pointer accessors
                      └─ gm::hf::FT8 / gm::hf::JS8   CPU decode + ZMQ publish


                    ┌─────────────────────────────────────────┐
                    │  target (explicit typed flow graph)     │
                    └─────────────────────────────────────────┘

rx888
  └─ HostBuffer<int16_t>
       └─ HFChannelizer
            └─ DeviceBuffer<complex<float>>
                 └─ MagBlock              RFFT + |·|² + uint8 decimation
                      └─ DeviceRingBuffer<uint8_t, 200>
                           ├─ FT8Scanner  Costas {3,1,4,0,6,5,2} scan + LDPC
                           └─ JS8Scanner  Costas {4,2,5,6,1,3,0} scan + LDPC
                                └─ each scanner → gm::hf::FT8 / JS8  CPU decode + ZMQ
```

## Block contract

Each processing block:

1. **Owns one `cudaStream_t`** — all GPU work for that block runs on this stream.
2. **Declares typed inputs** — constructor takes a const reference to the upstream buffer.
3. **Declares typed outputs** — exposes a buffer reference for downstream blocks.
4. **Signals completion** — records a `cudaEvent_t` on its stream after each write.
5. **Waits on upstream** — calls `cudaStreamWaitEvent(myStream, input.ready, 0)` before reading.
6. **Does not block the CPU** — GPU-to-GPU handoffs never stall a CPU thread.

CPU blocking (semaphores, `usleep`) is reserved for the HF channelizer input
boundary where the CPU must pace the USB/PCIe data rate.

## ZMQ bus

All external outputs (decoded messages, timing, waterfall pixels) publish to
a ZMQ XPUB/XSUB proxy:

```
Producers (connect to XSUB :5599)          Consumers (connect to XPUB :5600)
  FT8Scanner   → "ft8/decode"  JSON           psk_uploader.py   (ft8/decode, js8/decode)
  JS8Scanner   → "js8/decode"  JSON           ws_bridge.py      (all topics → WS :8765)
  FT8Scanner   → "ft8/timing"  JSON           browser dashboard (ws://host:8765)
  JS8Scanner   → "js8/timing"  JSON
  MagBlock     → "waterfall"   binary 2048B
```

Each producer owns its ZMQ PUB socket and publishes directly — there are no
callbacks from scanners back to upstream blocks.

## Epoch concept

FT8 and JS8 operate on 15-second epochs. The pipeline uses two scan strategies:

- **Epoch scan**: snapshot the full 15-second buffer at T+0 and decode once.
  Latency: up to 15 s. Used for corpus / reference decode.
- **Continuous scan**: stride through the ring every ~1 second, decoding partial
  epochs as they fill. Latency: ~1–3 s. Primary operational mode.

Both strategies read from the same `DeviceRingBuffer` — the epoch scan reads
a snapshot into a separate slot; the continuous scan reads directly from the ring.
