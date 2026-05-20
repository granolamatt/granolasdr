# granolasdr

**[Documentation](https://granolamatt.github.io/granolasdr)**

A wideband HF FT8 decoder for the RX888 SDR. Simultaneously monitors all major HF amateur bands (160m–10m) for FT8 signals using CUDA-accelerated DSP, and uploads decoded stations to [PSKReporter](https://pskreporter.info).

## How it works

```
RX888 SDR (140 MS/s real)
  └─ HFChannelizer (CUDA)
       Wideband R2C FFT (1M pts) → selects 10 HF bands → composite IFFT
       Output: 16,384 complex samples/slot at 4.375 MS/s
  └─ FT8Cuda (CUDA)
       698,880-pt C2C FFT at 4 time × 4 freq offsets (16 FFTs/block)
       uint8_t magnitude → 200-block GPU ring buffer (2.2 GB VRAM)
       Costas sync scan → candidate list (fo, to, ts, fs)
       Soft symbol kernel → float32 LLRs for each candidate (~70 MB D2H)
  └─ FT8 (CPU, ft8_lib)
       LDPC BP decode → callsign extraction (no CPU candidate search)
       Publishes JSON on tcp://*:5580 (ZMQ PUB)
  └─ psk_uploader.py (Python)
       Buffers decoded reports → uploads to PSKReporter every 5 min
```

Decoded bands: **160m, 80m, 60m, 40m, 30m, 20m, 17m, 15m, 12m, 10m**

## Prerequisites

### Hardware

- [RX888 MkII](https://github.com/RXToolsRX888/RX888) or compatible
- NVIDIA GPU (tested on RTX 5060 with 8 GB VRAM; `CMAKE_CUDA_ARCHITECTURES` defaults to 120)
- NVIDIA GPU with ≥ 8 GB VRAM (GPU ring buffer ~2.2 GB + LLR buffers ~140 MB)
- 16 GB system RAM recommended (~70 MB pinned for LLR staging; rest for OS/driver overhead)

### System dependencies

```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libusb-1.0-0-dev libzmq3-dev \
                 python3 python3-pip

# Python packages
pip3 install pyzmq
```

### CUDA toolkit

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (CUDA 12+ recommended).
`nvcc` and `libcufft` must be in `/opt/cuda` (the default Arch Linux path).
For other distributions adjust the `include_directories` and `LIBLOC` paths in `CMakeLists.txt`.

### ExtIO_sddc (RX888 driver)

This project requires a specific fork of ExtIO_sddc that provides the `sddc` shared library:

```bash
git clone https://github.com/granolamatt/ExtIO_sddc
cd ExtIO_sddc
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

The build installs `libsddc.so`. If it lands somewhere other than a standard library path, set `LD_LIBRARY_PATH` accordingly.

### ft8_lib

The `ft8_lib/` directory contains a modified copy of [kgoba/ft8_lib](https://github.com/kgoba/ft8_lib) with granolasdr-specific changes:
- `ftx_find_candidates_range()` for parallel frequency-sliced candidate search
- `int32_t` offsets in `ftx_candidate_t` for wider waterfalls
- `monitor.c` stripped to `monitor_reset()` (GPU handles FFT/windowing)

A prebuilt `ft8_lib/libft8.a` is included. To rebuild from source:

```bash
cd ft8_lib
make
```

## Build

```bash
git clone https://github.com/granolamatt/granolasdr
cd granolasdr
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The main binary is `build/hf_rx`.

To target a specific GPU architecture (e.g. Ada Lovelace / sm_89):

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 ..
```

## Running

### Start the decoder

```bash
./build/hf_rx
```

This starts the RX888 capture, CUDA processing pipeline, and FT8 decoder. Decoded messages are printed to stdout:

```
DECODED: W1AW K1ABC FN42 time_offset=0.450s freq=14074150.3Hz snr=-8.0 unix=1716000015
```

A ZMQ PUB socket opens on `tcp://*:5580` (all interfaces) and publishes each decode as JSON. If the host is publicly reachable, firewall port 5580 — no authentication is required to subscribe.

```json
{"call":"K1ABC","freq":14074150,"snr":-8.0,"unix":1716000015,"offset":0.45}
```

### Upload to PSKReporter

```bash
python3 psk_uploader.py --call W1AW --grid DM78
```

| Option | Default | Description |
|--------|---------|-------------|
| `--call` | required | Your callsign (receiver) |
| `--grid` | required | Your Maidenhead grid square |
| `--rig` | `rx888` | Antenna/rig description |
| `--port` | `5580` | ZMQ port |
| `--test` | off | Send to PSKReporter packet analyzer (port 14739) |
| `--send-test-packet` | off | Send one dummy packet and exit |

Uploads are batched and sent every 5 minutes as required by PSKReporter.
Verify your reports appeared: https://pskreporter.info/analyze.html

## Python tools

Install dependencies:

```bash
pip3 install -r requirements.txt
```

### `psk_uploader.py`

Subscribes to the ZMQ PUB socket and batches decoded messages into PSKReporter IPFIX UDP packets.

### `check_uploads.py`

Queries PSKReporter to confirm your reports were accepted:

```bash
python3 check_uploads.py --call W1AW --period 1800
```

### `compare_psk.py`

Compares your decoder output against what PSKReporter stations near you heard, to measure missed signals:

```bash
# Capture decoder output first:
./build/hf_rx 2>&1 | tee messages.txt

# Compare:
python3 compare_psk.py messages.txt --grid DM78 --max-dist 1500
```

Output shows:
- Signals confirmed in both (with frequency delta and SNR comparison)
- Signals you decoded that PSKReporter missed
- Signals PSKReporter heard nearby that you missed (sorted by sender distance — nearest first)

### `compare_corpus.py` — JTDX / WSJT-X comparison

The `--jtdx` flag enables corpus capture: every 15-second FT8 period, a 16-bit PCM WAV of
the 20m band is saved to `ft8_corpus/`. This lets you run JTDX or WSJT-X on the same audio
granolasdr processed and compare which signals each decoder found.

**WAV file details**

| Parameter | Value |
|-----------|-------|
| Band | 20m FT8 (14.074 MHz dial) |
| Sample rate | ~12,019 Hz (= 4,375,000 × 1920 / 698,880) |
| Format | 16-bit signed PCM, mono |
| Duration | 106 symbol-blocks ≈ 17 s per file |
| Filename | `ft8_corpus/20m_YYYYMMDD_HHMMSS.wav` |

**Workflow**

```bash
# 1. Run with corpus capture
./build/hf_rx --jtdx 2>&1 | tee granola_decodes.txt

# 2. Subscribe to ZMQ and capture JSON decodes
python3 -c "
import zmq, sys
ctx = zmq.Context()
s = ctx.socket(zmq.SUB)
s.connect('tcp://localhost:5580')
s.setsockopt(zmq.SUBSCRIBE, b'')
while True:
    print(s.recv_string(), flush=True)
" > granola_log.jsonl &

# 3. Decode the corpus WAV files with WSJT-X or JTDX
#    (run jt9 or wsjtx --decode on each file, redirect to ALL.TXT)
for f in ft8_corpus/*.wav; do
    jt9 --ft8 -d 3 "$f" >> ALL.TXT
done

# 4. Compare
python3 compare_corpus.py granola_log.jsonl ALL.TXT --date 20240518
```

`compare_corpus.py` groups decodes by 15-second FT8 epoch and call sign, then reports:
- **BOTH** — call decoded by both (shown with `--show-both`)
- **GRANOLA-ONLY** — call only granolasdr decoded
- **REF-ONLY** — call only the reference decoder found
- Summary: recall percentage (what fraction of reference's signals granolasdr caught)

## Project structure

```
gm/
  cuda/
    HFChannelizer.cc      — CUDA polyphase channelizer; maps wideband bins to 10 HF bands
    FT8Cuda.cc            — 4×4 oversampled FFT, GPU mag ring, scan dispatch, D2H
    FT8ScanCuda.cu        — GPU Costas sync scan; outputs candidate list
    FT8SoftCuda.cu        — GPU soft symbol kernel; outputs float32 LLRs per candidate
    HostCuda.cu           — CUDA kernels (magnitude, frequency shift)
  hf/
    ft8.cc                — FT8 decoder thread; LDPC decode; frequency mapping; ZMQ publisher
    ft8_capture.h         — FT8_TIME_OSR, FT8_FREQ_OSR, capture window constants
  rx888/                  — RX888 USB driver interface
  zmqcode/                — ZMQ server/pub utilities
ft8_lib/                  — FT8 codec (prebuilt libft8.a)
psk_uploader.py        — PSKReporter IPFIX uploader
compare_psk.py         — Decoder vs PSKReporter comparison tool
compare_corpus.py      — Granolasdr vs JTDX/WSJT-X decode comparison
check_uploads.py       — Verify PSKReporter accepted your uploads
calc_rf.py             — Derive kBandMap from HFChannelizer bin table
```

## License

MIT — see [LICENSE](LICENSE). Includes [ft8_lib](ft8_lib/LICENSE) by Kārlis Goba, also MIT.
