# granolasdr

A wideband HF FT8 decoder for the RX888 SDR. Simultaneously monitors all major HF amateur bands (160m–10m) for FT8 signals using CUDA-accelerated DSP, and uploads decoded stations to [PSKReporter](https://pskreporter.info).

## How it works

```
RX888 SDR (140 MS/s real)
  └─ HFChannelizer (CUDA)
       Wideband R2C FFT (1M pts) → selects 10 HF bands → composite IFFT
       Output: 16,384 complex samples/slot at 4.375 MS/s
  └─ FT8Cuda (CUDA)
       698,880-pt C2C FFT at 4 time × 4 freq offsets (16 FFTs/block)
       uint8_t waterfall magnitude → shared buffer
  └─ FT8 (CPU, ft8_lib)
       Candidate search → LDPC decode → callsign extraction
       Publishes JSON on tcp://*:5580 (ZMQ PUB)
  └─ psk_uploader.py (Python)
       Buffers decoded reports → uploads to PSKReporter every 5 min
```

Decoded bands: **160m, 80m, 60m, 40m, 30m, 20m, 17m, 15m, 12m, 10m**

## Prerequisites

### Hardware

- [RX888 MkII](https://github.com/RXToolsRX888/RX888) or compatible
- NVIDIA GPU (tested on RTX 5060 with 8 GB VRAM; `CMAKE_CUDA_ARCHITECTURES` defaults to 120)
- 16 GB system RAM (waterfall ring ~2.2 GB + decode slots ~2.2 GB; physical usage ~10 GB during operation)

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

## Project structure

```
gm/
  cuda/
    HFChannelizer.cc   — CUDA polyphase channelizer; maps wideband bins to 10 HF bands
    FT8Cuda.cc         — CUDA FT8 waterfall; 4×4 time/freq oversampled C2C FFT
    HostCuda.cu        — CUDA kernels (magnitude, copy)
  hf/
    ft8.cc             — FT8 decoder thread; frequency mapping; ZMQ publisher
    ft8_capture.h      — FT8_TIME_OSR and capture constants
  rx888/               — RX888 USB driver interface
  zmqcode/             — ZMQ server/pub utilities
ft8_lib/               — FT8 codec (prebuilt libft8.a)
psk_uploader.py        — PSKReporter IPFIX uploader
compare_psk.py         — Decoder vs PSKReporter comparison tool
check_uploads.py       — Verify PSKReporter accepted your uploads
calc_rf.py             — Derive kBandMap from HFChannelizer bin table
```

## License

MIT — see [LICENSE](LICENSE). Includes [ft8_lib](ft8_lib/LICENSE) by Kārlis Goba, also MIT.
