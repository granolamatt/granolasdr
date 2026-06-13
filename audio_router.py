#!/usr/bin/env python3
"""
audio_router.py — connect to granolasdr TCI WebSocket and route each sink
to a PulseAudio null sink (virtual soundcard).

Usage:
    python3 audio_router.py [--host HOST] [--port PORT] [--sinks N]
                            [--create-sinks] [--delete-sinks]

TCI binary frames: 64-byte header (16 × uint32 LE) followed by float32
stereo-interleaved PCM (L=R mono). We extract the left channel and pipe
float32le mono to a per-sink pacat process.

Header layout (WSJT-X Data_Stream):
    [0]  receiver    trx index (0–3)
    [1]  sampleRate  48000
    [2]  format      3 = float32
    [5]  length      stereo float count (= sample_count × 2)
    [6]  type        1 = RX_AUDIO_STREAM
"""

import argparse
import array
import struct
import subprocess
import sys
import time
import websocket  # pip install websocket-client

NUM_SINKS   = 4
TCI_PORT    = 40001
SAMPLE_RATE = 48000

TCI_HEADER      = struct.Struct("<16I")   # 16 × uint32 LE = 64 bytes
HDR_RECEIVER    = 0    # trx index
HDR_LENGTH      = 5    # stereo float count
HDR_TYPE        = 6    # 1 = RX_AUDIO_STREAM
RX_AUDIO_STREAM = 1


def delete_null_sinks():
    try:
        out = subprocess.check_output(
            ["pactl", "list", "modules", "short"],
            stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError as e:
        print(f"pactl list failed: {e.output.decode().strip()}", file=sys.stderr)
        return

    deleted = 0
    for line in out.splitlines():
        parts = line.split('\t')
        if len(parts) < 2 or 'module-null-sink' not in parts[1]:
            continue
        args = parts[2] if len(parts) > 2 else ''
        if 'sink_name=granola-sink' not in args:
            continue
        module_id = parts[0].strip()
        sink_name = next(
            (a.split('=', 1)[1] for a in args.split() if a.startswith('sink_name=')),
            module_id,
        )
        try:
            subprocess.check_call(["pactl", "unload-module", module_id],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Removed {sink_name} (module {module_id})")
            deleted += 1
        except subprocess.CalledProcessError:
            print(f"Failed to unload module {module_id} ({sink_name})", file=sys.stderr)

    print(f"Removed {deleted} sink(s)." if deleted else "No granola-sink modules found.")


def create_null_sinks(num_sinks):
    for i in range(num_sinks):
        sink = f"granola-sink{i}"
        cmd = [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={sink}",
            f"sink_properties=device.description=GranolaSDR-Sink{i}",
            f"rate={SAMPLE_RATE}",
            "format=float32le",
            "channels=1",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
            print(f"[sink{i}] null sink created: module {out}")
        except subprocess.CalledProcessError as e:
            print(f"[sink{i}] pactl failed: {e.output.decode().strip()}", file=sys.stderr)


def open_sink(sink_idx):
    """Open a pacat process to write float32le mono PCM to a PulseAudio null sink."""
    cmd = [
        "pacat",
        "--playback",
        f"--device=granola-sink{sink_idx}",
        f"--rate={SAMPLE_RATE}",
        "--format=float32le",
        "--channels=1",
        "--latency-msec=200",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def tune(sink, freq_hz, label=None, control_url="http://localhost:8080"):
    """Retune a sink to freq_hz Hz via the REST API."""
    try:
        import requests
    except ImportError:
        print("pip install requests to use tune()", file=sys.stderr)
        return None
    body = {"sink": sink, "freq_hz": freq_hz}
    if label:
        body["label"] = label
    r = requests.post(f"{control_url}/api/tune", json=body, timeout=2)
    r.raise_for_status()
    return r.json()


def status(control_url="http://localhost:8080"):
    """Print current sink frequencies from the REST API."""
    try:
        import requests
    except ImportError:
        print("pip install requests to use status()", file=sys.stderr)
        return None
    r = requests.get(f"{control_url}/api/status", timeout=2)
    r.raise_for_status()
    data = r.json()
    for s in data["sinks"]:
        print(f"  sink {s['id']}: {s['freq_hz']/1e6:.4f} MHz  [{s['label']}]")
    return data


def run_router(host, port, num_sinks):
    """Connect to TCI WebSocket and route audio frames to pacat processes."""
    url = f"ws://{host}:{port}"

    procs = [None] * num_sinks
    frame_counts = [0] * num_sinks

    def reopen_sink(i):
        if procs[i] is not None:
            try:
                procs[i].stdin.close()
                procs[i].wait()
            except Exception:
                pass
        try:
            procs[i] = open_sink(i)
            print(f"[sink{i}] opened granola-sink{i}")
        except Exception as e:
            print(f"[sink{i}] failed to open sink: {e}", file=sys.stderr)
            procs[i] = None

    for i in range(num_sinks):
        reopen_sink(i)

    while True:
        try:
            ws = websocket.create_connection(url, timeout=10)
            print(f"[TCI] connected to {url}")

            # Subscribe all requested sinks in one command string.
            ws.send("".join(f"audio_start:{i};" for i in range(num_sinks)))

            while True:
                opcode, data = ws.recv_data()
                if opcode != websocket.ABNF.OPCODE_BINARY:
                    continue  # skip text echoes / smeter

                if len(data) < 64:
                    continue
                hdr = TCI_HEADER.unpack_from(data, 0)
                if hdr[HDR_TYPE] != RX_AUDIO_STREAM:
                    continue
                trx = hdr[HDR_RECEIVER]
                if trx >= num_sinks:
                    continue
                float_count = hdr[HDR_LENGTH]
                expected = 64 + float_count * 4
                if len(data) < expected:
                    continue

                # Stereo-interleaved L,R,L,R… — take left channel every other float.
                stereo = array.array('f', data[64:expected])
                mono = stereo[0::2]

                if procs[trx] is None:
                    continue
                try:
                    procs[trx].stdin.write(mono.tobytes())
                    procs[trx].stdin.flush()
                except BrokenPipeError:
                    print(f"[sink{trx}] pipe closed, restarting", file=sys.stderr)
                    reopen_sink(trx)

                frame_counts[trx] += 1
                if frame_counts[trx] % 2000 == 0:
                    print(f"[sink{trx}] {frame_counts[trx]} frames routed")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[TCI] error: {e} — reconnecting in 2s", file=sys.stderr)
            try:
                ws.close()
            except Exception:
                pass
            time.sleep(2)

    for i in range(num_sinks):
        if procs[i] is not None:
            try:
                procs[i].stdin.close()
                procs[i].wait()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Route granolasdr TCI audio to PulseAudio")
    ap.add_argument("--host", default="localhost", help="TCI WebSocket host")
    ap.add_argument("--port", type=int, default=TCI_PORT,
                    help=f"TCI WebSocket port (default: {TCI_PORT})")
    ap.add_argument("--sinks", type=int, default=NUM_SINKS,
                    help=f"Number of sinks to route (default: {NUM_SINKS})")
    ap.add_argument("--create-sinks", action="store_true",
                    help="Create PulseAudio null sinks via pactl")
    ap.add_argument("--delete-sinks", action="store_true",
                    help="Remove all granola-sink PulseAudio modules and exit")
    ap.add_argument("--control", default="http://localhost:8080",
                    help="Control server base URL (for tune/status helpers)")
    args = ap.parse_args()

    if args.delete_sinks:
        delete_null_sinks()
        sys.exit(0)

    num = min(max(args.sinks, 1), NUM_SINKS)

    if args.create_sinks:
        create_null_sinks(num)
        time.sleep(0.5)

    run_router(args.host, args.port, num)


if __name__ == "__main__":
    main()
