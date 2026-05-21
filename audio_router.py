#!/usr/bin/env python3
"""
audio_router.py — subscribe to granolasdr 48 kHz audio ZMQ streams and route
each sink to a PulseAudio null sink (virtual soundcard).

Usage:
    python3 audio_router.py [--host HOST] [--sinks N] [--create-sinks] [--control URL]

--host HOST         ZMQ publisher host (default: localhost)
--sinks N           Number of sinks to route (default: 4, max 4)
--create-sinks      Run pactl to create null sinks before starting
--control URL       Control server base URL (default: http://localhost:8080)

Frame format (per sink, 200 Hz):
    4 bytes  sink_id   uint32
    4 bytes  seq       uint32  (monotonically increasing, wraps at 2^32)
    960 bytes          240 × float32 PCM at 48 kHz

Sinks and ports:
    sink 0 → 5581    sink 1 → 5582    sink 2 → 5583    sink 3 → 5584

REST control (requires requests library):
    tune(sink, freq_hz)       — retune a sink
    status()                  — show current sink frequencies
"""

import argparse
import struct
import subprocess
import sys
import threading
import time
import zmq

NUM_SINKS = 4
BASE_PORT = 5581
SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = 240
FRAME_HEADER = struct.Struct("<II")    # sink_id, seq
FRAME_SAMPLES = struct.Struct(f"<{SAMPLES_PER_FRAME}f")
FRAME_BYTES = FRAME_HEADER.size + FRAME_SAMPLES.size  # 968


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


def sink_router(host, sink_idx, stats_lock, stats):
    port = BASE_PORT + sink_idx
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVHWM, 400)  # ~2 sec buffer at 200 Hz

    try:
        sink_proc = open_sink(sink_idx)
    except Exception as e:
        print(f"[sink{sink_idx}] failed to open sink: {e}", file=sys.stderr)
        return

    prev_seq = None
    frame_count = 0
    drop_count = 0

    print(f"[sink{sink_idx}] routing port {port} → granola-sink{sink_idx}")

    try:
        while True:
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue

            if len(raw) != FRAME_BYTES:
                continue

            _sink_id, seq = FRAME_HEADER.unpack_from(raw, 0)
            samples = FRAME_SAMPLES.unpack_from(raw, FRAME_HEADER.size)

            if prev_seq is not None:
                gap = (seq - prev_seq - 1) & 0xFFFFFFFF
                if gap:
                    drop_count += gap
            prev_seq = seq
            frame_count += 1

            pcm_bytes = struct.pack(f"<{SAMPLES_PER_FRAME}f", *samples)
            try:
                sink_proc.stdin.write(pcm_bytes)
                sink_proc.stdin.flush()
            except BrokenPipeError:
                print(f"[sink{sink_idx}] sink pipe closed, restarting", file=sys.stderr)
                sink_proc = open_sink(sink_idx)

            if frame_count % 2000 == 0:
                with stats_lock:
                    stats[sink_idx] = {"frames": frame_count, "drops": drop_count}
                print(f"[sink{sink_idx}] {frame_count} frames, {drop_count} dropped "
                      f"({100*drop_count/max(frame_count+drop_count,1):.1f}% loss)")

    except KeyboardInterrupt:
        pass
    finally:
        sink_proc.stdin.close()
        sink_proc.wait()
        sock.close()
        ctx.term()


def main():
    ap = argparse.ArgumentParser(description="Route granolasdr ZMQ audio to PulseAudio")
    ap.add_argument("--host", default="localhost", help="ZMQ publisher host")
    ap.add_argument("--sinks", type=int, default=NUM_SINKS,
                    help=f"Number of sinks to route (default: {NUM_SINKS})")
    ap.add_argument("--create-sinks", action="store_true",
                    help="Create PulseAudio null sinks via pactl")
    ap.add_argument("--control", default="http://localhost:8080",
                    help="Control server base URL")
    args = ap.parse_args()

    num = min(max(args.sinks, 1), NUM_SINKS)

    if args.create_sinks:
        create_null_sinks(num)
        time.sleep(0.5)

    stats_lock = threading.Lock()
    stats = {}

    threads = []
    for i in range(num):
        t = threading.Thread(target=sink_router, args=(args.host, i, stats_lock, stats),
                             daemon=True, name=f"router-sink{i}")
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
