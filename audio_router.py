#!/usr/bin/env python3
"""
audio_router.py — subscribe to granolasdr 12 kHz audio ZMQ streams and route
each band to a PulseAudio null sink (virtual soundcard).

Usage:
    python3 audio_router.py [--host HOST] [--bands BAND1,BAND2,...] [--create-sinks]

--host HOST         ZMQ publisher host (default: localhost)
--bands BAND,...    Comma-separated bands to route; default: all 10
--create-sinks      Run pactl to create null sinks before starting

Frame format (per band, 200 Hz):
    4 bytes  band_id   uint32
    4 bytes  seq       uint32  (monotonically increasing, wraps at 2^32)
    240 bytes          60 × float32 PCM at 12 kHz

Bands and ports:
    160m  5581    80m  5582    60m  5583    40m  5584    30m  5585
     20m  5586    17m  5587    15m  5588    12m  5589    10m  5590
"""

import argparse
import struct
import subprocess
import sys
import threading
import time
import zmq

BANDS = ["160m", "80m", "60m", "40m", "30m", "20m", "17m", "15m", "12m", "10m"]
BASE_PORT = 5581
SAMPLE_RATE = 12000
SAMPLES_PER_FRAME = 60
FRAME_HEADER = struct.Struct("<II")    # band_id, seq
FRAME_SAMPLES = struct.Struct(f"<{SAMPLES_PER_FRAME}f")
FRAME_BYTES = FRAME_HEADER.size + FRAME_SAMPLES.size  # 248


def create_null_sinks(bands):
    for band in bands:
        sink = f"granola-{band}"
        cmd = [
            "pactl", "load-module", "module-null-sink",
            f"sink_name={sink}",
            f"sink_properties=device.description=GranolaSDR-{band}",
            f"rate={SAMPLE_RATE}",
            "format=float32le",
            "channels=1",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
            print(f"[{band}] null sink created: module {out}")
        except subprocess.CalledProcessError as e:
            print(f"[{band}] pactl failed: {e.output.decode().strip()}", file=sys.stderr)


def open_sink(band):
    """Open /dev/stdin-like write to the PulseAudio sink via parec/paplay workaround.
    Returns a Popen that accepts raw float32le on stdin."""
    sink = f"granola-{band}.monitor"
    # Use pacat to write float32le mono to the null sink's input
    cmd = [
        "pacat",
        "--playback",
        f"--device=granola-{band}",
        f"--rate={SAMPLE_RATE}",
        "--format=float32le",
        "--channels=1",
        "--latency-msec=200",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def band_router(host, band_idx, band_name, stats_lock, stats):
    port = BASE_PORT + band_idx
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVHWM, 400)  # ~2 sec buffer at 200 Hz

    try:
        sink_proc = open_sink(band_name)
    except Exception as e:
        print(f"[{band_name}] failed to open sink: {e}", file=sys.stderr)
        return

    prev_seq = None
    frame_count = 0
    drop_count = 0

    print(f"[{band_name}] routing port {port} → granola-{band_name}")

    try:
        while True:
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue

            if len(raw) != FRAME_BYTES:
                continue

            band_id, seq = FRAME_HEADER.unpack_from(raw, 0)
            samples = FRAME_SAMPLES.unpack_from(raw, FRAME_HEADER.size)

            if prev_seq is not None:
                gap = (seq - prev_seq - 1) & 0xFFFFFFFF
                if gap:
                    drop_count += gap
            prev_seq = seq
            frame_count += 1

            # Write raw float32 PCM to pacat stdin
            pcm_bytes = struct.pack(f"<{SAMPLES_PER_FRAME}f", *samples)
            try:
                sink_proc.stdin.write(pcm_bytes)
                sink_proc.stdin.flush()
            except BrokenPipeError:
                print(f"[{band_name}] sink pipe closed, restarting", file=sys.stderr)
                sink_proc = open_sink(band_name)

            if frame_count % 2000 == 0:
                with stats_lock:
                    stats[band_name] = {"frames": frame_count, "drops": drop_count}
                print(f"[{band_name}] {frame_count} frames, {drop_count} dropped "
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
    ap.add_argument("--bands", default=",".join(BANDS),
                    help="Comma-separated bands to route")
    ap.add_argument("--create-sinks", action="store_true",
                    help="Create PulseAudio null sinks via pactl")
    args = ap.parse_args()

    selected = [b.strip() for b in args.bands.split(",")]
    invalid = [b for b in selected if b not in BANDS]
    if invalid:
        print(f"Unknown bands: {invalid}. Valid: {BANDS}", file=sys.stderr)
        sys.exit(1)

    if args.create_sinks:
        create_null_sinks(selected)
        time.sleep(0.5)

    stats_lock = threading.Lock()
    stats = {}

    threads = []
    for band in selected:
        idx = BANDS.index(band)
        t = threading.Thread(target=band_router, args=(args.host, idx, band, stats_lock, stats),
                             daemon=True, name=f"router-{band}")
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
