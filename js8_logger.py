#!/usr/bin/env python3
"""
js8_logger.py — persist all JS8 decoded messages (all modes) to a JSON-lines file.

Subscribes to the ZMQ XPUB proxy and appends every js8/decode message to a
rotating daily log file.  Each line is a self-contained JSON object.

Usage:
    python3 js8_logger.py
    python3 js8_logger.py --xpub 5600 --dir /var/log/granolasdr/js8
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import zmq

# Dedup window: same text at same frequency within this many seconds is a duplicate.
# Covers the longest JS8 period (Normal = ~15 s) with margin for cross-mode repeats.
DEDUP_WINDOW_SEC = 60
# Frequency bucket size in Hz — signals within this range are treated as the same.
FREQ_BUCKET_HZ = 100


def log_path(log_dir: str) -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(log_dir, f"js8_{date}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpub", type=int, default=5600, metavar="PORT",
                    help="ZMQ XPUB proxy port (default: 5600)")
    ap.add_argument("--dir", default=".", metavar="DIR",
                    help="Directory for log files (default: current directory)")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{args.xpub}")
    sub.setsockopt(zmq.SUBSCRIBE, b"js8/decode")

    print(f"[js8_logger] ZMQ tcp://localhost:{args.xpub} → {args.dir}/js8_YYYYMMDD.jsonl",
          flush=True)

    current_path = None
    log_file = None
    # dedup_cache: (text, freq_bucket) -> last unix time seen
    dedup_cache: dict[tuple, float] = {}

    while True:
        try:
            frames = sub.recv_multipart()
        except KeyboardInterrupt:
            break
        if len(frames) != 2:
            continue
        _, payload = frames
        try:
            msg = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        text = msg.get("text", "")
        freq = msg.get("freq", 0)
        unix = msg.get("unix", time.time())
        dedup_key = (text, int(freq) // FREQ_BUCKET_HZ)
        last_seen = dedup_cache.get(dedup_key)
        if last_seen is not None and unix - last_seen < DEDUP_WINDOW_SEC:
            continue
        dedup_cache[dedup_key] = unix
        # Evict stale entries to keep memory bounded
        if len(dedup_cache) > 10_000:
            cutoff = unix - DEDUP_WINDOW_SEC
            dedup_cache = {k: v for k, v in dedup_cache.items() if v >= cutoff}

        path = log_path(args.dir)
        if path != current_path:
            if log_file:
                log_file.close()
            log_file = open(path, "a", buffering=1)  # line-buffered
            current_path = path
            print(f"[js8_logger] logging to {path}", flush=True)

        line = json.dumps(msg, separators=(",", ":"))
        log_file.write(line + "\n")

        call = msg.get("call", "")
        mode = msg.get("mode", "?")
        snr  = msg.get("snr", 0)
        print(f"[{mode}] {call or '(no call)':12s}  {freq:10.0f} Hz  {snr:+5.1f} dB  {text}",
              flush=True)

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
