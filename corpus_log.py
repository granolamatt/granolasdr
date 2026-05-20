#!/usr/bin/env python3
"""
corpus_log.py — filter granolasdr ZMQ decodes to epochs that have saved WAV files.

Usage:
    python3 corpus_log.py GRANOLA_LOG [--corpus-dir DIR] [--band BAND] [--wsjt]

GRANOLA_LOG: JSON-lines file captured from the ZMQ PUB socket.
             Each line: {"call":"K1ABC","freq":14074150,"snr":-8.0,"unix":1716000015,"offset":0.45}

Without --wsjt: prints one section per WAV file for visual inspection.
With --wsjt:    outputs JTDX-compatible format for use with compare_corpus.py.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone


def wav_epoch(path):
    """Parse UTC epoch from filename: 20m_YYYYMMDD_HHMMSS.wav"""
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(path))
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def ft8_period_start(unix_t):
    """Round unix timestamp down to nearest 15-second FT8 period."""
    return int(unix_t) // 15 * 15


BAND_CONFIG = {
    "20m": {"freq_lo": 14070000, "freq_hi": 14080000, "dial": 14074000, "prefix": "20m"},
    "10m": {"freq_lo": 28070000, "freq_hi": 28080000, "dial": 28074000, "prefix": "10m"},
}


def main():
    ap = argparse.ArgumentParser(description="Filter granolasdr decodes to WAV file epochs")
    ap.add_argument("granola_log", help="granolasdr JSON-lines log")
    ap.add_argument("--corpus-dir", default="ft8_corpus", help="directory of WAV files (default: ft8_corpus)")
    ap.add_argument("--band", default="20m", choices=list(BAND_CONFIG),
                    help="HF band to filter (default: 20m)")
    ap.add_argument("--wsjt", action="store_true", help="output in JTDX-compatible format for compare_corpus.py")
    args = ap.parse_args()

    band = BAND_CONFIG[args.band]
    wav_prefix = band["prefix"] + "_"

    wav_files = sorted(glob.glob(os.path.join(args.corpus_dir, wav_prefix + "*.wav")))
    if not wav_files:
        print(f"No {args.band} WAV files found in {args.corpus_dir} (looking for {wav_prefix}*.wav)",
              file=sys.stderr)
        sys.exit(1)

    # Build set of signal-period starts that have a WAV file.
    # WAV filename timestamp = snapshot time ≈ end of the signal period.
    # Signal period start = ft8_period_start(snap_time).
    wav_by_signal_period = {}
    for path in wav_files:
        snap_t = wav_epoch(path)
        if snap_t is None:
            continue
        period = ft8_period_start(snap_t)
        wav_by_signal_period[period] = os.path.basename(path)

    # Granolasdr unix timestamp = decode completion time ≈ start of NEXT period.
    # Signal period = ft8_period_start(unix) - 15.
    decodes_by_signal_period = defaultdict(list)
    with open(args.granola_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            freq = d.get("freq", 0)
            if not (band["freq_lo"] <= freq <= band["freq_hi"]):
                continue
            signal_period = ft8_period_start(d.get("unix", 0)) - 15
            if signal_period in wav_by_signal_period:
                d["_signal_period"] = signal_period
                d["_wav"] = wav_by_signal_period[signal_period]
                decodes_by_signal_period[signal_period].append(d)

    if not decodes_by_signal_period:
        print("No matching decodes found.", file=sys.stderr)
        sys.exit(0)

    if args.wsjt:
        # JTDX-compatible format: YYYYMMDD_HHMMSS SNR DT FREQ ~ MSG
        # Timestamp uses signal period start so it aligns with JTDX timestamps.
        for period in sorted(decodes_by_signal_period):
            for d in decodes_by_signal_period[period]:
                dt       = datetime.fromtimestamp(period, tz=timezone.utc)
                ts       = dt.strftime("%Y%m%d_%H%M%S")
                snr      = int(d.get("snr", 0))
                offset   = d.get("offset", 0.0)
                freq     = d.get("freq", band["dial"])
                audio_hz = freq - band["dial"]
                msg      = d.get("call", "?")
                print(f"{ts} {snr:+3d} {offset:5.1f} {audio_hz:5.0f} ~ {msg}")
    else:
        for period in sorted(decodes_by_signal_period):
            wav_name = decodes_by_signal_period[period][0]["_wav"]
            dt = datetime.fromtimestamp(period, tz=timezone.utc)
            print(f"\n=== {wav_name}  ({dt.strftime('%H:%M:%S')} UTC) ===")
            for d in sorted(decodes_by_signal_period[period], key=lambda x: x.get("freq", 0)):
                snr      = int(d.get("snr", 0))
                offset   = d.get("offset", 0.0)
                freq     = d.get("freq", band["dial"])
                audio_hz = freq - band["dial"]
                msg      = d.get("call", "?")
                print(f"  {snr:+3d}  {offset:4.1f}  {audio_hz:5.0f}  {msg}")
        print()


if __name__ == "__main__":
    main()
