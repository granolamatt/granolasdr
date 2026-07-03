"""Decode-level overlap experiment: does STFT oversampling (time AND frequency)
change the set of CRC-valid FT8 decodes?

Decodes the 20m band across the recording at (4,4), (2,2), (1,1) STFT overlap
using the ft8_lib binding, dedups unique messages, and diffs the sets. This is
the decode-truth version of the sync-level probe (CRC is the oracle).

Usage:  python3.14 coherent/decode_overlap.py ../raw.dat [secs]
"""
import sys
import time
import numpy as np
import ft8decode
from gnlh import open_recording
from ft8 import extract_band, ft8_dial_comp_hz

DIAL_AUDIO = 300.0
OSRS = [(4, 4), (2, 2), (1, 1)]


def decode_all(audio, sr, tosr, fosr, win_s=15.0, step_s=5.0):
    W, STEP = int(win_s * sr), int(step_s * sr)
    best = {}                       # text -> best score seen
    for start in range(0, max(1, len(audio) - W), STEP):
        for m in ft8decode.decode_audio(audio[start:start + W], sr, tosr, fosr, 200.0, 3400.0):
            t = m["text"]
            if t not in best or m["score"] > best[t]:
                best[t] = m["score"]
    return best


def main(path, secs=0.0):
    rec = open_recording(path)
    sr_in = rec["sample_rate"]
    x = rec["samples"]
    if secs:
        x = x[:int(secs * sr_in)]
    x = np.asarray(x)
    f0 = ft8_dial_comp_hz("20m")
    bb, sr = extract_band(x, sr_in, f0 - DIAL_AUDIO, decim=32)
    audio = np.real(bb).astype(np.float32)
    sr = int(round(sr))
    print(f"20m real audio: {len(audio)/sr:.1f} s @ {sr} Hz\n")

    results = {}
    for (to, fo) in OSRS:
        t0 = time.time()
        d = decode_all(audio, sr, to, fo)
        results[(to, fo)] = set(d.keys())
        print(f"osr ({to},{fo}): {len(d):4d} unique CRC-valid messages   ({time.time()-t0:.1f}s)")

    base = results[(1, 1)]
    print()
    for (to, fo) in OSRS:
        if (to, fo) == (1, 1):
            continue
        s = results[(to, fo)]
        only_over = s - base
        only_none = base - s
        print(f"({to},{fo}) vs (1,1):  both={len(s & base)}  "
              f"only ({to},{fo})={len(only_over)}  only (1,1)={len(only_none)}")
        for t in sorted(only_over)[:12]:
            print(f"    + {t}")
        for t in sorted(only_none)[:6]:
            print(f"    - (1,1) only) {t}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "../raw.dat"
    s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    main(p, s)
