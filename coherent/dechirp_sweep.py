"""Sweep the de-chirp A/B across bands x time windows of a capture and aggregate.
Finds where (if anywhere) real signals drift enough that de-chirp recovers them.

Usage: python3.14 dechirp_sweep.py <file> [dur_sec]
"""
import sys
import numpy as np
import ft8decode
from gnlh import open_recording
from ft8 import extract_band, ft8_dial_comp_hz
from coherent_refine import SR, coarse_candidates
from dechirp_capture_ab import refine_both, decode, DIAL

BANDS  = ["40m", "20m", "17m", "15m", "10m"]
STARTS = [300, 6000, 12000, 18000, 24000]     # spread across the ~413 min capture


def run_window(x, sr_in, band, start_sec, dur_sec):
    a = int(start_sec * sr_in)
    b = int((start_sec + dur_sec) * sr_in)
    if a >= len(x):
        return None
    seg = np.asarray(x[a:min(b, len(x))])
    bb, _ = extract_band(seg, sr_in, ft8_dial_comp_hz(band) - DIAL, decim=32)
    cands = coarse_candidates(np.real(bb).astype(np.float32))
    base, dech, slopes = set(), set(), {}
    for f, t in cands:
        start = int(round(t * SR))
        flat, dc, slope = refine_both(bb, start, f)
        mb, md = decode(flat), decode(dc)
        if mb:
            base.add(mb)
        if md:
            dech.add(md)
            slopes[md] = slope
    only = sorted(set(dech) - set(base))
    return base, dech, only, slopes


def main(path, dur_sec):
    rec = open_recording(path)
    sr_in = rec["sample_rate"]
    x = rec["samples"]
    print(f"capture {len(x)/sr_in/60:.0f} min; windows of {dur_sec:.0f}s\n")
    print(f"{'band':>5} {'t(s)':>7} {'cands?':>6} {'base':>5} {'dech':>5} {'+rec':>5} {'drift':>5}")
    print("-" * 45)
    tot_b = tot_d = tot_drift = 0
    drift_log = []
    for band in BANDS:
        for st in STARTS:
            r = run_window(x, sr_in, band, st, dur_sec)
            if r is None:
                continue
            base, dech, only, slopes = r
            drift = [(t, slopes[t]) for t in only if abs(slopes[t]) >= 0.5]
            print(f"{band:>5} {st:>7} {'':>6} {len(base):>5} {len(dech):>5} "
                  f"{len(only):>5} {len(drift):>5}")
            tot_b += len(base); tot_d += len(dech); tot_drift += len(drift)
            for t, s in drift:
                drift_log.append((band, st, s, t))
    print("-" * 45)
    print(f"TOTAL baseline={tot_b}  de-chirp={tot_d}  "
          f"net=+{tot_d-tot_b}  drifters={tot_drift}")
    if drift_log:
        print("\nDrifters de-chirp recovered (band, t, slope, msg):")
        for band, st, s, t in drift_log:
            print(f"  {band:>4} t={st:>6}  slope={s:+.2f} Hz/s  {t}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../ft8test.dat"
    dur  = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    main(path, dur)
