"""Probe a CW composite capture (--record-cw, 819.2 kHz): confirm real keyed CW
is present and characterize the detection metric CWSkimmerCuda uses.

Replicates the CW MagBlock front end (16384-pt FFT, 5 ms hop) and the detector's
per-bin percentile metric (p80 - p50 over a ~5 s window), then ranks bins and
flags which look like real keyed carriers (bimodal on/off) vs noise. This is the
data CW2 (adaptive detection floor) is tuned against.

Usage: python3.14 cw_probe.py <cwcapture.dat> [start_sec] [dur_sec]
"""
import sys
import numpy as np
from gnlh import open_recording

RFFT   = 16384          # CW MagBlock FFT
HOP    = RFFT // 4       # time_osr=4 -> 5 ms hop @ 819.2 kHz
SR     = 819200.0
BIN_HZ = SR / RFFT       # 50 Hz/bin
CONTENT_BINS = 9400      # 2 * Sigma bw (CW content occupies the lower bins)


def main(path, start_sec, dur_sec):
    rec = open_recording(path)
    sr_in = rec["sample_rate"]
    if sr_in != 819200:
        print(f"WARNING: sr={sr_in}, expected 819200 (CW composite)")
    x = rec["samples"]
    a = int(start_sec * sr_in)
    n = int(dur_sec * sr_in)
    seg = np.asarray(x[a:a + n])
    nfr = 1 + (len(seg) - RFFT) // HOP
    print(f"{path}: {start_sec:.0f}..{start_sec+dur_sec:.0f}s  {nfr} frames "
          f"({dur_sec:.0f}s @ 5ms hop), {CONTENT_BINS} content bins @ {BIN_HZ:.0f} Hz")

    # STFT magnitude, content bins only. Compress to a MagBlock-like uint8 scale
    # (log magnitude) so the percentile spread matches the on-hardware metric.
    win = np.hanning(RFFT).astype(np.float32)
    mag = np.empty((nfr, CONTENT_BINS), np.float32)
    for i in range(nfr):
        seg_i = seg[i*HOP:i*HOP+RFFT] * win
        F = np.fft.fft(seg_i)[:CONTENT_BINS]
        mag[i] = 20.0 * np.log10(np.abs(F) + 1e-9)

    # Per-bin percentile metric (what the detector gates on) + a keying indicator.
    p50 = np.percentile(mag, 50, axis=0)
    p80 = np.percentile(mag, 80, axis=0)
    metric = p80 - p50                                  # CWSkimmerCuda snr[bin]
    # Keying duty: fraction of frames above the bin's own mid level. A real keyed
    # carrier sits ~20-60%; steady carrier ~100%; noise erratic/low.
    mid = p50 + 0.5 * (mag.max(axis=0) - p50)
    duty = (mag > mid).mean(axis=0)
    # Per-bin noise floor = its own median; band noise = rolling median of p50.
    band_floor = np.array([np.median(p50[max(0,b-40):b+40]) for b in range(CONTENT_BINS)])
    excess = p50 - band_floor                           # how far this bin's floor sits above the band

    order = np.argsort(metric)[::-1]
    print(f"\nmetric (p80-p50) distribution: median={np.median(metric):.1f} "
          f"p90={np.percentile(metric,90):.1f} max={metric.max():.1f}")
    print(f"bins over the fixed gate (metric>=12): {(metric>=12).sum()} "
          f"of {CONTENT_BINS}\n")
    print(f"{'magbin':>7} {'~kHzoff':>8} {'metric':>7} {'duty%':>6} {'floor+':>7}  verdict")
    print("-" * 60)
    shown = 0
    for b in order:
        if metric[b] < 8 or shown >= 25:
            break
        # keyed if the metric is high AND duty cycle is in the CW range
        keyed = 0.12 <= duty[b] <= 0.75
        v = "keyed CW?" if keyed else ("steady/carrier" if duty[b] > 0.85 else "noise-ish")
        print(f"{b:>7} {b*BIN_HZ/2000:>8.2f} {metric[b]:>7.1f} {duty[b]*100:>6.0f} "
              f"{excess[b]:>7.1f}  {v}")
        shown += 1
    keyed_n = sum(1 for b in order[:200] if metric[b] >= 12 and 0.12 <= duty[b] <= 0.75)
    print(f"\ntop-200-by-metric with keyed-CW duty (0.12-0.75): {keyed_n}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../cwtest.dat"
    st   = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    du   = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    main(path, st, du)
