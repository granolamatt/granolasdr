"""Overlap experiment: does STFT oversampling (time AND frequency) surface FT8
sync candidates that no-overlap misses?

Extracts the 20m FT8 band from a recording, runs Costas sync at high overlap
(time_osr=freq_osr=OSR) and no overlap (1,1), and diffs the candidate sets by
frequency.  Sync-level proxy for "does overlap help"; a decode-truth version
(ft8_lib LDPC+CRC binding) is the follow-on if we want CRC-confirmed counts.

Usage:  python coherent/overlap_test.py raw.dat [band] [OSR]
"""
import os
import sys
import numpy as np
from gnlh import open_recording
from ft8 import extract_band, stft, costas_sync, find_candidates, ft8_dial_comp_hz, SYM_SEC

FMAX_HZ = 3000.0        # FT8 audio span above the dial
THRESH  = 4.0           # sync SNR to call something a real signal (score/median)
DECODE  = 3.0           # sync-SNR floor below which a signal is unlikely to decode


def analyze(bb, sr, time_osr, freq_osr):
    """Return (score, bin_hz, hop_sec) for the given overlap."""
    nsps    = int(round(sr * SYM_SEC))          # 2048 at 12.8 kHz
    spec    = stft(bb, nsps, time_osr, freq_osr)
    bin_hz  = sr / (nsps * freq_osr)
    hop_sec = (nsps // time_osr) / sr
    score, _ = costas_sync(spec, time_osr, freq_osr, FMAX_HZ, bin_hz)
    return score, bin_hz, hop_sec


def peak_at(score, bin_hz, hop_sec, freq_hz, time_sec, df_hz=6.25, dt_sec=0.32):
    """Best sync score near (freq_hz, time_sec) in another osr's grid (handles the
    sub-bin/sub-symbol offset the coarse grid can't represent)."""
    f = int(round(freq_hz / bin_hz)); t = int(round(time_sec / hop_sec))
    fr = max(1, int(df_hz / bin_hz)); tr = max(1, int(dt_sec / hop_sec))
    f0, f1 = max(0, f - fr), min(score.shape[1], f + fr + 1)
    t0, t1 = max(0, t - tr), min(score.shape[0], t + tr + 1)
    if f0 >= f1 or t0 >= t1:
        return 0.0
    return float(score[t0:t1, f0:f1].max())


def main(path, band="20m", osr=4):
    rec = open_recording(path)
    f0 = ft8_dial_comp_hz(band)
    secs = float(os.environ.get("SECS", "0"))
    print(f"{path}  band={band}  FT8 dial @ {f0/1000:.1f} kHz  OSR={osr}"
          f"{f'  (first {secs:.0f}s)' if secs else ''}\n")
    bb, sr = extract_band(rec["samples"], rec["sample_rate"], f0, decim=32)
    if secs:
        bb = bb[:int(secs * sr)]
    print(f"baseband: {len(bb)/sr:.1f} s @ {sr:.0f} Hz\n")

    sc_o, bh_o, hp_o = analyze(bb, sr, osr, osr)
    sc_n, bh_n, hp_n = analyze(bb, sr, 1, 1)

    # Distinct real signals from the overlap run (strong NMS: 12 Hz / 1 s).
    sigs = find_candidates(sc_o, osr, osr, bh_o, hp_o, THRESH,
                           min_sep_hz=12.0, min_sep_sec=1.0)
    print(f"distinct signals (overlap, snr>={THRESH}): {len(sigs)}\n")
    if not sigs:
        print("none above threshold — lower THRESH or check the band/mapping.")
        return

    # For each, the best no-overlap sync score at the same freq/time.
    lost, weaker, deltas = 0, 0, []
    for s in sigs:
        n = peak_at(sc_n, bh_n, hp_n, s["freq_hz"], s["time_sec"])
        deltas.append(s["snr"] - n)
        if n < DECODE:
            lost += 1
        elif n < s["snr"] - 0.5:
            weaker += 1

    deltas = np.array(deltas)
    print(f"paired sync-SNR (overlap vs no-overlap), n={len(sigs)}:")
    print(f"  mean drop when overlap removed : {deltas.mean():+.2f}  (median {np.median(deltas):+.2f})")
    print(f"  signals that fall below decode floor ({DECODE}) without overlap : {lost}")
    print(f"  signals meaningfully weaker (but still above floor)             : {weaker}")
    print(f"  signals essentially unchanged                                   : {len(sigs)-lost-weaker}\n")
    print("strongest signals (freq, overlap snr -> no-overlap snr):")
    for s in sorted(sigs, key=lambda c: -c["snr"])[:15]:
        n = peak_at(sc_n, bh_n, hp_n, s["freq_hz"], s["time_sec"])
        flag = "  <-- LOST" if n < DECODE else ("  (weaker)" if n < s["snr"]-0.5 else "")
        print(f"  {s['freq_hz']:7.1f} Hz  t={s['time_sec']:6.1f}s  {s['snr']:5.2f} -> {n:5.2f}{flag}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python coherent/overlap_test.py <recording.dat> [band] [OSR]")
        sys.exit(1)
    p = sys.argv[1]
    band = sys.argv[2] if len(sys.argv) > 2 else "20m"
    osr = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    main(p, band, osr)
