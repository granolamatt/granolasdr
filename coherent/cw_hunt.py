"""Hunt a CW composite capture: scan it with cw_offline, score every decoded
signal, rank by confidence, and extract a verification audio clip for each so you
can copy them by ear and mark real/fake (ground truth).

Pipeline: for each window -> ../build/cw_offline (detect + track + decode) -> parse
[CW] spots -> aggregate per call (count, best SNR, freq, time) -> rank -> cw_audio
a clip at each call's strongest moment -> write a manifest (index.txt) + WAVs.

Usage:
  python3.14 cw_hunt.py <capture.dat> [--start S] [--span SEC] [--win SEC]
     [--audio-top N] [--outdir DIR] [--dur SEC]
"""
import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CW_OFFLINE = os.path.join(HERE, "..", "build", "cw_offline")
CW_AUDIO = os.path.join(HERE, "cw_audio.py")
# [CW] t=   106s   7036.800 kHz  ~28 wpm  snr 20  KJ9C
SPOT_RE = re.compile(r"\[CW\]\s+t=\s*(\d+)s\s+([\d.]+)\s+kHz\s+~\s*(\d+)\s+wpm\s+snr\s+(\d+)\s+(\S+)")


def scan_window(cap, start, dur):
    """Run cw_offline on one window, return list of (t, freq_khz, wpm, snr, call)."""
    try:
        out = subprocess.run([CW_OFFLINE, cap, str(start), str(dur)],
                             capture_output=True, text=True, timeout=600).stdout
    except Exception as e:
        print(f"  window {start}: cw_offline failed: {e}", file=sys.stderr)
        return []
    spots = []
    for line in out.splitlines():
        m = SPOT_RE.search(line)
        if m:
            spots.append((int(m.group(1)), float(m.group(2)), int(m.group(3)),
                          int(m.group(4)), m.group(5)))
    return spots


def confidence(count, snr):
    if count >= 3 or snr >= 20:  return "HIGH"
    if count >= 2 or snr >= 14:  return "MED"
    return "LOW"


def make_audio(cap, freq, t, dur, out):
    subprocess.run([sys.executable, CW_AUDIO, cap, "--freq", f"{freq:.1f}",
                    "--mode", "cw", "--start", str(max(0, t - 4)), "--dur", str(dur),
                    "--out", out], capture_output=True, timeout=300)


def main():
    ap = argparse.ArgumentParser(description="Hunt + score + audition CW signals in a capture")
    ap.add_argument("capture")
    ap.add_argument("--start", type=float, default=0.0, help="scan start (s)")
    ap.add_argument("--span", type=float, default=600.0, help="total seconds to scan")
    ap.add_argument("--win", type=float, default=60.0, help="window length per cw_offline run (s)")
    ap.add_argument("--audio-top", type=int, default=20, help="generate clips for the top N calls (0=all)")
    ap.add_argument("--dur", type=float, default=18.0, help="clip length (s)")
    ap.add_argument("--outdir", default="hunt", help="output dir for WAVs + index.txt")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Key on (call, 0.5 kHz freq cluster): a signal is a callsign AT a frequency.
    # Keying on call alone merges a real station with same-call misreads on other
    # carriers, inflating its count. A high in-cluster drift then flags a genuine
    # carrier's small bin-wander vs a call scattered across the band (misreads).
    agg = defaultdict(lambda: {"count": 0, "snr": 0, "freq": 0.0, "t": 0, "wpm": 0,
                               "fmin": 1e9, "fmax": 0.0})
    nwin = int(args.span // args.win)
    print(f"scanning {args.capture}: {args.start:.0f}..{args.start+args.span:.0f}s "
          f"in {nwin} x {args.win:.0f}s windows...")
    for w in range(nwin):
        st = args.start + w * args.win
        spots = scan_window(args.capture, st, args.win)
        for (t, freq, wpm, snr, call) in spots:
            a = agg[(call, round(freq * 2) / 2)]        # 0.5 kHz cluster
            a["count"] += 1
            a["fmin"] = min(a["fmin"], freq); a["fmax"] = max(a["fmax"], freq)
            if snr > a["snr"]:
                a["snr"], a["freq"], a["t"], a["wpm"] = snr, freq, st + t, wpm
        print(f"  {st:.0f}s: {len(spots)} spots, {len(agg)} signals so far")

    ranked = sorted(agg.items(), key=lambda kv: (kv[1]["count"], kv[1]["snr"]), reverse=True)
    if not ranked:
        print("no CW decoded in this span — try another --start or a wider --span")
        return

    idx_path = os.path.join(args.outdir, "index.txt")
    n_audio = len(ranked) if args.audio_top == 0 else min(args.audio_top, len(ranked))
    print(f"\n{len(ranked)} unique calls; auditioning top {n_audio}\n")
    hdr = f"{'#':>3} {'call':<9} {'conf':<4} {'cnt':>3} {'snr':>3} {'freq_kHz':>9} " \
          f"{'t(s)':>6} {'wpm':>3} {'drift':>5}  wav"
    with open(idx_path, "w") as idx:
        idx.write(hdr + "\n" + "-" * len(hdr) + "\n")
        print(hdr); print("-" * len(hdr))
        for i, ((call, _bucket), a) in enumerate(ranked, 1):
            conf = confidence(a["count"], a["snr"])
            drift = a["fmax"] - a["fmin"]                 # kHz spread across windows
            wav = ""
            if i <= n_audio:
                wav = f"{i:02d}_{call}_{a['freq']:.1f}.wav"
                make_audio(args.capture, a["freq"], int(a["t"]), args.dur,
                           os.path.join(args.outdir, wav))
            row = (f"{i:>3} {call:<9} {conf:<4} {a['count']:>3} {a['snr']:>3} "
                   f"{a['freq']:>9.3f} {int(a['t']):>6} {a['wpm']:>3} {drift:>5.2f}  {wav}")
            print(row); idx.write(row + "\n")
    print(f"\nmanifest: {idx_path}   WAVs: {args.outdir}/  "
          f"(HIGH=likely real, LOW=verify carefully)")


if __name__ == "__main__":
    main()
