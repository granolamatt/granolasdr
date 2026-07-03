"""Validation: does the ft8decode binding decode real FT8 from the recording?
Extracts 20m as real audio (FT8 dial -> 300 Hz), windows it, decodes at (2,2),
and prints unique CRC-valid messages. Real callsigns == the whole chain works.

Usage:  python coherent/validate_decode.py ../raw.dat [secs]
"""
import sys
import numpy as np
import ft8decode
from gnlh import open_recording
from ft8 import extract_band, ft8_dial_comp_hz

DIAL_AUDIO = 300.0     # place the FT8 dial at 300 Hz in the audio passband


def main(path, secs=90.0):
    rec = open_recording(path)
    sr_in = rec["sample_rate"]
    x = np.asarray(rec["samples"][:int(secs * sr_in)])
    f0 = ft8_dial_comp_hz("20m")
    bb, sr = extract_band(x, sr_in, f0 - DIAL_AUDIO, decim=32)   # complex @ 12800
    audio = np.real(bb).astype(np.float32)                       # real audio
    sr = int(round(sr))
    print(f"20m real audio: {len(audio)/sr:.1f} s @ {sr} Hz\n")

    W, STEP = int(15 * sr), int(5 * sr)
    seen = {}
    for start in range(0, max(1, len(audio) - W), STEP):
        msgs = ft8decode.decode_audio(audio[start:start + W], sr, 2, 2, 200.0, 3400.0)
        for m in msgs:
            if m["text"] not in seen:
                seen[m["text"]] = (start / sr + m["time_sec"], m["freq_hz"], m["score"])
    for text, (t, f, sc) in sorted(seen.items(), key=lambda kv: kv[1][0]):
        print(f"  {t:6.1f}s  {f:6.1f} Hz  score={sc:3d}  {text}")
    print(f"\n{len(seen)} unique messages decoded")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "../raw.dat"
    s = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
    main(p, s)
