"""Band-energy probe: which bands are active in a recording, and does the
composite frequency mapping land where FT8 should be?

Averages a power spectrum over the recording (4096-pt FFTs -> 100 Hz bins, the
composite bin spacing) and reports energy per band, plus the energy right at
each band's FT8 dial sub-region. This validates the composite->RF mapping before
we build the overlap experiment on top of it.

Usage:  python coherent/probe.py raw.dat
"""
import sys
import numpy as np
from gnlh import open_recording, band_layout, HF_BANDS, FT8_DIAL


def main(path):
    rec = open_recording(path)
    x, sr = rec["samples"], rec["sample_rate"]
    dur = len(x) / sr
    print(f"{path}: {len(x)} samples, {dur:.1f} s ({dur/60:.2f} min) at {sr} Hz\n")

    N = 4096                          # 100 Hz/bin, matches composite bin spacing
    win = np.hanning(N).astype(np.float32)
    total_frames = len(x) // N
    nframes = min(3000, total_frames)
    step = max(1, total_frames // nframes)

    acc = np.zeros(N)
    cnt = 0
    for i in range(0, total_frames, step):
        seg = np.asarray(x[i*N:(i+1)*N])
        if len(seg) < N:
            break
        X = np.fft.fft(seg * win)
        acc += np.abs(X) ** 2
        cnt += 1
    psd = acc / max(cnt, 1)           # index b = frequency b*100 Hz = composite bin b
    print(f"averaged {cnt} frames; noise-ish median PSD = {np.median(psd):.3e}\n")

    ws_lut = {name: ws for name, ws, _ in HF_BANDS}
    print(f"{'band':6} {'comp kHz':>13} {'band E':>11} {'FT8 E':>11} {'FT8/med':>9}")
    med = np.median(psd)
    for name, cb, bw, ws in band_layout():
        band_e = psd[cb:cb+bw].sum()
        ft8_e = 0.0
        if name in FT8_DIAL:
            # FT8 sub-region: dial .. dial+30 (3 kHz) in composite bins
            d = cb + (FT8_DIAL[name] - ws)
            ft8_e = psd[d:d+30].sum()
        lo, hi = cb*100/1000, (cb+bw)*100/1000
        ratio = (ft8_e / 30) / med if med > 0 else 0
        print(f"{name:6} {lo:6.1f}-{hi:5.1f} {band_e:11.3e} {ft8_e:11.3e} {ratio:8.1f}x")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python coherent/probe.py <recording.dat>")
        sys.exit(1)
    main(sys.argv[1])
