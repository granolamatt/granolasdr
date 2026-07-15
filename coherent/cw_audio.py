"""Offline CW audio extraction from a --record-cw capture (819.2 kHz complex CW
composite, GNLH format). Turns the raw IQ into listenable/decodable audio so a
skimmer spot can be verified by ear or fed to a reference decoder.

Three modes:
  cw   single signal -> narrow mono WAV with a ~600 Hz BFO tone (ear-copy / fldigi)
  ssb  a few-kHz slice -> mono WAV, tune-around multiple signals like an SSB RX
  iq   band window -> stereo I/Q WAV for CW Skimmer / SDR software (same-RF reference)

RF frequency maps into the packed CW composite via kCWBands (mirrors
gm/hf/cw_bands.h + CWSkimmerCuda::binToHz). See --help.

Usage:
  python3.14 cw_audio.py <capture.dat> --freq <RF kHz> [--mode cw|ssb|iq]
     [--start S] [--dur S] [--out W.wav] [--rate N] [--bfo Hz] [--bw Hz] [--width Hz]
"""
import argparse
import numpy as np
from scipy.signal import resample_poly
from scipy.io import wavfile
from gnlh import open_recording

SR = 819200.0            # CW composite sample rate

# CW sub-bands, packed contiguously into the composite. Mirrors gm/hf/cw_bands.h
# (wb_start/wb_end in 100 Hz wideband-FFT bin units; bw = wb_end - wb_start).
# name, wb_start, wb_end
CW_BANDS = [
    ("160m",  18000,  18400), ("80m",  35000,  35600), ("40m",  70000,  70500),
    ("30m",  101000, 101300), ("20m", 140000, 140700), ("17m", 180680, 180980),
    ("15m",  210000, 210900), ("12m", 248900, 249200), ("10m", 280000, 280700),
]


def rf_khz_to_offset(rf_khz):
    """RF kHz -> (composite Hz offset in the 819.2 kHz stream, band name).
    Inverse of CWSkimmerCuda::binToHz: wb=RF*10; p = cum + (wb - wb_start); f = p*100."""
    wb = int(round(rf_khz * 10.0))
    cum = 0
    for name, ws, we in CW_BANDS:
        if ws <= wb < we:
            p = cum + (wb - ws)
            return p * 100.0, name
        cum += (we - ws)
    raise SystemExit(f"{rf_khz} kHz is not inside any CW sub-band "
                     f"(bands: {', '.join(b[0] for b in CW_BANDS)})")


def read_window(path, start_sec, dur_sec):
    rec = open_recording(path)
    sr_in = rec["sample_rate"]
    if sr_in != int(SR):
        print(f"WARNING: sr={sr_in}, expected {int(SR)} (CW composite) — is this a CW capture?")
    x = rec["samples"]
    a = int(start_sec * sr_in)
    n = int(dur_sec * sr_in)
    return np.asarray(x[a:a + n]), sr_in


def rational(sr_in, rate):
    """up/down for resample_poly sr_in -> rate, reduced by gcd."""
    g = np.gcd(int(sr_in), int(rate))
    return int(rate) // g, int(sr_in) // g


def to_int16(sig, headroom=0.92):
    peak = np.max(np.abs(sig))
    if peak <= 0:
        peak = 1.0
    return np.round(sig / peak * headroom * 32767.0).astype(np.int16)


def main():
    ap = argparse.ArgumentParser(description="Extract audio from a CW composite capture")
    ap.add_argument("capture")
    ap.add_argument("--freq", type=float, required=True, help="RF frequency in kHz (e.g. 7036.8)")
    ap.add_argument("--mode", choices=["cw", "ssb", "iq"], default="cw")
    ap.add_argument("--start", type=float, default=0.0, help="window start (s)")
    ap.add_argument("--dur", type=float, default=30.0, help="window length (s)")
    ap.add_argument("--out", default=None, help="output WAV path")
    ap.add_argument("--rate", type=int, default=None, help="audio/IQ rate (default 8000 cw/ssb, 48000 iq)")
    ap.add_argument("--bfo", type=float, default=600.0, help="cw beat pitch (Hz)")
    ap.add_argument("--bw", type=float, default=300.0, help="cw filter bandwidth (Hz)")
    ap.add_argument("--width", type=float, default=3000.0, help="ssb passband width (Hz)")
    args = ap.parse_args()

    rate = args.rate or (48000 if args.mode == "iq" else 8000)
    out = args.out or f"cw_{args.mode}_{args.freq:.1f}_{int(args.start)}s.wav"

    f_offset, band = rf_khz_to_offset(args.freq)
    seg, sr_in = read_window(args.capture, args.start, args.dur)
    n = np.arange(len(seg), dtype=np.float64)
    print(f"{args.freq:.1f} kHz ({band}) -> composite offset {f_offset/1000:.1f} kHz  "
          f"| {args.start:.0f}..{args.start+args.dur:.0f}s  mode={args.mode} rate={rate}")

    up, down = rational(sr_in, rate)

    if args.mode == "iq":
        # Center the chosen freq at DC; keep complex; stereo I/Q WAV for CW Skimmer.
        bb = seg * np.exp(-2j * np.pi * f_offset * n / sr_in)
        bb = resample_poly(bb, up, down)
        peak = max(np.max(np.abs(bb.real)), np.max(np.abs(bb.imag)), 1e-9)
        iq = np.empty((len(bb), 2), np.int16)
        iq[:, 0] = np.round(bb.real / peak * 0.92 * 32767.0)
        iq[:, 1] = np.round(bb.imag / peak * 0.92 * 32767.0)
        wavfile.write(out, rate, iq)
        print(f"wrote {out}  ({len(bb)/rate:.1f}s stereo I/Q @ {rate} Hz)")
        print(f"-> in CW Skimmer, set the receiver/center frequency to {args.freq:.1f} kHz")
        return

    if args.mode == "cw":
        # Put the carrier at +bfo, low-pass to a CW bandwidth, take the real part.
        bb = seg * np.exp(-2j * np.pi * (f_offset - args.bfo) * n / sr_in)
        bb = resample_poly(bb, up, down)
        audio = bb.real.astype(np.float64)
        # Narrow bandpass around the BFO (±bw/2) so neighbours don't intrude.
        if 0 < args.bw < rate:
            f = np.fft.rfftfreq(len(audio), 1.0 / rate)
            H = ((f >= max(0.0, args.bfo - args.bw / 2)) & (f <= args.bfo + args.bw / 2)).astype(float)
            audio = np.fft.irfft(np.fft.rfft(audio) * H, n=len(audio))
    else:  # ssb: tuned freq near the low audio edge, keep a ~width slice
        bb = seg * np.exp(-2j * np.pi * (f_offset - 300.0) * n / sr_in)
        bb = resample_poly(bb, up, down)
        audio = bb.real.astype(np.float64)
        if 0 < args.width < rate:
            f = np.fft.rfftfreq(len(audio), 1.0 / rate)
            H = (f <= 300.0 + args.width).astype(float)
            audio = np.fft.irfft(np.fft.rfft(audio) * H, n=len(audio))

    wavfile.write(out, rate, to_int16(audio))
    print(f"wrote {out}  ({len(audio)/rate:.1f}s mono @ {rate} Hz)")


if __name__ == "__main__":
    main()
