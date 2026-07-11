"""Real-capture A/B: how many decodes does de-chirp recover on a live recording,
and are they genuinely drifting?

For each coarse candidate in a (band, time-window) of a --record GNLH capture,
refine two ways and decode both:
  BASELINE = fixed-df align (gm/hf/refine.cc before de-chirp)
  DECHIRP  = same, plus the gated linear-slope search (what ships now)
Reports baseline vs de-chirp decode counts, the messages only de-chirp got, and
the fitted slope of each — a slope near 0 is a marginal recovery, a slope past
~0.5 Hz/s is a real drifter the old path could not have decoded.

Usage: python3.14 dechirp_capture_ab.py <file> [band] [start_sec] [dur_sec]
"""
import sys
import numpy as np
import ft8decode
from gnlh import open_recording
from ft8 import extract_band, ft8_dial_comp_hz
from coherent_refine import (SR, FRAME, NSYM, GRAY, DATA_POS, CPOS, CTONE,
                             sym_tones, costas_energy, llr174, coarse_candidates)

DIAL = 300.0
_NN = np.arange(FRAME)
_T  = _NN / SR


def aligned(bb, start, f, dt, df, slope):
    """Frame mixed to DC by instantaneous freq (f+df) + slope*t (de-chirp)."""
    s = start + dt
    if s < 0 or s + FRAME > len(bb):
        return None
    ph = 2 * np.pi * ((f + df) * _T + 0.5 * slope * _T * _T)
    return bb[s:s + FRAME] * np.exp(-1j * ph)


def energy(bb, start, f, dt, df, slope):
    fr = aligned(bb, start, f, dt, df, slope)
    return (-1.0 if fr is None else costas_energy(fr))


def refine_both(bb, start, f):
    """Returns (flat_frame, dechirp_frame, slope) sharing the flat dt/df search."""
    # coordinate descent, slope=0 (mirrors refine.cc)
    best_e, best_dt = -1.0, 0
    for dt in range(-512, 513, 64):
        e = energy(bb, start, f, dt, 0.0, 0.0)
        if e > best_e:
            best_e, best_dt = e, dt
    best_e, best_df = -1.0, 0.0
    for df in np.linspace(-3.5, 3.5, 13):
        e = energy(bb, start, f, best_dt, df, 0.0)
        if e > best_e:
            best_e, best_df = e, df
    best_e = -1.0
    for dt in range(-512, 513, 64):
        e = energy(bb, start, f, dt, best_df, 0.0)
        if e > best_e:
            best_e, best_dt = e, dt
    e_flat = best_e
    flat = aligned(bb, start, f, best_dt, best_df, 0.0)

    # gated slope search
    cand_slope, e_slope = 0.0, e_flat
    for slope in np.linspace(-5.0, 5.0, 21):
        if slope == 0.0:
            continue
        e = energy(bb, start, f, best_dt, best_df, slope)
        if e > e_slope:
            e_slope, cand_slope = e, slope
    slope = 0.0
    df = best_df
    if cand_slope != 0.0 and e_slope > e_flat * 1.05:
        slope = cand_slope
        be = -1.0
        for d2 in np.linspace(-3.5, 3.5, 13):
            e = energy(bb, start, f, best_dt, d2, slope)
            if e > be:
                be, df = e, d2
    dech = aligned(bb, start, f, best_dt, df, slope)
    return flat, dech, slope


def decode(frame):
    return None if frame is None else ft8decode.decode_llr(llr174(sym_tones(frame)))


def main(path, band, start_sec, dur_sec):
    rec = open_recording(path)
    sr_in = rec["sample_rate"]
    x = rec["samples"]
    a = int(start_sec * sr_in)
    b = int((start_sec + dur_sec) * sr_in) if dur_sec else len(x)
    x = np.asarray(x[a:min(b, len(x))])
    bb, _ = extract_band(x, sr_in, ft8_dial_comp_hz(band) - DIAL, decim=32)
    print(f"{band}: {start_sec:.0f}..{start_sec+dur_sec:.0f}s  ({len(bb)/SR:.1f}s baseband)")

    cands = coarse_candidates(np.real(bb).astype(np.float32))
    print(f"coarse candidates: {len(cands)}")

    base, dech = {}, {}          # text -> best slope seen
    dech_slope = {}
    for f, t in cands:
        start = int(round(t * SR))
        flat, dc, slope = refine_both(bb, start, f)
        mb, md = decode(flat), decode(dc)
        if mb:
            base[mb] = base.get(mb, 0)
        if md:
            dech[md] = dech.get(md, 0)
            dech_slope[md] = slope

    only = sorted(set(dech) - set(base))
    print(f"\nbaseline decodes : {len(base)}")
    print(f"de-chirp decodes : {len(dech)}")
    print(f"de-chirp recovered {len(only)} the baseline missed:")
    drifters = 0
    for t in only:
        s = dech_slope[t]
        tag = "  <- DRIFTER" if abs(s) >= 0.5 else "  (marginal)"
        if abs(s) >= 0.5:
            drifters += 1
        print(f"    slope={s:+.2f} Hz/s  {t}{tag}")
    lost = sorted(set(base) - set(dech))
    if lost:
        print(f"baseline got, de-chirp lost ({len(lost)}): {lost}")
    print(f"\nnet: +{len(dech)-len(base)} decodes, {drifters} genuine drifters "
          f"(|slope|>=0.5 Hz/s)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../ft8test.dat"
    band = sys.argv[2] if len(sys.argv) > 2 else "20m"
    st   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    du   = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
    main(path, band, st, du)
