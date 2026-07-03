"""Coherent refine experiment: can coarse-detect + Costas-pilot coherent refine
match the brute-force 4x4 overlap decode count, at low overlap cost?

Pipeline per coarse candidate (found at (2,2)):
  1. extract the complex frame (79 symbols) centered on the candidate
  2. refine frequency by maximizing Costas sync energy       (alignment)
  3. estimate carrier phase phi(t) from the 3 Costas blocks  (coherence)
     -- "take the conjugate of their pattern" = z*conj(pilot)
  4. per data symbol, tone correlations z_j = FFT(symbol)[j]
     - non-coherent LLR: s_j = |z_j|
     - coherent   LLR: s_j = Re(z_j * exp(-j*phi))
  5. ft8_lib LDPC+CRC on the 174 LLRs (decode_llr)

Compares decode sets: (2,2) direct, refine+non-coherent, refine+coherent, (4,4).

Usage:  python3.14 coherent/coherent_refine.py ../raw.dat [secs]
"""
import os
import sys
import numpy as np
import ft8decode
from gnlh import open_recording
from ft8 import extract_band, ft8_dial_comp_hz

COARSE   = int(os.environ.get("COARSE", "2"))    # coarse-detect osr (1 or 2)
MINSCORE = int(os.environ.get("MINSCORE", "10"))  # coarse Costas sync threshold

SR       = 12800
NSPS     = 2048            # samples/symbol (SR*0.16)
NSYM     = 79
FRAME    = NSYM * NSPS
DIAL     = 300.0
COSTAS   = np.array([3, 1, 4, 0, 6, 5, 2])
COS_POS  = [0, 36, 72]
GRAY     = np.array([0, 1, 3, 2, 5, 6, 4, 7])
# data-symbol frame positions (skip the 3 Costas blocks)
DATA_POS = [k + (7 if k < 29 else 14) for k in range(58)]

# Costas pilot positions/tones across the frame (21 symbols).
CPOS  = np.array([p + k for p in COS_POS for k in range(7)])
CTONE = np.array([COSTAS[k] for _ in COS_POS for k in range(7)])


def sym_tones(frame):
    """[79,8] complex tone correlations = FFT of each symbol, bins 0..7."""
    Y = frame[:FRAME].reshape(NSYM, NSPS)
    return np.fft.fft(Y, axis=1)[:, :8]


def costas_energy(frame):
    # Only the 21 pilot symbols are needed for sync — FFT just those (4x faster).
    seg = frame[:FRAME].reshape(NSYM, NSPS)[CPOS]        # [21, NSPS]
    F = np.fft.fft(seg, axis=1)
    return np.abs(F[np.arange(len(CPOS)), CTONE]).sum()


def refine(bb, start, f, dt_max=int(os.environ.get("DTMAX", "512")), dt_step=64, f_span=3.5, nf=13):
    """Joint fine time+freq sync. Centers on candidate freq f, then searches a
    sub-symbol time shift and residual frequency to maximize Costas energy.
    Precise timing is essential: a 512-sample error rotates tone phase by radians
    and destroys coherence. Returns the aligned complex frame (or None)."""
    nn = np.arange(FRAME)
    best_e, best_frame = -1.0, None
    for dt in range(-dt_max, dt_max + 1, dt_step):
        s = start + dt
        if s < 0 or s + FRAME > len(bb):
            continue
        base = bb[s:s + FRAME] * np.exp(-2j * np.pi * f * nn / SR)   # center on candidate
        for df in np.linspace(-f_span, f_span, nf):
            y = base * np.exp(-2j * np.pi * df * nn / SR)
            e = costas_energy(y)
            if e > best_e:
                best_e, best_frame = e, y
    return best_frame


def phase_track(Z):
    """Linear carrier-phase phi(p) from the 3 Costas blocks (conjugate pilots)."""
    bp, bphi = [], []
    for p in COS_POS:
        acc = 0j
        for k in range(7):
            acc += Z[p + k, COSTAS[k]]     # pilot already carries phi; sum is coherent
        bp.append(p + 3); bphi.append(np.angle(acc))
    bphi = np.unwrap(bphi)
    b, a = np.polyfit(bp, bphi, 1)
    return a + b * np.arange(NSYM)


def llr174(Z, phi=None):
    """174 max-log Gray LLRs. phi=None -> non-coherent (|z|); else coherent Re(z e^-jphi)."""
    log = np.zeros(174, np.float32)
    for k, p in enumerate(DATA_POS):
        z = Z[p]
        s = np.abs(z) if phi is None else np.real(z * np.exp(-1j * phi[p]))
        s2 = s[GRAY]
        log[3*k+0] = max(s2[4], s2[5], s2[6], s2[7]) - max(s2[0], s2[1], s2[2], s2[3])
        log[3*k+1] = max(s2[2], s2[3], s2[6], s2[7]) - max(s2[0], s2[1], s2[4], s2[5])
        log[3*k+2] = max(s2[1], s2[3], s2[5], s2[7]) - max(s2[0], s2[2], s2[4], s2[6])
    return log


def coarse_candidates(audio, win_s=15.0, step_s=5.0):
    """Unique (freq_hz, global_time_sec) candidates from windowed (2,2) find."""
    W, STEP = int(win_s * SR), int(step_s * SR)
    seen = []
    for start in range(0, max(1, len(audio) - W), STEP):
        for c in ft8decode.find_candidates(audio[start:start+W], SR, COARSE, COARSE, 200.0, 3400.0, MINSCORE, 600):
            f, t = c["freq_hz"], start / SR + c["time_sec"]
            if not any(abs(f-ff) < 4 and abs(t-tt) < 0.5 for ff, tt in seen):
                seen.append((f, t))
    return seen


def main(path, secs=0.0):
    rec = open_recording(path)
    sr_in = rec["sample_rate"]
    x = rec["samples"]
    if secs:
        x = x[:int(secs * sr_in)]
    band = os.environ.get("BAND", "20m")
    bb, sr = extract_band(np.asarray(x), sr_in, ft8_dial_comp_hz(band) - DIAL, decim=32)
    audio = np.real(bb).astype(np.float32)
    print(f"{band}: {len(bb)/SR:.1f} s\n")

    cands = coarse_candidates(audio)
    print(f"coarse ({COARSE},{COARSE}) min_score={MINSCORE} candidates: {len(cands)}")

    nc_set, co_set = set(), set()
    for f, t in cands:
        start = int(round(t * SR))
        frame = refine(bb, start, f)
        if frame is None:
            continue
        Z = sym_tones(frame)
        phi = phase_track(Z)
        m_nc = ft8decode.decode_llr(llr174(Z, None))
        m_co = ft8decode.decode_llr(llr174(Z, phi))
        if m_nc:
            nc_set.add(m_nc)
        if m_co:
            co_set.add(m_co)

    # references
    direct22 = set(m["text"] for start in range(0, max(1, len(audio)-15*SR), 5*SR)
                   for m in ft8decode.decode_audio(audio[start:start+15*SR], SR, 2, 2, 200., 3400.))
    direct44 = set(m["text"] for start in range(0, max(1, len(audio)-15*SR), 5*SR)
                   for m in ft8decode.decode_audio(audio[start:start+15*SR], SR, 4, 4, 200., 3400.))

    print(f"\n(2,2) direct decode : {len(direct22)}")
    print(f"(4,4) direct decode : {len(direct44)}   <- brute-force target")
    print(f"refine + non-coherent: {len(nc_set)}")
    print(f"refine + coherent    : {len(co_set)}")

    both = nc_set & direct44
    only_ref = nc_set - direct44
    only_44 = direct44 - nc_set
    print(f"\nrefine(non-coh) vs (4,4):  both={len(both)}  "
          f"only refine={len(only_ref)}  only (4,4)={len(only_44)}")
    union = nc_set | direct44
    print(f"union (either method): {len(union)}   refine recall of (4,4): "
          f"{len(both)}/{len(direct44)} = {100*len(both)/max(1,len(direct44)):.0f}%")
    if only_44:
        print("(4,4) gets, refine misses:")
        for t in sorted(only_44):
            print(f"    - {t}")
    if only_ref:
        print("refine gets, (4,4) misses:")
        for t in sorted(only_ref)[:12]:
            print(f"    + {t}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "../raw.dat"
    s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    main(p, s)
