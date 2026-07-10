"""De-chirp proof: does linear frequency drift cost FT8 decodes, and does a
de-chirp refine recover them?

Controlled synthesis (no capture needed):
  1. encode a real message  -> 79 FT8 tones (ft8decode.encode)
  2. synthesize complex baseband at 12.8 kHz, continuous phase
  3. inject a KNOWN linear drift  f(t) = f0 + tone*6.25 + drift*t   (Hz)
  4. add complex AWGN at a target SNR (2500 Hz convention)
  5. decode two ways:
       BASELINE  = fixed-df refine (identical to gm/hf/refine.cc today)
       DECHIRP   = refine that also searches a linear slope (df + slope*t)
  6. count decodes == the message, over N noise trials, per (SNR, drift)

If BASELINE collapses past ~0.5 Hz/s and DECHIRP holds, drift is the miss cause
and de-chirp is the fix.  Usage:  python3.14 coherent/dechirp_test.py
"""
import numpy as np
import ft8decode

SR    = 12800
NSPS  = 2048
NSYM  = 79
FRAME = NSYM * NSPS
TONE  = 6.25
F0    = 1500.0                      # signal placed here in the baseband
COSTAS  = np.array([3, 1, 4, 0, 6, 5, 2])
COS_POS = [0, 36, 72]
GRAY    = np.array([0, 1, 3, 2, 5, 6, 4, 7])
DATA_POS = [k + (7 if k < 29 else 14) for k in range(58)]
CPOS  = np.array([p + k for p in COS_POS for k in range(7)])
CTONE = np.array([COSTAS[k] for _ in COS_POS for k in range(7)])
FRAME_SEC = FRAME / SR
# Global sample index + time for each of the 21 pilot symbols (vectorized search).
_PILOT_IDX = CPOS[:, None] * NSPS + np.arange(NSPS)[None, :]     # [21, 2048]
_PILOT_T   = _PILOT_IDX / SR
_PR        = np.arange(len(CPOS))


def costas_energy(frame):
    seg = frame[:FRAME].reshape(NSYM, NSPS)[CPOS]
    F = np.fft.fft(seg, axis=1)
    return np.abs(F[np.arange(len(CPOS)), CTONE]).sum()


def sym_tones(frame):
    Y = frame[:FRAME].reshape(NSYM, NSPS)
    return np.fft.fft(Y, axis=1)[:, :8]


def llr174(Z):
    log = np.zeros(174, np.float32)
    for k, p in enumerate(DATA_POS):
        s2 = np.abs(Z[p])[GRAY]
        log[3*k+0] = max(s2[4], s2[5], s2[6], s2[7]) - max(s2[0], s2[1], s2[2], s2[3])
        log[3*k+1] = max(s2[2], s2[3], s2[6], s2[7]) - max(s2[0], s2[1], s2[4], s2[5])
        log[3*k+2] = max(s2[1], s2[3], s2[5], s2[7]) - max(s2[0], s2[2], s2[4], s2[6])
    return log


def refine_baseline(bb, start, f, dt_max=512, dt_step=64, f_span=3.5, nf=13):
    """Fixed-df fine align (mirrors gm/hf/refine.cc: one df for the whole frame)."""
    nn = np.arange(FRAME)
    best_e, best = -1.0, None
    for dt in range(-dt_max, dt_max + 1, dt_step):
        s = start + dt
        if s < 0 or s + FRAME > len(bb):
            continue
        base = bb[s:s + FRAME] * np.exp(-2j * np.pi * f * nn / SR)
        for df in np.linspace(-f_span, f_span, nf):
            y = base * np.exp(-2j * np.pi * df * nn / SR)
            e = costas_energy(y)
            if e > best_e:
                best_e, best = e, y
    return best


def refine_dechirp(bb, start, f, dt_max=512, dt_step=64,
                   f_span=3.5, nf=13, slope_max=5.0, ns=21):
    """Fixed-df align, then a joint (df, slope) search. Instantaneous correction
    freq = df + slope*t, so phase gains a 0.5*slope*t^2 term."""
    nn = np.arange(FRAME)
    t = nn / SR
    # stage 1: best dt at df=0, slope=0
    best_e, best_base = -1.0, None
    for dt in range(-dt_max, dt_max + 1, dt_step):
        s = start + dt
        if s < 0 or s + FRAME > len(bb):
            continue
        base = bb[s:s + FRAME] * np.exp(-2j * np.pi * f * nn / SR)
        e = costas_energy(base)
        if e > best_e:
            best_e, best_base = e, base
    if best_base is None:
        return None, 0.0
    # stage 2: joint (df, slope) search, vectorized over slope. Only the 21 pilot
    # symbols drive Costas energy, so correct + FFT just those.
    P = best_base.reshape(NSYM, NSPS)[CPOS]              # [21, 2048]
    tt = _PILOT_T                                        # [21, 2048]
    slopes = np.linspace(-slope_max, slope_max, ns)
    quad = 0.5 * slopes[:, None, None] * (tt * tt)[None]  # [ns, 21, 2048]
    best_e, best_df, best_slope = -1.0, 0.0, 0.0
    for df in np.linspace(-f_span, f_span, nf):
        ph = 2 * np.pi * (df * tt[None] + quad)          # [ns, 21, 2048]
        F = np.fft.fft(P[None] * np.exp(-1j * ph), axis=2)
        e = np.abs(F[:, _PR, CTONE]).sum(axis=1)         # [ns]
        j = int(e.argmax())
        if e[j] > best_e:
            best_e, best_df, best_slope = e[j], df, slopes[j]
    ph = 2 * np.pi * (best_df * t + 0.5 * best_slope * t * t)
    return best_base * np.exp(-1j * ph), best_slope


def snr_to_sigma(snr_db, amp=1.0):
    # SNR in 2500 Hz: S/(N0*2500), N0 = sigma^2/SR (complex, two-sided).
    # SNR_lin = amp^2 * SR / (2500 * sigma^2)  ->  sigma = sqrt(amp^2*SR/(2500*SNR_lin))
    snr_lin = 10 ** (snr_db / 10.0)
    return np.sqrt(amp * amp * SR / (2500.0 * snr_lin))


def synth(tones, drift, sigma, rng, pad=1024):
    """Complex baseband: continuous-phase 8-FSK at F0 + tone*6.25, plus a global
    linear drift chirp, plus complex AWGN. Signal sits at [pad, pad+FRAME)."""
    fk = F0 + np.repeat(tones.astype(float), NSPS) * TONE     # inst. tone freq
    phase = 2 * np.pi * np.cumsum(fk) / SR
    sig = np.exp(1j * phase).astype(np.complex128)
    t = np.arange(FRAME) / SR
    sig *= np.exp(1j * 2 * np.pi * 0.5 * drift * t * t)       # d/dt(0.5*drift*t^2)=drift*t
    bb = np.zeros(FRAME + 2 * pad, np.complex128)
    bb[pad:pad + FRAME] = sig
    bb += (rng.standard_normal(bb.shape) + 1j * rng.standard_normal(bb.shape)) * (sigma / np.sqrt(2))
    return bb, pad


def decode_frame(frame):
    if frame is None:
        return None
    return ft8decode.decode_llr(llr174(sym_tones(frame)))


def main():
    msg = "CQ KF0RRR EN10"
    tones = ft8decode.encode(msg)
    if tones is None:
        raise SystemExit("encode failed")
    tones = np.asarray(tones)
    print(f"message: {msg!r}  ({len(tones)} tones, sync={list(tones[:7])})")

    # Sanity: clean (no drift, no noise) must round-trip through both refines.
    rng = np.random.default_rng(0)
    bb, pad = synth(tones, 0.0, 0.0, rng)
    canon = decode_frame(refine_baseline(bb, pad, F0))
    canon_dc = decode_frame(refine_dechirp(bb, pad, F0)[0])
    print(f"clean round-trip: baseline={canon!r}  dechirp={canon_dc!r}")
    if canon is None:
        raise SystemExit("clean baseline round-trip failed — synthesis/refine bug")
    print(f"drift budget (fixed-df, ~half bin end-to-end): "
          f"~{TONE/FRAME_SEC:.2f} Hz/s\n")

    SNRS   = [-6, -10, -13]
    DRIFTS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    NTRIAL = 24

    for snr in SNRS:
        sigma = snr_to_sigma(snr)
        print(f"=== SNR {snr:+d} dB (2500 Hz), {NTRIAL} trials/point ===")
        print(f"{'drift Hz/s':>10} | {'baseline':>9} | {'dechirp':>9} | recovered")
        print("-" * 48)
        for drift in DRIFTS:
            b_ok = d_ok = 0
            for tr in range(NTRIAL):
                rng = np.random.default_rng(1000 + tr)
                bb, pad = synth(tones, drift, sigma, rng)
                if decode_frame(refine_baseline(bb, pad, F0)) == canon:
                    b_ok += 1
                if decode_frame(refine_dechirp(bb, pad, F0)[0]) == canon:
                    d_ok += 1
            rec = d_ok - b_ok
            bar = "+" * max(0, rec)
            print(f"{drift:>10.2f} | {b_ok:>4}/{NTRIAL:<4} | {d_ok:>4}/{NTRIAL:<4} | {bar}")
        print()


if __name__ == "__main__":
    main()
