"""Timing-sensitivity experiment — the last gate before a coherent refine mode.

coherent_ceiling.py showed the coherent gain (~1.5 dB, detector E) with GENIE
timing.  Coherent detection is phase-sensitive, and a timing error rotates each
tone's phase by 2*pi*(m*6.25)*tau/SR, so the open question is how much of that
gain survives estimating symbol timing the way the pipeline does: a Costas-energy
(non-coherent) search over a grid.

Model: place the signal at a random fractional offset (so the true optimum never
lands on the grid, as in reality), search integer slice offsets over +/-256 samples
maximizing Costas energy at a given step, detect at the winner.  Compare:

  A  non-coherent rect-FFT          (baseline; timing-robust)
  E  coherent isolated-symbol MF + estimated phi0   (the candidate coherent mode)

All modes share a coarse locate (step 64 over +/-256); the mode sets the FINE step
(2 ~= genie, 32, 64=what refine uses today), i.e. the residual timing error.
If E's advantage over A collapses at step 64, a coherent mode needs a much finer
timing lock to be worth wiring in.

Usage:  python3.14 coherent/coherent_timing.py [n_msgs]
"""
import sys
import numpy as np
import ft8decode
from coherent_ceiling import (NSPS, NSYM, modulate, REF_ISO, llr174,
                              estimate_phi0, COSTAS, COS_POS, MESSAGES)

FRAME = NSYM * NSPS
GUARD = 2 * NSPS               # padding around the frame for the search window
SEARCH = 256                   # +/- coarse offset the search covers (samples)


def frac_delay(x, d):
    """Delay complex signal x by d samples (fractional) via FFT phase ramp."""
    f = np.fft.fftfreq(len(x))
    return np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * f * d))


def costas_energy(rx, off):
    """Non-coherent Costas sync energy for a frame sliced at `off` (21 pilots)."""
    e = 0.0
    for p in COS_POS:
        for j in range(7):
            seg = rx[off + (p+j)*NSPS : off + (p+j)*NSPS + NSPS]
            e += abs(np.fft.fft(seg)[COSTAS[j]])
    return e


def _argmax_energy(rx, lo, hi, step):
    best_e, best = -1.0, lo
    for off in range(lo, hi + 1, step):
        e = costas_energy(rx, off)
        if e > best_e:
            best_e, best = e, off
    return best


def best_offset(rx, base, fine_step):
    """Coarse locate (step 64 over +/-SEARCH), then fine-refine (fine_step over
    +/-64 around the coarse winner) — mirrors how a real coherent refine would
    lock timing.  fine_step selects the residual: 2 ~= genie, 64 = refine today."""
    c = _argmax_energy(rx, base - SEARCH, base + SEARCH, 64)
    return _argmax_energy(rx, c - 64, c + 64, fine_step)


def detect(rx, off):
    """Return A (rect-FFT |.|) and corrI (isolated-symbol MF) metrics at `off`."""
    A = np.zeros((NSYM, 8))
    corrI = np.zeros((NSYM, 8), np.complex128)
    for k in range(NSYM):
        seg = rx[off + k*NSPS : off + k*NSPS + NSPS]
        A[k] = np.abs(np.fft.fft(seg)[:8])
        for m in range(8):
            corrI[k, m] = np.vdot(REF_ISO[m], seg)
    return A, corrI


# timing modes: (label, fine step).  All do the same coarse locate; the fine step
# sets the residual timing error (2 ~= genie, 64 = what refine uses today).
MODES = [("fine(2)", 2), ("step 32", 32), ("step 64", 64)]


def run(n_msgs):
    rng = np.random.default_rng(2027)
    esn0s = np.arange(2, 9, 1.0)
    # ok[mode][detector] -> array over esn0
    ok = {mi: {"A": np.zeros(len(esn0s)), "E": np.zeros(len(esn0s))} for mi in range(len(MODES))}

    jobs = []
    for m in MESSAGES:
        tones = np.array(ft8decode.encode(m), dtype=int)
        core, _ = modulate(tones, 0.0)
        A, corrI = detect(np.concatenate([np.zeros(GUARD, np.complex64), core,
                                          np.zeros(GUARD, np.complex64)]), GUARD)
        canon = ft8decode.decode_llr(llr174(np.real(corrI * np.exp(-1j*estimate_phi0(corrI)))))
        if canon:
            jobs.append((tones, canon))

    trials = 0
    for t in range(n_msgs):
        tones, canon = jobs[t % len(jobs)]
        core, _ = modulate(tones, rng.uniform(0, 2*np.pi))
        tau = rng.uniform(-200, 200)                    # random true offset (fractional)
        buf = np.concatenate([np.zeros(GUARD, np.complex128), core,
                              np.zeros(GUARD, np.complex128)])
        buf = frac_delay(buf, tau)                       # signal now centered at GUARD+tau
        for si, esn0 in enumerate(esn0s):
            sigma = np.sqrt(NSPS / 10 ** (esn0 / 10.0))
            noise = (rng.standard_normal(len(buf)) + 1j*rng.standard_normal(len(buf))) * (sigma/np.sqrt(2))
            rx = buf + noise
            for mi, (_, step) in enumerate(MODES):
                off = best_offset(rx, GUARD, step)
                A, corrI = detect(rx, off)
                ok[mi]["A"][si] += ft8decode.decode_llr(llr174(A)) == canon
                ok[mi]["E"][si] += ft8decode.decode_llr(
                    llr174(np.real(corrI * np.exp(-1j*estimate_phi0(corrI))))) == canon
        trials += 1

    def thresh(o):
        r = o / trials
        for i in range(1, len(r)):
            if r[i-1] < 0.5 <= r[i]:
                return esn0s[i-1] + (0.5 - r[i-1]) / (r[i] - r[i-1])
        return np.nan

    print(f"\nFT8 timing sensitivity — {trials} msgs/Es-N0, random +/-200-sample true offset\n")
    print(f"{'Es/N0':>6} |" + "".join(f"  {lbl+' A':>10} {lbl+' E':>10}" for lbl, _ in MODES))
    print("-" * (8 + 22*len(MODES)))
    for si, esn0 in enumerate(esn0s):
        row = f"{esn0:6.0f} |"
        for mi in range(len(MODES)):
            row += f"  {ok[mi]['A'][si]/trials:10.2f} {ok[mi]['E'][si]/trials:10.2f}"
        print(row)

    print("\n50% decode Es/N0 and coherent gain (A->E):")
    for mi, (lbl, _) in enumerate(MODES):
        tA, tE = thresh(ok[mi]["A"]), thresh(ok[mi]["E"])
        print(f"  {lbl:>10}:  A={tA:5.2f}  E={tE:5.2f}   gain={tA-tE:+.2f} dB")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
