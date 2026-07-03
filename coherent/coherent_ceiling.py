"""Coherent ceiling experiment for FT8.

FT8's modulation index is h = df*T = 6.25*0.16 = 1.0.  With h=1 the continuous
phase returns to the same value (mod 2pi) at every symbol boundary regardless of
data, so there is NO data-dependent phase state to run a Viterbi trellis over.
What is left is (a) coherent matched filtering against the real GFSK pulse vs the
pipeline's rectangular-window FFT, and (b) BT=2.0 GFSK ISI (pulse spans 3 symbols).

Before building any trellis, measure the ceiling: on synthetic AWGN, with timing,
carrier phase and neighbour symbols all GENIE-known, does ideal coherent detection
beat non-coherent at all?  Three detectors, identical everywhere except the metric:

  A  non-coherent, rectangular FFT     |FFT(rx_k)[m]|            (what the pipeline does)
  B  non-coherent, GFSK matched filter |<rx_k, ref_km>|         (pulse gain only)
  C  coherent,     GFSK matched filter  Re<rx_k, ref_km>        (pulse + coherence)

C-vs-B is the pure coherence gain; B-vs-A is the matched-filter (pulse) gain.
If genie-aided C barely beats A, no phase estimator or trellis can help the real
system, and we stop.  The horizontal gap between the decode-rate curves is the answer.

Usage:  python3.14 coherent/coherent_ceiling.py [n_msgs_per_snr]
"""
import sys
import numpy as np
from scipy.special import erf
import ft8decode

SR      = 12800
NSPS    = 2048
NSYM    = 79
BT      = 2.0
GRAY    = np.array([0, 1, 3, 2, 5, 6, 4, 7])
DATA_POS = [k + (7 if k < 29 else 14) for k in range(58)]

MESSAGES = [
    "CQ KF0RRR EM48", "CQ DX W1AW FN31", "K1ABC W9XYZ EN37", "W9XYZ K1ABC RRR",
    "KF0RRR N0CALL -12", "CQ TEST VE3XYZ FN25", "JA1XYZ KF0RRR 73", "G4ABC F5XYZ IO91",
]
MESSAGES = [m for m in MESSAGES if ft8decode.encode(m) is not None] or [
    "CQ KF0RRR EM48",
]


def gfsk_pulse(nsps=NSPS, bt=BT):
    """WSJT-X GFSK frequency pulse, spans 3 symbols."""
    k = np.pi * np.sqrt(2.0 / np.log(2.0))
    t = np.arange(3 * nsps) / nsps - 1.5
    return 0.5 * (erf(k * bt * (t + 0.5)) - erf(k * bt * (t - 0.5)))

PULSE = gfsk_pulse()
P0, P1, P2 = PULSE[:NSPS], PULSE[NSPS:2*NSPS], PULSE[2*NSPS:3*NSPS]
DPHI_PEAK = 2.0 * np.pi / NSPS          # h = 1


def modulate(tones, phi0=0.0):
    """79 tones -> complex baseband core (NSYM*NSPS samples), continuous-phase GFSK.
    Returns (core_wave, phi_pre) where phi_pre[k] is the exact phase at the sample
    just before symbol k's core interval (genie reference for coherent detection)."""
    n = len(tones)
    dphi = np.zeros((n + 2) * NSPS)
    for i, m in enumerate(tones):
        dphi[i*NSPS : i*NSPS + 3*NSPS] += DPHI_PEAK * m * PULSE
    phi = phi0 + np.cumsum(dphi)
    core = np.exp(1j * phi[NSPS : (n + 1) * NSPS])
    # phase just before each symbol-k core interval (padded index (k+1)*NSPS - 1)
    phi_pre = phi[np.arange(n) * NSPS + (NSPS - 1)]
    return core.astype(np.complex64), phi_pre


COSTAS   = np.array([3, 1, 4, 0, 6, 5, 2])
COS_POS  = (0, 36, 72)


def local_ref0(tones, k, m):
    """GFSK reference for symbol k = tone m (neighbours genie-known), starting at
    phase 0.  The true start phase phi_pre[k] (or an estimate) is applied later as
    a single rotation, so this matched filter is computed once per (k,m)."""
    tl = tones[k-1] if k > 0 else 0
    tr = tones[k+1] if k < len(tones) - 1 else 0
    dph = DPHI_PEAK * (tl * P2 + m * P1 + tr * P0)
    return np.exp(1j * np.cumsum(dph))


def llr174(S):
    """[79,8] real tone metrics -> 174 max-log Gray LLRs."""
    log = np.zeros(174, np.float32)
    for k, p in enumerate(DATA_POS):
        s2 = S[p][GRAY]
        log[3*k+0] = max(s2[4],s2[5],s2[6],s2[7]) - max(s2[0],s2[1],s2[2],s2[3])
        log[3*k+1] = max(s2[2],s2[3],s2[6],s2[7]) - max(s2[0],s2[1],s2[4],s2[5])
        log[3*k+2] = max(s2[1],s2[3],s2[5],s2[7]) - max(s2[0],s2[2],s2[4],s2[6])
    return log


# Isolated-symbol matched filters (no neighbour knowledge): pulse tails from
# absent neighbours are just dropped, so the ref only depends on the candidate tone.
REF_ISO = np.array([np.exp(1j * np.cumsum(DPHI_PEAK * m * P1)) for m in range(8)])


def metrics(rx, tones):
    """Return A[79,8] rectangular-FFT magnitudes, corr0[79,8] complex GFSK matched-
    filter correlations (genie neighbours), and corrI[79,8] isolated-symbol (no
    neighbour knowledge) correlations — both at phase 0."""
    A = np.zeros((NSYM, 8))
    corr0 = np.zeros((NSYM, 8), np.complex128)
    corrI = np.zeros((NSYM, 8), np.complex128)
    for k in range(NSYM):
        seg = rx[k*NSPS : (k+1)*NSPS]
        A[k] = np.abs(np.fft.fft(seg)[:8])                  # rectangular FFT
        for m in range(8):
            corr0[k, m] = np.vdot(local_ref0(tones, k, m), seg)  # genie neighbours
            corrI[k, m] = np.vdot(REF_ISO[m], seg)               # isolated symbol
    return A, corr0, corrI


def estimate_phi0(corr0):
    """Single carrier phase from the 21 Costas pilots.  With h=1 the per-symbol
    start phase is constant (= phi0) mod 2pi, so one estimate serves the frame."""
    acc = 0j
    for p in COS_POS:
        for j in range(7):
            acc += corr0[p + j, COSTAS[j]]
    return np.angle(acc)


def run(n_msgs):
    rng = np.random.default_rng(12345)
    # Per-symbol matched-filter SNR (Es/N0): Es = NSPS (|core|=1), N0 = sigma^2.
    # sigma = sqrt(NSPS / 10^(esn0/10)).  The FT8 waterfall sits low here because
    # of the 2048-sample/symbol processing gain; only the A/B/C gaps matter.
    esn0s = np.arange(-6, 9, 1.0)
    ok = {d: np.zeros(len(esn0s)) for d in "ABCDE"}

    def decode_all(A, corr0, corrI, phi_pre):
        phi0 = estimate_phi0(corr0)     # genie-neighbour pilots
        phiI = estimate_phi0(corrI)     # isolated-symbol pilots
        return {
            "A": llr174(A),                                             # rect FFT, |.|
            "B": llr174(np.abs(corr0)),                                 # GFSK MF, |.|, genie nbrs
            "C": llr174(np.real(corr0 * np.exp(-1j*phi_pre[:, None]))), # coherent, genie phase+nbrs
            "D": llr174(np.real(corr0 * np.exp(-1j*phi0))),             # coherent, est phi0, genie nbrs
            "E": llr174(np.real(corrI * np.exp(-1j*phiI))),             # coherent, est phi0, no nbrs
        }

    # Canonical (noiseless) decode per message — scores the recovered codeword,
    # not the display string, so hashed-callsign normalization doesn't count as a miss.
    jobs = []
    for m in MESSAGES:
        tones = np.array(ft8decode.encode(m), dtype=int)
        core, phi = modulate(tones, 0.0)
        A, corr0, corrI = metrics(core, tones)
        canon = ft8decode.decode_llr(decode_all(A, corr0, corrI, phi)["C"])
        if canon:
            jobs.append((tones, canon))

    trials = 0
    for t in range(n_msgs):
        tones, canon = jobs[t % len(jobs)]
        phi0 = rng.uniform(0, 2*np.pi)
        core, phi_pre = modulate(tones, phi0)
        for si, esn0 in enumerate(esn0s):
            sigma = np.sqrt(NSPS / 10 ** (esn0 / 10.0))
            noise = (rng.standard_normal(len(core)) + 1j*rng.standard_normal(len(core))) * (sigma/np.sqrt(2))
            rx = (core + noise).astype(np.complex64)
            A, corr0, corrI = metrics(rx, tones)
            llrs = decode_all(A, corr0, corrI, phi_pre)
            for d in "ABCDE":
                ok[d][si] += ft8decode.decode_llr(llrs[d]) == canon
        trials += 1

    print(f"\nFT8 coherent ceiling — {trials} msgs/Es-N0, genie timing\n")
    print(f"{'Es/N0':>6} | {'A rectFFT':>10} {'B MF|.|':>8} {'C genie':>8} "
          f"{'D estph':>8} {'E est,noNbr':>11}")
    print("-" * 60)
    for si, esn0 in enumerate(esn0s):
        print(f"{esn0:6.0f} | {ok['A'][si]/trials:10.2f} {ok['B'][si]/trials:8.2f} "
              f"{ok['C'][si]/trials:8.2f} {ok['D'][si]/trials:8.2f} {ok['E'][si]/trials:11.2f}")

    def thresh(o):   # Es/N0 at 50% decode (linear interp)
        r = o / trials
        for i in range(1, len(r)):
            if r[i-1] < 0.5 <= r[i]:
                return esn0s[i-1] + (0.5 - r[i-1]) / (r[i] - r[i-1])
        return np.nan
    tA, tB, tC, tD, tE = (thresh(ok[d]) for d in "ABCDE")
    print(f"\n50% decode Es/N0:  A={tA:.2f}  B={tB:.2f}  C={tC:.2f}  D={tD:.2f}  E={tE:.2f} dB")
    print(f"pulse/ISI gain      (A->B): {tA-tB:+.2f} dB   <- what a trellis buys")
    print(f"coherence ceiling   (A->C): {tA-tC:+.2f} dB   <- genie phase+neighbours")
    print(f"coherent, est phi0  (A->D): {tA-tD:+.2f} dB   <- genie neighbours only")
    print(f"coherent, realistic (A->E): {tA-tE:+.2f} dB   <- est phi0, no neighbour knowledge")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
