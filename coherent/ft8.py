"""FT8 front end in numpy: extract a band from the composite, STFT with explicit
time/frequency overlap, and Costas sync.  The overlap knobs (time_osr, freq_osr)
are the whole point of the experiment.

FT8: 8-FSK, 6.25 baud (0.16 s/symbol), 6.25 Hz tone spacing, 79 symbols, Costas
{3,1,4,0,6,5,2} at symbol positions 0/36/72.
"""
import numpy as np
from scipy.signal import resample_poly
from gnlh import band_layout, FT8_DIAL, HF_BANDS

COSTAS      = np.array([3, 1, 4, 0, 6, 5, 2])
COSTAS_POS  = (0, 36, 72)
NSYM        = 79
SYM_SEC     = 0.16          # 6.25 baud


def ft8_dial_comp_hz(band):
    """Composite-stream frequency (Hz) of a band's FT8 dial."""
    ws  = {n: s for n, s, _ in HF_BANDS}[band]
    cb  = {n: c for n, c, _, _ in band_layout()}[band]
    return (cb + (FT8_DIAL[band] - ws)) * 100.0


def extract_band(samples, sr, f0_hz, decim=32):
    """Mix f0_hz to DC and decimate -> complex baseband; returns (bb, new_sr).
    FT8 signals then sit at 0..~3 kHz (audio above the dial)."""
    n  = np.arange(len(samples), dtype=np.float64)
    bb = np.asarray(samples, dtype=np.complex64) * np.exp(-2j * np.pi * f0_hz * n / sr)
    bb = resample_poly(bb, 1, decim).astype(np.complex64)
    return bb, sr / decim


def stft(x, nsps, time_osr, freq_osr):
    """Magnitude STFT. hop = nsps/time_osr (time overlap), nfft = nsps*freq_osr
    (frequency overlap -> finer bins)."""
    nfft = nsps * freq_osr
    hop  = nsps // time_osr
    nfr  = 1 + (len(x) - nfft) // hop
    win  = np.hanning(nfft).astype(np.float32)
    spec = np.empty((nfr, nfft), np.float32)
    for i in range(nfr):
        seg = x[i * hop: i * hop + nfft] * win
        spec[i] = np.abs(np.fft.fft(seg))
    return spec


def costas_sync(spec, time_osr, freq_osr, fmax_hz, bin_hz):
    """Vectorized Costas sync over all (time_offset, base_freq).  Returns a
    normalized score array [t0, f0]; score = summed Costas-tone magnitude over
    the 21 sync symbols, divided by the spectrogram median (scale-robust across
    osr)."""
    nfr, nfft = spec.shape
    tmax = 7 * freq_osr                          # tone span in bins
    f0max = min(int(fmax_hz / bin_hz), nfft - tmax)
    last  = (NSYM - 1) * time_osr
    t0max = nfr - last
    if t0max <= 0 or f0max <= 0:
        return np.zeros((0, 0)), f0max
    score = np.zeros((t0max, f0max), np.float32)
    for pos in COSTAS_POS:
        for k in range(7):
            tr = (pos + k) * time_osr
            fb = int(COSTAS[k]) * freq_osr
            score += spec[tr:tr + t0max, fb:fb + f0max]
    med = np.median(spec) + 1e-9
    return score / (21.0 * med), f0max


def find_candidates(score, time_osr, freq_osr, bin_hz, hop_sec,
                    thresh, min_sep_hz=6.25, min_sep_sec=0.5):
    """Non-max suppression on the sync score -> candidate list of dicts
    {freq_hz, time_sec, snr}. freq_hz is the baseband audio freq (osr-invariant)."""
    if score.size == 0:
        return []
    fsep = max(1, int(min_sep_hz / bin_hz))
    tsep = max(1, int(min_sep_sec / hop_sec))
    order = np.argsort(score, axis=None)[::-1]
    taken = np.zeros(score.shape, bool)
    out = []
    for idx in order:
        t0, f0 = np.unravel_index(idx, score.shape)
        s = score[t0, f0]
        if s < thresh:
            break
        if taken[max(0, t0 - tsep):t0 + tsep + 1, max(0, f0 - fsep):f0 + fsep + 1].any():
            continue
        taken[t0, f0] = True
        out.append(dict(freq_hz=f0 * bin_hz, time_sec=t0 * hop_sec, snr=float(s)))
    return out
