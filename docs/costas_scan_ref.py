"""
Reference NumPy implementation of the FT8 / JS8 Costas sync scan kernel.

Matches:
  gm/cuda/FT8ScanCuda.cu   kCostas = {3,1,4,0,6,5,2}
  gm/cuda/JS8ScanCuda.cu   kCostas = {4,2,5,6,1,3,0}

Score metric — max-log 8-FSK
  For each of 21 Costas symbols (3 groups × 7):
      score += mag[expected_tone] - max(mag[other_7_tones])
  fscore = score / num_valid_symbols
  Emit candidate if fscore >= min_score

Vectorization strategy
  Outer dimensions (time_off=30, sub=n_sub, fo=num_bins) are fully vectorized
  via NumPy advanced indexing.  The 21-symbol loop is the only explicit loop —
  it has constant trip count and maps to the smem scan in the CUDA kernel.

Memory pressure per symbol iteration: 30 × n_sub × num_bins × 8 bytes (int32).
  n_sub=16, num_bins=65536  →  ~120 MB   (comfortable)
  n_sub=16, num_bins=1M     →  ~1.9 GB   (use costas_scan_chunked instead)
"""

import numpy as np
from typing import NamedTuple

# ── Costas tone arrays ────────────────────────────────────────────────────────
COSTAS_FT8        = np.array([3, 1, 4, 0, 6, 5, 2], dtype=np.int32)
COSTAS_JS8_NORMAL = np.array([4, 2, 5, 6, 1, 3, 0], dtype=np.int32)

_COSTAS_GROUP_STRIDE = 36   # symbol positions: 0, 36, 72
_N_COSTAS_GROUPS     = 3
_N_COSTAS_SYMBOLS    = _N_COSTAS_GROUPS * 7   # 21 total
_N_FSK_TONES         = 8


class ScanResult(NamedTuple):
    fo:       np.ndarray  # int32   [N]  frequency offset (bin index)
    time_off: np.ndarray  # uint8   [N]  time offset 0..29
    time_sub: np.ndarray  # uint8   [N]  time sub-sample
    freq_sub: np.ndarray  # uint8   [N]  frequency sub-sample
    score:    np.ndarray  # float32 [N]  mean max-log 8-FSK margin per symbol


# ── Symbol access plan ────────────────────────────────────────────────────────

def _build_symbol_plan(
    snap_start: int,
    num_blocks: int,
    ring_size:  int,
    kCostas:    np.ndarray,   # [7]
):
    """
    Precompute ring slot indices and validity for all (time_off, symbol) pairs.

    Returns
    -------
    ring_slot_flat : int32 [30, 21]  ring buffer row for each (time_off, symbol)
    valid_flat     : bool  [30, 21]  False when block_abs >= num_blocks
    sm_flat        : int32 [21]      expected tone bin per symbol
    """
    time_offs = np.arange(30,               dtype=np.int32)   # [30]
    ms        = np.arange(_N_COSTAS_GROUPS, dtype=np.int32)   # [3]
    ks        = np.arange(7,               dtype=np.int32)    # [7]

    # block_abs[time_off, m, k] = time_off + 36*m + k   shape [30, 3, 7]
    block_abs = (time_offs[:, None, None]
                 + _COSTAS_GROUP_STRIDE * ms[None, :, None]
                 + ks[None, None, :])

    valid     = block_abs < num_blocks                         # [30, 3, 7]
    safe_ba   = np.where(valid, block_abs, 0)
    ring_slot = (snap_start + safe_ba) % ring_size             # [30, 3, 7]

    return (
        ring_slot.reshape(30, _N_COSTAS_SYMBOLS),              # [30, 21]
        valid.reshape(30, _N_COSTAS_SYMBOLS),                  # [30, 21]
        np.tile(kCostas, _N_COSTAS_GROUPS),                   # [21]
    )


# ── Core scan ─────────────────────────────────────────────────────────────────

def costas_scan(
    mag:        np.ndarray,   # uint8 [ring_size, n_sub, num_bins]
    snap_start: int,
    num_blocks: int,
    time_osr:   int,
    freq_osr:   int,
    kCostas:    np.ndarray,   # [7]  values in 0..7
    min_score:  float = 3.0,
    max_cands:  int   = 2048,
) -> ScanResult:
    """
    Vectorized Costas sync scan.  Equivalent to ft8SyncScanKernel /
    js8SyncScanKernel.  The 21-symbol loop is the only explicit iteration;
    all (time_off, sub, fo) dimensions are vectorized.
    """
    ring_size, n_sub, num_bins = mag.shape
    assert n_sub == time_osr * freq_osr, "n_sub must equal time_osr * freq_osr"
    assert kCostas.shape == (7,)

    ring_slot_flat, valid_flat, sm_flat = _build_symbol_plan(
        snap_start, num_blocks, ring_size, kCostas)

    # Pad fo axis by 7 zeros so edge accesses (fo + tone, tone=0..7) are safe.
    mag_pad  = np.pad(mag, ((0, 0), (0, 0), (0, _N_FSK_TONES - 1)))

    sub_idx  = np.arange(n_sub,    dtype=np.int32)   # [n_sub]
    fo_idx   = np.arange(num_bins, dtype=np.int32)   # [num_bins]
    tone_idx = np.arange(_N_FSK_TONES, dtype=np.int32)  # [8]

    score     = np.zeros((30, n_sub, num_bins), dtype=np.int32)
    num_valid = np.zeros((30, n_sub, num_bins), dtype=np.int32)

    for sym in range(_N_COSTAS_SYMBOLS):
        sm  = int(sm_flat[sym])                       # expected tone (0–7)
        rs  = ring_slot_flat[:, sym]                  # [30]
        v   = valid_flat[:, sym]                      # [30] bool

        # all8[to, sub, fo, tone] = mag[rs[to], sub, fo + tone]
        # Advanced indexing broadcasts all four axes simultaneously.
        # Shape: [30, n_sub, num_bins, 8]
        all8 = mag_pad[
            rs[:, None, None, None],                  # [30, 1,    1,       1]
            sub_idx[None, :, None, None],             # [1,  n_sub, 1,       1]
            fo_idx[None, None, :, None] + tone_idx,  # [1,  1,    num_bins, 8]
        ].astype(np.int32)                            # [30, n_sub, num_bins, 8]

        cur        = all8[..., sm]                    # [30, n_sub, num_bins]
        best_other = all8[..., tone_idx != sm].max(axis=-1)  # [30, n_sub, num_bins]

        v3 = v[:, None, None]                         # [30, 1, 1]
        score     += v3 * (cur - best_other)
        num_valid += v3.astype(np.int32)

    # Normalize: num_valid >= 19 for all valid (time_off, num_blocks=106) pairs.
    fscore = np.where(
        num_valid > 0,
        score.astype(np.float32) / np.maximum(num_valid, 1).astype(np.float32),
        -np.inf,
    )

    # Mirror CUDA early-exit: fo > num_bins-8 is invalid (tone+7 would overflow).
    fscore[:, :, num_bins - (_N_FSK_TONES - 1):] = -np.inf

    # Collect and rank candidates.
    to_hit, sub_hit, fo_hit = np.nonzero(fscore >= min_score)
    scores_hit = fscore[to_hit, sub_hit, fo_hit]

    order = np.argsort(-scores_hit)[:max_cands]
    to_hit, sub_hit, fo_hit, scores_hit = (
        to_hit[order], sub_hit[order], fo_hit[order], scores_hit[order])

    return ScanResult(
        fo       = fo_hit.astype(np.int32),
        time_off = to_hit.astype(np.uint8),
        time_sub = (sub_hit // freq_osr).astype(np.uint8),
        freq_sub = (sub_hit  % freq_osr).astype(np.uint8),
        score    = scores_hit.astype(np.float32),
    )


# ── Chunked wrapper for large num_bins ────────────────────────────────────────

def costas_scan_chunked(
    mag:        np.ndarray,
    snap_start: int,
    num_blocks: int,
    time_osr:   int,
    freq_osr:   int,
    kCostas:    np.ndarray,
    min_score:  float = 3.0,
    max_cands:  int   = 2048,
    chunk_bins: int   = 65536,
) -> ScanResult:
    """
    costas_scan over large num_bins by processing fo in slices.

    Working memory per chunk: 30 × n_sub × chunk_bins × 8 × 4 bytes.
    Default chunk_bins=65536, n_sub=16 → ~120 MB.
    Full pipeline num_bins=1,048,576 splits into 16 chunks at that size.
    """
    _, _, num_bins = mag.shape
    parts: list[ScanResult] = []

    for start in range(0, num_bins, chunk_bins):
        end   = min(start + chunk_bins, num_bins)
        chunk = mag[:, :, start:end]
        res   = costas_scan(chunk, snap_start, num_blocks, time_osr, freq_osr,
                            kCostas, min_score=min_score, max_cands=max_cands)
        if len(res.fo):
            parts.append(ScanResult(
                fo       = res.fo + np.int32(start),
                time_off = res.time_off,
                time_sub = res.time_sub,
                freq_sub = res.freq_sub,
                score    = res.score,
            ))

    if not parts:
        z = np.empty(0, dtype=np.int32)
        return ScanResult(z, z.view(np.uint8), z.view(np.uint8),
                          z.view(np.uint8), z.view(np.float32))

    fo_all = np.concatenate([p.fo       for p in parts])
    to_all = np.concatenate([p.time_off for p in parts])
    ts_all = np.concatenate([p.time_sub for p in parts])
    fs_all = np.concatenate([p.freq_sub for p in parts])
    sc_all = np.concatenate([p.score    for p in parts])

    order = np.argsort(-sc_all)[:max_cands]
    return ScanResult(fo_all[order].astype(np.int32),
                      to_all[order], ts_all[order], fs_all[order],
                      sc_all[order])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inject_signal(mag, snap_start, num_blocks, kCostas,
                   ring_size, time_osr, freq_osr,
                   fo, time_off, time_sub, freq_sub, amplitude=200):
    """Write a clean Costas signal into the mag ring at a known position."""
    sub = time_sub * freq_osr + freq_sub
    for m in range(_N_COSTAS_GROUPS):
        for k in range(7):
            ba = time_off + _COSTAS_GROUP_STRIDE * m + k
            if ba >= num_blocks:
                continue
            rs = (snap_start + ba) % ring_size
            sm = int(kCostas[k])
            mag[rs, sub, fo + sm] = np.clip(amplitude, 0, 255)


def score_at(mag, snap_start, num_blocks, time_osr, freq_osr, kCostas,
             fo, time_off, time_sub, freq_sub):
    """
    Return the exact max-log 8-FSK score for a single candidate position.
    Useful for unit-testing and debugging individual results.
    """
    ring_size, n_sub, _ = mag.shape
    sub = time_sub * freq_osr + freq_sub
    total, n_valid = 0, 0
    for m in range(_N_COSTAS_GROUPS):
        for k in range(7):
            ba = time_off + _COSTAS_GROUP_STRIDE * m + k
            if ba >= num_blocks:
                continue
            rs = (snap_start + ba) % ring_size
            sm = int(kCostas[k])
            window = mag[rs, sub, fo:fo + _N_FSK_TONES].astype(np.int32)
            cur        = window[sm]
            best_other = max(window[j] for j in range(_N_FSK_TONES) if j != sm)
            total  += cur - best_other
            n_valid += 1
    return float(total) / n_valid if n_valid else -np.inf


# ── Demo / self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    ring_size  = 200
    num_blocks = 106    # FT8_CAPTURE_BLOCKS
    time_osr   = 4
    freq_osr   = 4
    n_sub      = time_osr * freq_osr   # 16
    num_bins   = 8192   # demo size; full pipeline uses 1,048,576 → use chunked
    snap_start = 7      # arbitrary non-zero ring offset

    signals = [
        dict(fo=512,  time_off=5,  time_sub=2, freq_sub=1, amplitude=220),
        dict(fo=3000, time_off=20, time_sub=0, freq_sub=3, amplitude=185),
        dict(fo=7800, time_off=29, time_sub=3, freq_sub=2, amplitude=160),
    ]

    for label, kCostas in [("FT8", COSTAS_FT8), ("JS8-Normal", COSTAS_JS8_NORMAL)]:
        print(f"\n{'─' * 58}")
        print(f"  {label}  kCostas = {kCostas.tolist()}")
        print(f"{'─' * 58}")

        mag = rng.integers(10, 30, size=(ring_size, n_sub, num_bins), dtype=np.uint8)

        for sig in signals:
            _inject_signal(mag, snap_start, num_blocks, kCostas,
                           ring_size, time_osr, freq_osr, **sig)

        result = costas_scan(mag, snap_start, num_blocks,
                             time_osr, freq_osr, kCostas, min_score=3.0)

        print(f"  Candidates found: {len(result.fo)}")
        for i in range(min(8, len(result.fo))):
            print(f"    fo={result.fo[i]:6d}  to={result.time_off[i]:2d}"
                  f"  ts={result.time_sub[i]}  fs={result.freq_sub[i]}"
                  f"  score={result.score[i]:.2f}")

        print()
        all_pass = True
        for sig in signals:
            mask  = ((result.fo       == sig['fo'])
                   & (result.time_off == sig['time_off'])
                   & (result.time_sub == sig['time_sub'])
                   & (result.freq_sub == sig['freq_sub']))
            found = bool(np.any(mask))

            # Cross-check with scalar reference
            ref_score = score_at(mag, snap_start, num_blocks,
                                 time_osr, freq_osr, kCostas,
                                 sig['fo'], sig['time_off'],
                                 sig['time_sub'], sig['freq_sub'])
            vec_score = float(result.score[mask][0]) if found else float('nan')
            match = found and abs(vec_score - ref_score) < 1e-4

            tag = "PASS" if match else "FAIL"
            if not match:
                all_pass = False
            print(f"  [{tag}] fo={sig['fo']:5d} to={sig['time_off']:2d}"
                  f" ts={sig['time_sub']} fs={sig['freq_sub']}"
                  f"  vec={vec_score:.2f}  ref={ref_score:.2f}")

        print(f"\n  {'All signals recovered and scores match.' if all_pass else 'FAILURES detected.'}")
