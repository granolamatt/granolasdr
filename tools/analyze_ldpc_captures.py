#!/usr/bin/env python3
"""
Analyze LLR captures from granolasdr to diagnose GPU QP-ADMM decode gap.

Usage:
  # 1. Collect captures (run granolasdr without --gpu-ldpc):
  LDPC_CAPTURE=ldpc_captures.bin ./build/granolasdr [options]

  # 2. Analyze:
  cd /home/matt/workarea/qp-admm
  python /home/matt/workarea/granolasdr/tools/analyze_ldpc_captures.py ldpc_captures.bin

Each capture vector is a raw float32[174] FT8SoftCuda LLR array where:
  positive LLR = bit 1 likely  (FT8SoftCuda convention)

The script applies the same normalization as the CUDA kernel (scale to variance=24),
negates for the Python ADMM sign convention (positive = bit 0 likely), then runs
Python QP-ADMM at several (rho, max_iter) settings and reports convergence.

If Python ADMM converges on all vectors → GPU CUDA kernel has a bug or needs
more iterations. If Python ADMM also fails on many → algorithm limitation.
"""
import sys
import os
import numpy as np

# Must be run from inside /home/matt/workarea/qp-admm so qp_admm is importable.
try:
    from qp_admm.codes import ft8 as ft8_code
    from qp_admm.decoder import qp_admm, bp_baseline
except ImportError:
    print("ERROR: run this script from /home/matt/workarea/qp-admm/", file=sys.stderr)
    print("  cd /home/matt/workarea/qp-admm && python <this_script> <capture_file>", file=sys.stderr)
    sys.exit(1)

FTX_LDPC_N = 174


def normalize_llr(llr: np.ndarray) -> np.ndarray:
    """Apply the same variance=24 normalization as the CUDA kernel (for BP)."""
    var = np.var(llr)
    scale = np.sqrt(24.0 / var) if var > 1e-6 else 1.0
    return llr * scale


def normalize_llr_centered(llr: np.ndarray) -> np.ndarray:
    """Center (subtract mean) then normalize to var=24 for ADMM.

    FT8 codewords have ~22% ones / 78% zeros on average, giving raw LLR mean
    of roughly -2.5 (positive = bit 1).  After negation for ADMM (positive = bit 0),
    q_j values are biased toward +1.4, driving all x toward 0.  The trivial
    all-zeros codeword satisfies every parity check and becomes the LP attractor.
    Centering removes this bias so ADMM finds the true codeword.
    """
    centered = llr - np.mean(llr)
    var = np.var(centered)
    scale = np.sqrt(24.0 / var) if var > 1e-6 else 1.0
    return centered * scale


def bp_sum_product(llrs: np.ndarray, H_csr, H_syndrome,
                   max_iter: int = 25, return_iters: bool = False):
    """
    Sum-product BP matching ft8_lib's bp_decode algorithm exactly.
    Input sign convention: positive = bit 1 likely (ft8_lib / FT8SoftCuda convention).
    No negation needed — pass llrs_norm directly (not negated).
    """
    B, n = llrs.shape
    m = H_csr.shape[0]
    rows, cols = H_csr.nonzero()

    # Adjacency tables matching ft8_lib kFTX_LDPC_Nm / kFTX_LDPC_Mn layout
    check_edges = [[] for _ in range(m)]
    var_edges   = [[] for _ in range(n)]
    for e, (r, c) in enumerate(zip(rows, cols)):
        check_edges[r].append((e, c))
        var_edges[c].append((e, r))

    tov = np.zeros((B, n, 3), dtype=np.float64)   # check→var messages (3 per variable)
    var_edge_slot = np.full((n, 3), -1, dtype=np.int32)
    for j in range(n):
        for slot, (e, _) in enumerate(var_edges[j]):
            var_edge_slot[j, slot] = e

    iters = np.full(B, max_iter, dtype=np.int32)
    converged_mask = np.zeros(B, dtype=bool)
    best_x = np.zeros((B, n), dtype=np.int8)

    eps = 1e-10

    for it in range(max_iter):
        # Hard decision: positive LLR → bit 1 (ft8_lib convention)
        L_total = llrs.copy()
        for j in range(n):
            for slot in range(3):
                L_total[:, j] += tov[:, j, slot]
        x_hat = (L_total > 0).astype(np.int8)

        synd = (x_hat.astype(np.int32) @ H_syndrome.T.toarray()) % 2
        newly_ok = np.all(synd == 0, axis=1) & ~converged_mask
        if newly_ok.any():
            iters[newly_ok] = it + 1
            converged_mask |= newly_ok
            best_x[newly_ok] = x_hat[newly_ok]
        if converged_mask.all():
            break

        # v2c messages: L_total - own tov contribution
        # toc[check][slot] = tanh(-v2c / 2)
        toc = {}
        for i in range(m):
            edges = check_edges[i]
            d = len(edges)
            toc_row = np.zeros((B, d))
            for k, (e, j) in enumerate(edges):
                slot = next(s for s, (ee, _) in enumerate(var_edges[j]) if ee == e)
                v2c = L_total[:, j] - tov[:, j, slot]
                toc_row[:, k] = np.tanh(np.clip(-v2c / 2, -10, 10))
            toc[i] = toc_row

        # c2v messages: product of all other tanh values, then -2*atanh
        new_tov = np.zeros_like(tov)
        for j in range(n):
            for slot, (e, i) in enumerate(var_edges[j]):
                edges = check_edges[i]
                k = next(kk for kk, (ee, _) in enumerate(edges) if ee == e)
                prod = np.ones(B)
                for kk in range(len(edges)):
                    if kk != k:
                        prod *= toc[i][:, kk]
                prod = np.clip(prod, -1 + eps, 1 - eps)
                new_tov[:, j, slot] = -2 * np.arctanh(prod)
        tov = new_tov

    # Final hard decision for non-converged
    L_total = llrs.copy()
    for j in range(n):
        for slot in range(3):
            L_total[:, j] += tov[:, j, slot]
    x_hat = (L_total > 0).astype(np.int8)
    best_x[~converged_mask] = x_hat[~converged_mask]

    if return_iters:
        return best_x, iters
    return best_x


def run_configs(llrs_norm_negated: np.ndarray, llrs_norm: np.ndarray,
                H_csr, H_csc, H_syndrome):
    """Run ADMM/BP at several configs, return result tuples."""
    configs = [
        ("ADMM rho=1.0 iter=50",      "qp",  dict(rho=1.0, max_iter=50)),
        ("ADMM rho=1.0 iter=100",     "qp",  dict(rho=1.0, max_iter=100)),
        ("ADMM rho=1.0 iter=200",     "qp",  dict(rho=1.0, max_iter=200)),
        ("MinSum α=0.75 iter=25",     "bp",  dict(max_iter=25, alpha=0.75)),
        ("MinSum α=1.0  iter=25",     "bp",  dict(max_iter=25, alpha=1.0)),
        ("SumProduct iter=25",        "sp",  dict(max_iter=25)),
        ("SumProduct iter=50",        "sp",  dict(max_iter=50)),
    ]

    results = []
    B = llrs_norm_negated.shape[0]
    for label, decoder, kwargs in configs:
        if decoder == "qp":
            x_hat, iters = qp_admm.decode(
                llrs_norm_negated, H_csr, H_csc, H_syndrome,
                return_iters=True, **kwargs)
        elif decoder == "sp":
            # Sum-product uses positive=bit1 convention — pass un-negated
            max_iter = kwargs.get("max_iter", 25)
            x_hat, iters = bp_sum_product(
                llrs_norm, H_csr, H_syndrome, max_iter=max_iter, return_iters=True)
        else:
            x_hat, iters = bp_baseline.decode(
                llrs_norm_negated, H_csr, H_csc, H_syndrome,
                return_iters=True, **kwargs)

        # Parity check
        synd = (x_hat.astype(np.int32) @ H_syndrome.T.toarray()) % 2
        converged = np.all(synd == 0, axis=1)
        n_conv = int(converged.sum())
        mean_iters = float(iters[converged].mean()) if n_conv > 0 else float('nan')

        # All-zeros check: the zero codeword trivially satisfies every parity row
        # (row sum = 0, even). ADMM converging to all-zeros is a false positive.
        n_allzero = int(np.all(x_hat[converged] == 0, axis=1).sum()) if n_conv > 0 else 0
        n_allone  = int(np.all(x_hat[converged] == 1, axis=1).sum()) if n_conv > 0 else 0

        results.append((label, n_conv, B, mean_iters, n_allzero, n_allone))
    return results


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <capture_file.bin>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % FTX_LDPC_N != 0:
        print(f"WARNING: file size {raw.size} not divisible by {FTX_LDPC_N}, truncating", file=sys.stderr)
        raw = raw[:raw.size - (raw.size % FTX_LDPC_N)]

    llrs = raw.reshape(-1, FTX_LDPC_N)
    B = llrs.shape[0]
    print(f"Loaded {B} LLR vectors from {path}")

    if B == 0:
        print("No vectors to analyze.")
        return

    # Show LLR statistics
    all_var = np.var(llrs, axis=1)
    per_vec_mean = np.mean(llrs, axis=1)
    print(f"LLR variance:    min={all_var.min():.2f}  median={np.median(all_var):.2f}  max={all_var.max():.2f}")
    print(f"LLR magnitude:   mean={np.abs(llrs).mean():.3f}  max={np.abs(llrs).max():.3f}")
    print(f"LLR global mean: {np.mean(llrs):.4f}  (per-vector: min={per_vec_mean.min():.3f}  "
          f"median={np.median(per_vec_mean):.3f}  max={per_vec_mean.max():.3f})")

    # BP normalization: scale to var=24, keep mean (matches ft8_lib / CUDA BP path).
    # ADMM normalization: subtract per-vector mean FIRST, then scale to var=24.
    #   Reason: FT8 codewords have ~22% ones, giving raw LLR mean ≈ -2.5.
    #   After negation for ADMM, q_j mean ≈ +1.4, pushing all x toward 0.
    #   The all-zeros codeword trivially satisfies H*x=0 → ADMM LP attractor.
    #   Centering before negation prevents this.
    llrs_norm = np.array([normalize_llr(llr) for llr in llrs], dtype=np.float64)
    llrs_norm_centered = np.array([normalize_llr_centered(llr) for llr in llrs], dtype=np.float64)
    norm_var = np.var(llrs_norm, axis=1)
    norm_var_c = np.var(llrs_norm_centered, axis=1)
    print(f"After normalization (BP):    var mean={norm_var.mean():.2f}  llr mean={np.mean(llrs_norm):.3f}")
    print(f"After normalization (ADMM):  var mean={norm_var_c.mean():.2f}  llr mean={np.mean(llrs_norm_centered):.3f}  (centered)")

    # ---- Hard-decision sanity check (no decoding) ----
    # ft8_lib convention: positive LLR = bit 1.  Hard-decide on RAW and NORMALIZED LLRs.
    # Any correctly-formatted capture should have some fraction passing parity with
    # raw hard decisions alone (strong signals already form valid codewords).
    # ~0% with raw hard decisions = strong sign of sign or ordering problem.
    print("\n--- Hard-decision sanity (no LDPC decoding) ---")
    H_csr_tmp = ft8_code.build_parity_check_matrix()
    H_np = H_csr_tmp.toarray().astype(np.int32)
    for label, llr_arr in [("raw (un-normalized)", llrs), ("normalized", llrs_norm)]:
        bits = (llr_arr > 0).astype(np.int32)  # positive = bit 1 (ft8_lib convention)
        synd = (bits @ H_np.T) % 2
        n_pass = int(np.all(synd == 0, axis=1).sum())
        print(f"  Hard decision on {label}: {n_pass}/{B} pass parity ({100*n_pass/B:.1f}%)")
    print()

    llrs_for_admm = -llrs_norm_centered  # centered then negated: mean(q_j) ≈ 0

    # Build FT8 H matrix
    print("Building FT8 H matrix...", end=" ", flush=True)
    H_csr = ft8_code.build_parity_check_matrix()
    H_csc = H_csr.T.tocsc()
    H_syndrome = H_csr.astype(np.int8)
    print("done.")

    print(f"\nRunning {B} vectors through ADMM/BP (float64)...\n")
    results = run_configs(llrs_for_admm, llrs_norm, H_csr, H_csc, H_syndrome)

    print(f"{'Config':<25}  {'Conv':>5}  {'/ Total':>7}  {'Rate':>6}  {'Mean iters':>11}  {'AllZero':>8}  {'AllOne':>7}")
    print("-" * 84)
    for label, n_conv, total, mean_iters, n_allzero, n_allone in results:
        rate = 100.0 * n_conv / total if total > 0 else 0.0
        iters_str = f"{mean_iters:.1f}" if not np.isnan(mean_iters) else "n/a"
        flag = "  ← TRIVIAL" if n_conv > 0 and n_allzero == n_conv else ""
        print(f"{label:<25}  {n_conv:>5}  {total:>7}  {rate:>5.1f}%  {iters_str:>11}  {n_allzero:>8}{flag}  {n_allone:>7}")

    print()
    n_allzero_admm = results[0][4]  # first ADMM config
    n_conv_admm    = results[0][1]
    if n_conv_admm > 0 and n_allzero_admm == n_conv_admm:
        print("*** ALL ADMM 'convergences' are the trivial all-zeros codeword!")
        print("    This is a false-positive: ADMM objective is wrong (sign or scaling).")
    elif n_conv_admm > 0 and n_allzero_admm > n_conv_admm // 2:
        print(f"*** {n_allzero_admm}/{n_conv_admm} ADMM convergences are all-zeros (trivial codeword).")
        print("    Most ADMM 'decodes' are false positives.")

    print()
    print("Interpretation:")
    print("  SumProduct ~100%                     → ft8_lib algorithm confirmed correct")
    print("  MinSum << SumProduct                 → min-sum is a weak proxy for ft8_lib at this SNR")
    print("  ADMM AllZero == Conv                 → ADMM converging to trivial codeword (bug)")
    print("  ADMM rate close to SumProduct        → algorithm is fine; investigate GPU kernel")
    print("  ADMM rate << SumProduct              → algorithm limitation")


if __name__ == "__main__":
    main()
