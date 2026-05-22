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
    """Apply the same variance=24 normalization as the CUDA kernel."""
    var = np.var(llr)
    scale = np.sqrt(24.0 / var) if var > 1e-6 else 1.0
    return llr * scale


def run_configs(llrs_norm_negated: np.ndarray, H_csr, H_csc, H_syndrome):
    """Run ADMM/BP at several configs, return (config_label, n_converged) list."""
    configs = [
        ("ADMM rho=1.0 iter=50",  "qp",  dict(rho=1.0, max_iter=50)),
        ("ADMM rho=1.0 iter=100", "qp",  dict(rho=1.0, max_iter=100)),
        ("ADMM rho=1.0 iter=200", "qp",  dict(rho=1.0, max_iter=200)),
        ("ADMM rho=0.5 iter=100", "qp",  dict(rho=0.5, max_iter=100)),
        ("ADMM rho=0.5 iter=200", "qp",  dict(rho=0.5, max_iter=200)),
        ("BP   iter=25",          "bp",  dict(max_iter=25)),
        ("BP   iter=50",          "bp",  dict(max_iter=50)),
    ]

    results = []
    B = llrs_norm_negated.shape[0]
    for label, decoder, kwargs in configs:
        if decoder == "qp":
            x_hat, iters = qp_admm.decode(
                llrs_norm_negated, H_csr, H_csc, H_syndrome,
                return_iters=True, **kwargs)
        else:
            x_hat, iters = bp_baseline.decode(
                llrs_norm_negated, H_csr, H_csc, H_syndrome,
                return_iters=True, **kwargs)

        # Parity check
        synd = (x_hat.astype(np.int32) @ H_syndrome.T.toarray()) % 2
        converged = np.all(synd == 0, axis=1)
        n_conv = int(converged.sum())
        mean_iters = float(iters[converged].mean()) if n_conv > 0 else float('nan')
        results.append((label, n_conv, B, mean_iters))
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
    print(f"LLR variance: min={all_var.min():.2f}  median={np.median(all_var):.2f}  max={all_var.max():.2f}")
    print(f"LLR magnitude: mean={np.abs(llrs).mean():.3f}  max={np.abs(llrs).max():.3f}")

    # Apply normalization (same as CUDA kernel) and negate for Python ADMM sign convention.
    # FT8SoftCuda: positive = bit 1 likely.
    # Python ADMM (q_j = +llr_j/2): positive = bit 0 likely.
    # So negate after normalizing.
    llrs_norm = np.array([normalize_llr(llr) for llr in llrs], dtype=np.float64)
    norm_var = np.var(llrs_norm, axis=1)
    print(f"After normalization: var mean={norm_var.mean():.2f} (target ~24)")
    llrs_for_admm = -llrs_norm  # negate for Python sign convention

    # Build FT8 H matrix
    print("Building FT8 H matrix...", end=" ", flush=True)
    H_csr = ft8_code.build_parity_check_matrix()
    H_csc = H_csr.T.tocsc()
    H_syndrome = H_csr.astype(np.int8)
    print("done.")

    print(f"\nRunning {B} vectors through ADMM/BP (float64)...\n")
    results = run_configs(llrs_for_admm, H_csr, H_csc, H_syndrome)

    print(f"{'Config':<25}  {'Conv':>5}  {'/ Total':>7}  {'Rate':>6}  {'Mean iters':>11}")
    print("-" * 64)
    for label, n_conv, total, mean_iters in results:
        rate = 100.0 * n_conv / total if total > 0 else 0.0
        iters_str = f"{mean_iters:.1f}" if not np.isnan(mean_iters) else "n/a"
        print(f"{label:<25}  {n_conv:>5}  {total:>7}  {rate:>5.1f}%  {iters_str:>11}")

    print()
    print("Interpretation:")
    print("  If ADMM rho=1.0 iter=100 rate ~100%  → GPU CUDA bug (needs fix or more iters)")
    print("  If ADMM rate << BP rate (iter=25)    → algorithm limitation for this code/SNR")
    print("  If rho=0.5 significantly better      → try lowering RHO in FT8LdpcCuda.cu")


if __name__ == "__main__":
    main()
