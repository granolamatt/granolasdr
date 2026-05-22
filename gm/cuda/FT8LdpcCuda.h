#ifndef _GM_CUDA_FT8LDPCCUDA_H_
#define _GM_CUDA_FT8LDPCCUDA_H_

#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/FT8ScanCuda.h"

// FT8 LDPC batch decoders — GPU stream-aware launchers.
//
// Call ft8_ldpc_init_constants() once (in FT8Cuda constructor) to copy the
// FT8 (174,91) H-matrix into __constant__ memory.
//
// Both launchers take the same arguments: n_candidates blocks × threads,
// streaming on ldpc_stream without host synchronization.
// x_hat_d and parity_d must be pre-allocated to FT8_LDPC_BATCH × 174 and
// FT8_LDPC_BATCH entries respectively.
//
//   ft8_bp_decode_batch    — Sum-product belief propagation (25 iters).
//                            Matches ft8_lib's bp_decode(); ~90%+ rate.
//   ft8_ldpc_decode_batch  — QP-ADMM LP decoder (100 iters, with centering).
//                            Kept for comparison; BP is preferred.

#define FT8_LDPC_BATCH FT8_GPU_CAND_MAX

void ft8_ldpc_init_constants();

void ft8_bp_decode_batch(
    uint32_t     n_candidates,
    const float* log174_d,
    uint8_t*     x_hat_d,
    bool*        parity_d,
    cudaStream_t ldpc_stream);

void ft8_ldpc_decode_batch(
    uint32_t     n_candidates,
    const float* log174_d,
    uint8_t*     x_hat_d,
    bool*        parity_d,
    cudaStream_t ldpc_stream);

#endif // _GM_CUDA_FT8LDPCCUDA_H_
