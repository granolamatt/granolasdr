#ifndef _GM_CUDA_FT8LDPCCUDA_H_
#define _GM_CUDA_FT8LDPCCUDA_H_

#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/FT8ScanCuda.h"

// FT8 QP-ADMM LDPC batch decoder — GPU stream-aware launcher.
//
// Call ft8_ldpc_init_constants() once (in FT8Cuda constructor) to copy the
// FT8 (174,91) H-matrix into __constant__ memory.
//
// ft8_ldpc_decode_batch() launches n_candidates blocks × 192 threads on
// ldpc_stream without synchronizing. x_hat_d and parity_d must be device
// buffers pre-allocated to FT8_LDPC_BATCH × 174 and FT8_LDPC_BATCH entries.

#define FT8_LDPC_BATCH FT8_GPU_CAND_MAX

void ft8_ldpc_init_constants();
void ft8_ldpc_decode_batch(
    const float* log174_d,
    uint8_t*     x_hat_d,
    bool*        parity_d,
    cudaStream_t ldpc_stream,
    uint32_t     n_candidates);

#endif // _GM_CUDA_FT8LDPCCUDA_H_
