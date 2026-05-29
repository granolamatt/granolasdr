#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/JS8ScanCuda.h"

// JS8 LDPC batch BP decoder for JS8 Normal (174,87).
//
// Call js8_ldpc_init_constants() once (in JS8Cuda constructor) to copy the
// JS8 (174,87) H-matrix into __constant__ memory.
//
// js8_bp_decode_batch launches n_candidates blocks × 192 threads.
// x_hat_d and parity_d must be pre-allocated to JS8_LDPC_BATCH × 174
// and JS8_LDPC_BATCH entries respectively.

#define JS8_LDPC_BATCH JS8_GPU_CAND_MAX

void js8_ldpc_init_constants();

void js8_bp_decode_batch(
    uint32_t     n_candidates,
    const float* log174_d,
    uint8_t*     x_hat_d,
    bool*        parity_d,
    cudaStream_t ldpc_stream);

// CPU single-candidate BP decoder — same algorithm as GPU kernel.
// llr:  N=174 channel LLRs (positive = bit likely 1).
// xhat: output N=174 bits of the full codeword.
// Returns true if all parity checks pass.
bool js8_ldpc_decode_cpu(const float* llr, uint8_t* xhat);
