#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/JS8ScanCuda.h"

// JS8 LDPC batch star-sum SP decoder for JS8 Normal (174,87).
//
// Call js8_ldpc_init_constants() once (in JS8Cuda constructor) to copy the
// JS8 (174,87) H-matrix into __constant__ memory.
//
// js8_sp_flooded_decode_batch launches max_cands blocks × 192 threads; blocks
// beyond *cand_count_d exit immediately.  x_hat_d and parity_d must be
// pre-allocated to JS8_LDPC_BATCH × 174 bytes and JS8_LDPC_BATCH bools.

#define JS8_LDPC_BATCH JS8_GPU_CAND_MAX

void js8_ldpc_init_constants();

// Star-sum SP flooded decoder (Gallager phi, flooding schedule, 50 iters).
// cand_count_d: device pointer to actual candidate count written by the scan kernel.
// max_cands:    grid upper bound (pass JS8_GPU_CAND_MAX).
void js8_sp_flooded_decode_batch(
    const uint32_t* cand_count_d,
    uint32_t     max_cands,
    const float* log174_d,
    uint8_t*     x_hat_d,
    bool*        parity_d,
    cudaStream_t ldpc_stream);

// CPU single-candidate BP decoder — used by tests.
// llr:  N=174 channel LLRs (positive = bit likely 1).
// xhat: output N=174 bits of the full codeword.
// Returns true if all parity checks pass.
bool js8_ldpc_decode_cpu(const float* llr, uint8_t* xhat);
