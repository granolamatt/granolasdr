#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "gm/cuda/FT8ScanCuda.h"

static const int kFtxLdpcN = 174;

// Launch the FT8 soft symbols kernel on scan_stream AFTER a candidate scan.
// Reads the mag ring and writes FTX_LDPC_N LLRs per candidate to log174_d.
// cand_count_d is read (not reset).
// log174_d must be at least max_cands * kFtxLdpcN floats.
void ft8_soft_symbols(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    const int32_t* cand_fo_d,
    const uint8_t* cand_to_d,
    const uint8_t* cand_ts_d,
    const uint8_t* cand_fs_d,
    const uint32_t* cand_count_d,
    float* log174_d,
    int num_bins, int num_blocks, int time_osr, int freq_osr,
    uint32_t max_cands,
    cudaStream_t stream);
