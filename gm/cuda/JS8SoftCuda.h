#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// JS8 soft-symbol LLR extraction — natural binary tone mapping (no Gray code).
void js8_soft_symbols(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    const int32_t* cand_fo_d,
    const uint8_t* cand_to_d,
    const uint8_t* cand_ts_d,
    const uint8_t* cand_fs_d,
    const uint32_t* cand_count_d,
    float* log174_d,
    int num_bins, int num_blocks, int time_osr, int freq_osr,
    int max_cands,
    cudaStream_t stream);
