#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Launch the JS8 Fast-mode sync score scan kernel.
//
// Uses MODIFIED Costas arrays (Fast/Turbo/Ultra/Slow share this pattern):
//   Block 0: {0, 6, 2, 3, 5, 4, 1}
//   Block 1: {1, 5, 0, 2, 3, 6, 4}
//   Block 2: {2, 5, 0, 6, 4, 1, 3}
//
// All other grid/memory parameters are identical to js8_gpu_scan.
// Caller must pass ring_size=100 and num_blocks=100 for the 100-slot Fast ring.
void js8_fast_gpu_scan(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    int32_t*  cand_fo_d,
    uint8_t*  cand_to_d,
    uint8_t*  cand_ts_d,
    uint8_t*  cand_fs_d,
    int16_t*  cand_score_d,
    uint32_t* cand_count_d,
    uint32_t  max_cands,
    int num_bins, int num_blocks, int time_osr, int freq_osr, float min_score,
    cudaStream_t stream,
    bool legacy_costas = false);
