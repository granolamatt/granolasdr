#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Upper bound on GPU candidates returned per epoch.
static const uint32_t FT8_GPU_CAND_MAX = 2048;

// Launch the FT8 sync score scan kernel.
//
// mag_d   : device ring [ring_size blocks][time_osr*freq_osr][num_bins]
//           Each "block" is time_osr*freq_osr*num_bins bytes.
// snap_start : index of the oldest block in the ring = (ring_write_idx - num_blocks) % ring_size
// ring_size  : must equal num_blocks (FT8_CAPTURE_BLOCKS)
//
// All cand_* pointers are device pointers.  cand_count_d is reset to 0 by this call before the
// kernel runs.  After stream synchronization, *cand_count_d holds the number of candidates found.
void ft8_gpu_scan(
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
    const uint8_t* block_active_d = nullptr,
    bool legacy_costas = false);
