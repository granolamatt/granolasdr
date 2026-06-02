#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Upper bound on GPU candidates returned per JS8 scan.
static const uint32_t JS8_GPU_CAND_MAX = 2048;

// Launch the JS8 sync score scan kernel.
//
// Identical to ft8_gpu_scan() but uses JS8 Normal Costas array {4,2,5,6,1,3,0}.
// mag_d   : device ring [ring_size blocks][time_osr*freq_osr][num_bins]
// snap_start : (ring_write_idx - num_blocks) % ring_size
// ring_size  : must equal RING_BLOCKS from FT8Cuda
//
// All cand_* pointers are device pointers.  cand_count_d is reset to 0 before the kernel.
void js8_gpu_scan(
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
    const uint8_t* block_active_d = nullptr);
