#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Two-stage CFAR pre-filter for the Costas sync scan.
//
// Reads the uint8 mag ring, integrates each frequency bin over the full
// capture window (num_slots ring rows × n_sub sub-bands), then marks each
// 256-bin block as active when at least one bin in that block exceeds the
// block-local mean magnitude times cfar_multiplier.
//
// The result (block_active_d) is consumed by ft8_gpu_scan / js8_gpu_scan:
// blocks whose entry is 0 return immediately, skipping all shared-memory
// loads and the 21-symbol inner loop.
//
// Expected reduction: ~15k candidate bins out of 1M → ~60 active blocks
// out of 4096 → scan grid from ~2M thread blocks to ~29k.
//
// block_active_d : device buffer uint8[ceil(num_bins / 256)]; caller allocates.
// cfar_multiplier: threshold = block_local_mean × multiplier.  Start at 2.0;
//                  lower to recover weaker signals at the cost of more scan blocks.
// active_blocks_d: optional device uint32 counter; atomically incremented for each
//                  active block. Zero it before the call if you want an accurate count.
void chi_prefilter(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    int n_sub, int num_bins, int num_slots,
    float cfar_multiplier,
    uint8_t* block_active_d,
    cudaStream_t stream,
    uint32_t* active_blocks_d = nullptr);
