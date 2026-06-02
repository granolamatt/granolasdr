#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Decimate a [bin_start, bin_end) slice of ring slot sub-band 0 to out_bins
// output bytes using linear average pooling.
// ring_slot points to the first byte of the ring slot (sub-band 0 base).
// Each output pixel averages ceil((bin_end-bin_start)/out_bins) input bins.
void waterfall_decimate(
    const uint8_t* ring_slot_d,
    int bin_start, int bin_end,
    uint8_t* out_d, int out_bins,
    cudaStream_t stream);
