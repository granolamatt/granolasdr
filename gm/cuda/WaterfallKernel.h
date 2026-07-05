#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Decimate [bin_start, bin_end) bins from all FT8_TIME_OSR time sub-arrays
// in one MagBlock ring slot, apply heat colormap, and write RGBA output.
//
// slot_base  : pointer to first byte of the ring slot (device)
// bin_start  : first bin (inclusive) within a sub-array
// bin_end    : last bin (exclusive) within a sub-array
// rgba_out   : device output: [FT8_TIME_OSR * out_bins * 4] uint8 RGBA,
//              row 0 = oldest sub-array (t=0), row 3 = newest (t=3)
// out_bins   : number of output pixels per row
// num_bins   : total bins per sub-array (ring_.num_bins)
// row_stride : elements between consecutive time sub-arrays = freq_osr * num_bins.
//              MUST come from the ring's actual freq_osr (1 for CW, 4 for FT8);
//              hardcoding FT8_FREQ_OSR over-strides the CW ring off its slot.
// wf_floor   : uint8 magnitude value mapped to colormap v=0 (black)
// wf_ceil    : uint8 magnitude value mapped to colormap v=1 (red)
void waterfall_rgba(
    const uint8_t* slot_base,
    int bin_start, int bin_end,
    uint8_t* rgba_out, int out_bins,
    int num_bins, int row_stride,
    uint8_t wf_floor, uint8_t wf_ceil,
    cudaStream_t stream);
