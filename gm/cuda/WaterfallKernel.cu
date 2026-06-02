#include <cuda_runtime.h>
#include <stdint.h>
#include "gm/cuda/WaterfallKernel.h"

// One thread per output pixel.
// Maps output bin b → average of input bins [lo, hi) where
// lo = bin_start + b * src_bins / out_bins
// hi = bin_start + (b+1) * src_bins / out_bins
// This is true average pooling with linear frequency mapping.
__global__ static void waterfall_decimate_kernel(
    const uint8_t* __restrict__ slot,
    int bin_start, int src_bins,
    uint8_t* __restrict__ out, int out_bins)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= out_bins) return;

    int lo = bin_start + (int)((long long)b       * src_bins / out_bins);
    int hi = bin_start + (int)((long long)(b + 1) * src_bins / out_bins);
    if (hi <= lo) hi = lo + 1;

    uint32_t sum = 0;
    for (int i = lo; i < hi; ++i)
        sum += slot[i];
    out[b] = (uint8_t)(sum / (uint32_t)(hi - lo));
}

void waterfall_decimate(
    const uint8_t* ring_slot_d,
    int bin_start, int bin_end,
    uint8_t* out_d, int out_bins,
    cudaStream_t stream)
{
    int src_bins = bin_end - bin_start;
    if (src_bins <= 0 || out_bins <= 0) return;
    int threads = 256;
    int blocks  = (out_bins + threads - 1) / threads;
    waterfall_decimate_kernel<<<blocks, threads, 0, stream>>>(
        ring_slot_d, bin_start, src_bins, out_d, out_bins);
}
