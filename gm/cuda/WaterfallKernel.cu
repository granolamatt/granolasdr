#include <cuda_runtime.h>
#include <stdint.h>
#include "gm/cuda/WaterfallKernel.h"

// Decimate a wideband mag ring slot to WATERFALL_BINS output bytes.
// Maps output bin b → src bin round((rfft_length-1) * (b/(BINS-1))^2),
// which gives quadratic (log-frequency-like) spacing across the spectrum.
// The t=0, f=0 sub-block (first rfft_length bytes of the ring slot) is used.
__global__ static void waterfall_decimate_kernel(
    const uint8_t* __restrict__ ring_slot,  // first rfft_length bytes of the slot
    uint8_t*       __restrict__ out,
    int rfft_length,
    int out_bins)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= out_bins) return;
    float t = (out_bins > 1) ? (float)b / (float)(out_bins - 1) : 0.0f;
    int src = (int)roundf((float)(rfft_length - 1) * t * t);
    out[b] = ring_slot[src];
}

void ft8_waterfall_decimate(
    const uint8_t* ring_slot_d,
    uint8_t*       out_d,
    int            rfft_length,
    int            out_bins,
    cudaStream_t   stream)
{
    int threads = 256;
    int blocks  = (out_bins + threads - 1) / threads;
    waterfall_decimate_kernel<<<blocks, threads, 0, stream>>>(
        ring_slot_d, out_d, rfft_length, out_bins);
}
