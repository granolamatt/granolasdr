#include <cuda_runtime.h>
#include <cstdio>
#include <stdint.h>
#include "gm/cuda/WaterfallKernel.h"
#include "gm/hf/ft8_capture.h"   // FT8_TIME_OSR, FT8_FREQ_OSR

// Heat colormap matching control/index.html heatColor():
//   black → dark blue → cyan → green/yellow → red
__device__ static void heat_color(uint8_t vi, uint8_t& r, uint8_t& g, uint8_t& b)
{
    if (vi < 64)       { r = 0;              g = 0;                    b = (uint8_t)(vi * 3); }
    else if (vi < 128) { r = 0;              g = (uint8_t)((vi-64)*4); b = (uint8_t)(255-(vi-64)*4); }
    else if (vi < 192) { r = (uint8_t)((vi-128)*4); g = 255;           b = 0; }
    else               { r = 255;            g = (uint8_t)(255-(vi-192)*4); b = 0; }
}

// One thread per output pixel.
// blockIdx.y = time sub-array row (0 .. FT8_TIME_OSR-1).
// blockIdx.x * blockDim.x + threadIdx.x = output column [0, out_bins).
//
// Each time sub-array strides by FT8_FREQ_OSR sub-arrays (f=0..3); we use f=0.
// Sub-array layout in slot: index (t * FT8_FREQ_OSR + f) at byte offset
//   (t * FT8_FREQ_OSR + f) * num_bins.
__global__ static void waterfall_rgba_kernel(
    const uint8_t* __restrict__ slot,
    int bin_start, int src_bins,
    uint8_t* __restrict__ rgba_out, int out_bins,
    int num_bins,
    uint8_t wf_floor, uint8_t wf_ceil)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;   // = time index t
    if (col >= out_bins) return;

    // Base of this time sub-array (f=0).
    const uint8_t* row_base = slot + (long long)row * FT8_FREQ_OSR * num_bins;

    // Average-pool [lo, hi) input bins into one output pixel.
    int lo = bin_start + (int)((long long)col       * src_bins / out_bins);
    int hi = bin_start + (int)((long long)(col + 1) * src_bins / out_bins);
    if (hi <= lo) hi = lo + 1;

    uint32_t sum = 0;
    for (int b = lo; b < hi; ++b)
        sum += row_base[b];
    uint8_t mag = (uint8_t)(sum / (uint32_t)(hi - lo));

    // Scale to [0, 255] using floor/ceil window.
    int range = (int)wf_ceil - (int)wf_floor;
    if (range <= 0) range = 1;
    int v = ((int)mag - (int)wf_floor) * 255 / range;
    if (v < 0)   v = 0;
    if (v > 255) v = 255;

    uint8_t r, g, b;
    heat_color((uint8_t)v, r, g, b);

    int idx = (row * out_bins + col) * 4;
    rgba_out[idx + 0] = r;
    rgba_out[idx + 1] = g;
    rgba_out[idx + 2] = b;
    rgba_out[idx + 3] = 255;
}

void waterfall_rgba(
    const uint8_t* slot_base,
    int bin_start, int bin_end,
    uint8_t* rgba_out, int out_bins,
    int num_bins,
    uint8_t wf_floor, uint8_t wf_ceil,
    cudaStream_t stream)
{
    int src_bins = bin_end - bin_start;
    if (src_bins <= 0 || out_bins <= 0 || num_bins <= 0) {
        fprintf(stderr, "[WF] waterfall_rgba: invalid params bin_start=%d bin_end=%d out_bins=%d\n",
                bin_start, bin_end, out_bins);
        return;
    }
    int threads = 256;
    int blocks_x = (out_bins + threads - 1) / threads;
    dim3 grid(blocks_x, FT8_TIME_OSR);
    waterfall_rgba_kernel<<<grid, threads, 0, stream>>>(
        slot_base, bin_start, src_bins, rgba_out, out_bins, num_bins,
        wf_floor, wf_ceil);
}
