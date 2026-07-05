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
// 256-bin histogram of the slot magnitudes (one thread per (row, bin)).
__global__ static void waterfall_hist_kernel(
    const uint8_t* __restrict__ slot,
    int bin_start, int bin_end, int row_stride,
    unsigned int* __restrict__ hist)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    int bin = bin_start + col;
    if (bin >= bin_end) return;
    const uint8_t* row_base = slot + (long long)row * row_stride;
    atomicAdd(&hist[row_base[bin]], 1u);
}

void waterfall_histogram(
    const uint8_t* slot_base,
    int bin_start, int bin_end,
    int row_stride,
    unsigned int* hist_out,
    cudaStream_t stream)
{
    int src_bins = bin_end - bin_start;
    if (src_bins <= 0) return;
    int threads = 256;
    dim3 grid((src_bins + threads - 1) / threads, FT8_TIME_OSR);
    waterfall_hist_kernel<<<grid, threads, 0, stream>>>(
        slot_base, bin_start, bin_end, row_stride, hist_out);
}

// One thread per output pixel.  blockIdx.y = time sub-array row (0..FT8_TIME_OSR-1).
// Pixel = (mag - noise_ref + wf_offset) * wf_gain, colormapped.  noise_ref is the
// auto-tracked noise floor, so this maps "dB above noise" to brightness — identical
// behaviour on CW and FT8 despite their different absolute magnitude scales.
__global__ static void waterfall_rgba_kernel(
    const uint8_t* __restrict__ slot,
    int bin_start, int src_bins,
    uint8_t* __restrict__ rgba_out, int out_bins,
    int num_bins, int row_stride,
    float noise_ref, float wf_gain, int wf_offset)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;   // = time index t
    if (col >= out_bins) return;

    // Base of this time sub-array (f=0).  row_stride = freq_osr * num_bins from
    // the ring's actual geometry (was hardcoded FT8_FREQ_OSR — wrong for CW).
    const uint8_t* row_base = slot + (long long)row * row_stride;

    // Average-pool [lo, hi) input bins into one output pixel.
    int lo = bin_start + (int)((long long)col       * src_bins / out_bins);
    int hi = bin_start + (int)((long long)(col + 1) * src_bins / out_bins);
    if (hi <= lo) hi = lo + 1;

    uint32_t sum = 0;
    for (int b = lo; b < hi; ++b)
        sum += row_base[b];
    float mag = (float)sum / (float)(hi - lo);

    int v = (int)((mag - noise_ref + (float)wf_offset) * wf_gain);
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
    int num_bins, int row_stride,
    float noise_ref, float wf_gain, int wf_offset,
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
        slot_base, bin_start, src_bins, rgba_out, out_bins, num_bins, row_stride,
        noise_ref, wf_gain, wf_offset);
}
