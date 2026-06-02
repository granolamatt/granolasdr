#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/ChiFilter.h"

// One thread per frequency bin.
// Integrates the uint8 magnitude over all (slot, sub) in the capture window,
// then marks the block active if any bin exceeds the block-local CFAR threshold.
__global__ void chiPrefilterKernel(
    const uint8_t* __restrict__ mag,
    int snap_start, int ring_size,
    int n_sub, int num_bins, int num_slots,
    float cfar_multiplier,
    uint8_t* __restrict__ block_active,
    uint32_t* __restrict__ active_blocks)
{
    const int fo = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    // For each bin, compute the MAX over sub-bands of the per-sub time-slot sum.
    // Summing all sub-bands (16×) drowns the signal: 8-FSK energy concentrates
    // in one sub-band per bin, so the max preserves the peak SNR rather than
    // averaging it into 15 noise sub-bands. After taking the max, the signal
    // bin sits at ~2.5× the per-sub noise baseline — enough headroom for the
    // CFAR threshold.
    // Use size_t for addressing: rs * n_sub * num_bins can exceed INT_MAX
    // (199 × 16 × 1,048,576 = 3.3 GB).
    uint32_t my_max = 0;
    if (fo < num_bins) {
        for (int s = 0; s < n_sub; ++s) {
            uint32_t sub_sum = 0;
            for (int t = 0; t < num_slots; ++t) {
                int    rs   = (snap_start + t) % ring_size;
                size_t base = (size_t)rs * n_sub * num_bins;
                sub_sum += mag[base + (size_t)s * num_bins + fo];
            }
            if (sub_sum > my_max) my_max = sub_sum;
        }
    }

    // Reduction 1: sum of per-bin max values → block-local mean of the max.
    __shared__ uint32_t smem[256];
    smem[threadIdx.x] = (fo < num_bins) ? my_max : 0u;
    __syncthreads();
    for (int stride = 128; stride >= 1; stride >>= 1) {
        if ((int)threadIdx.x < stride)
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    int active_bins = min((int)blockDim.x, num_bins - (int)(blockIdx.x * blockDim.x));
    if (active_bins < 1) active_bins = 1;
    float threshold = ((float)smem[0] / active_bins) * cfar_multiplier;
    __syncthreads();

    // Reduction 2: OR — any bin whose max-sub sum exceeds threshold marks the block active.
    smem[threadIdx.x] = (fo < num_bins && (float)my_max > threshold) ? 1u : 0u;
    __syncthreads();
    for (int stride = 128; stride >= 1; stride >>= 1) {
        if ((int)threadIdx.x < stride)
            smem[threadIdx.x] |= smem[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint8_t val = (uint8_t)smem[0];
        block_active[blockIdx.x] = val;
        if (active_blocks && val)
            atomicAdd(active_blocks, 1u);
    }
}

void chi_prefilter(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    int n_sub, int num_bins, int num_slots,
    float cfar_multiplier,
    uint8_t* block_active_d,
    cudaStream_t stream,
    uint32_t* active_blocks_d)
{
    const int BLOCK_SZ = 256;
    int grid_x = (num_bins + BLOCK_SZ - 1) / BLOCK_SZ;
    chiPrefilterKernel<<<grid_x, BLOCK_SZ, 0, stream>>>(
        mag_d, snap_start, ring_size,
        n_sub, num_bins, num_slots,
        cfar_multiplier, block_active_d, active_blocks_d);
}
