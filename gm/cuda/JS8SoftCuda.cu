// JS8 soft-symbol LLR kernel — identical to FT8SoftCuda.cu but without the
// Gray-code permutation.  JS8 encodes 3 bits per FSK tone using natural binary
// (tone j carries bit pattern j directly), so s2[j] = mag at fo+j with no
// permutation.  Using the FT8 Gray map gives wrong bit-group assignments for
// bits 1 and 0, breaking LDPC convergence.

#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/JS8ScanCuda.h"
#include "gm/cuda/JS8SoftCuda.h"

#define JS8_ND_SOFT   58   // data symbols per JS8 frame
#define FTX_LDPC_N_D 174   // LLR bits per candidate

static __device__ inline float fmax4d(float a, float b, float c, float d) {
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

__global__ void js8_soft_symbols_kernel(
    const uint8_t* __restrict__ mag,
    int snap_start, int ring_size,
    const int32_t* __restrict__ cand_fo,
    const uint8_t* __restrict__ cand_to,
    const uint8_t* __restrict__ cand_ts,
    const uint8_t* __restrict__ cand_fs,
    const uint32_t* __restrict__ cand_count_d,
    float* __restrict__ log174_d,
    int num_bins, int num_blocks, int time_osr, int freq_osr,
    int max_cands)
{
    int flat     = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int cand_idx = flat / JS8_ND_SOFT;
    int k        = flat % JS8_ND_SOFT;

    int n = (int)min(*cand_count_d, (uint32_t)max_cands);
    if (cand_idx >= n) return;

    int time_off = (int)cand_to[cand_idx];
    int time_sub = (int)cand_ts[cand_idx];
    int freq_sub = (int)cand_fs[cand_idx];
    int fo       = cand_fo[cand_idx];

    // Skip Costas sync groups: symbols 0-6, 36-42, 72-78 are sync.
    // Data symbols k=0..28 → frame symbol k+7; k=29..57 → frame symbol k+14.
    int sym_idx   = k + ((k < 29) ? 7 : 14);
    int block_abs = time_off + sym_idx;
    int bit_idx   = k * 3;

    float* out = log174_d + (size_t)cand_idx * FTX_LDPC_N_D + bit_idx;

    if (block_abs < 0 || block_abs >= num_blocks) {
        out[0] = out[1] = out[2] = 0.0f;
        return;
    }

    // Ring addressing (identical to ft8_soft_symbols_kernel).
    int sub          = time_sub * freq_osr + freq_sub;
    int sub_offset   = sub * num_bins;
    int block_stride = time_osr * freq_osr * num_bins;
    int ring_slot    = (snap_start + block_abs) % ring_size;

    // Natural binary: s2[j] = magnitude at tone j (no Gray permutation).
    float s2[8];
    for (int j = 0; j < 8; ++j) {
        int bin = fo + j;
        s2[j] = (bin >= 0 && bin < num_bins)
                ? (float)mag[(size_t)ring_slot * block_stride + sub_offset + bin] * 0.5f
                : 0.0f;
    }

    // Max-log MAP LLR for natural-binary 8-FSK.
    // Bit 2 (MSB): 1 for tones 4-7, 0 for tones 0-3.
    // Bit 1:       1 for tones 2,3,6,7; 0 for 0,1,4,5.
    // Bit 0 (LSB): 1 for odd tones; 0 for even tones.
    out[0] = fmax4d(s2[4], s2[5], s2[6], s2[7]) - fmax4d(s2[0], s2[1], s2[2], s2[3]);
    out[1] = fmax4d(s2[2], s2[3], s2[6], s2[7]) - fmax4d(s2[0], s2[1], s2[4], s2[5]);
    out[2] = fmax4d(s2[1], s2[3], s2[5], s2[7]) - fmax4d(s2[0], s2[2], s2[4], s2[6]);
}

void js8_soft_symbols(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    const int32_t* cand_fo_d,
    const uint8_t* cand_to_d,
    const uint8_t* cand_ts_d,
    const uint8_t* cand_fs_d,
    const uint32_t* cand_count_d,
    float* log174_d,
    int num_bins, int num_blocks, int time_osr, int freq_osr,
    int max_cands,
    cudaStream_t stream)
{
    const int BLOCK_SZ = 256;
    const int total    = max_cands * JS8_ND_SOFT;
    int grid           = (total + BLOCK_SZ - 1) / BLOCK_SZ;

    js8_soft_symbols_kernel<<<grid, BLOCK_SZ, 0, stream>>>(
        mag_d, snap_start, ring_size,
        cand_fo_d, cand_to_d, cand_ts_d, cand_fs_d, cand_count_d,
        log174_d, num_bins, num_blocks, time_osr, freq_osr, max_cands);
}
