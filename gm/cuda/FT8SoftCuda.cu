#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/FT8ScanCuda.h"
#include "gm/cuda/FT8SoftCuda.h"

// kFT8_Gray_map[8] = {0,1,3,2,5,6,4,7} — tone permutation for Gray-coded 8-FSK.
// Duplicated from ft8_lib/ft8/constants.c because the .cu file cannot link against libft8.a.
static __constant__ uint8_t kFT8_Gray_map_d[8] = {0, 1, 3, 2, 5, 6, 4, 7};

#define FT8_ND_SOFT    58   // data symbols per FT8 message (FT8_ND)
#define FTX_LDPC_N_D  174   // LLR bits per candidate

static __device__ inline float fmax4d(float a, float b, float c, float d) {
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

// One thread per (candidate, data-symbol).
// flat_idx = cand_idx * FT8_ND_SOFT + k
//
// Ring addressing copied verbatim from ft8SyncScanKernel in FT8ScanCuda.cu.
// The -120 dB offset from WF_ELEM_MAG cancels in LLR differences — omitted here.
__global__ void ft8_soft_symbols_kernel(
    const uint8_t* __restrict__ mag,
    int snap_start, int ring_size,
    const int32_t* __restrict__ cand_fo,
    const uint8_t* __restrict__ cand_to,
    const uint8_t* __restrict__ cand_ts,
    const uint8_t* __restrict__ cand_fs,
    const uint32_t* __restrict__ cand_count_d,
    float* __restrict__ log174_d,
    int num_bins, int num_blocks, int time_osr, int freq_osr,
    uint32_t max_cands)
{
    int flat     = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int cand_idx = flat / FT8_ND_SOFT;
    int k        = flat % FT8_ND_SOFT;

    int n = (int)min(*cand_count_d, max_cands);
    if (cand_idx >= n) return;

    int time_off = (int)cand_to[cand_idx];
    int time_sub = (int)cand_ts[cand_idx];
    int freq_sub = (int)cand_fs[cand_idx];
    int fo       = cand_fo[cand_idx];

    // Skip Costas sync groups: symbols 0-6, 36-43, 72-79 are sync.
    // Data symbols k=0..28 map to message block k+7; k=29..57 map to k+14.
    int sym_idx   = k + ((k < 29) ? 7 : 14);
    int block_abs = time_off + sym_idx;
    int bit_idx   = k * 3;

    float* out = log174_d + (size_t)cand_idx * FTX_LDPC_N_D + bit_idx;

    if (block_abs < 0 || block_abs >= num_blocks) {
        out[0] = out[1] = out[2] = 0.0f;
        return;
    }

    // Ring addressing (see ft8SyncScanKernel lines 55-80).
    int sub          = time_sub * freq_osr + freq_sub;
    int sub_offset   = sub * num_bins;
    int block_stride = time_osr * freq_osr * num_bins;
    int ring_slot    = (snap_start + block_abs) % ring_size;

    float s2[8];
    for (int j = 0; j < 8; ++j) {
        int bin = fo + (int)kFT8_Gray_map_d[j];
        s2[j] = (bin >= 0 && bin < num_bins)
                ? (float)mag[(size_t)ring_slot * block_stride + sub_offset + bin] * 0.5f
                : 0.0f;
    }

    out[0] = fmax4d(s2[4], s2[5], s2[6], s2[7]) - fmax4d(s2[0], s2[1], s2[2], s2[3]);
    out[1] = fmax4d(s2[2], s2[3], s2[6], s2[7]) - fmax4d(s2[0], s2[1], s2[4], s2[5]);
    out[2] = fmax4d(s2[1], s2[3], s2[5], s2[7]) - fmax4d(s2[0], s2[2], s2[4], s2[6]);
}

void ft8_soft_symbols(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    const int32_t* cand_fo_d,
    const uint8_t* cand_to_d,
    const uint8_t* cand_ts_d,
    const uint8_t* cand_fs_d,
    const uint32_t* cand_count_d,
    float* log174_d,
    int num_bins, int num_blocks, int time_osr, int freq_osr,
    uint32_t max_cands,
    cudaStream_t stream)
{
    const int BLOCK_SZ = 256;
    const int total    = (int)max_cands * FT8_ND_SOFT;
    int grid           = (total + BLOCK_SZ - 1) / BLOCK_SZ;

    ft8_soft_symbols_kernel<<<grid, BLOCK_SZ, 0, stream>>>(
        mag_d, snap_start, ring_size,
        cand_fo_d, cand_to_d, cand_ts_d, cand_fs_d, cand_count_d,
        log174_d, num_bins, num_blocks, time_osr, freq_osr, max_cands);
}
