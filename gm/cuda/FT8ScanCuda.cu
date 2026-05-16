#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/FT8ScanCuda.h"

// FT8 Costas tone pattern (3 groups of 7 sync symbols).
static __constant__ uint8_t kCostas[7] = {3, 1, 4, 0, 6, 5, 2};

// One thread per (time_sub, freq_sub, time_offset, freq_offset).
//
// Grid:
//   blockIdx.x * blockDim.x + threadIdx.x  = freq_offset
//   blockIdx.y                              = combo = time_offset * (time_osr*freq_osr)
//                                             + time_sub * freq_osr + freq_sub
//                                             (30 * 16 = 480 values for osr=4)
//
// The mag ring has ring_size "block slots"; each slot is:
//   time_osr * freq_osr * num_bins bytes
// addressed as:
//   mag[ring_slot * block_stride + (time_sub * freq_osr + freq_sub) * num_bins + freq_offset]
__global__ void ft8SyncScanKernel(
    const uint8_t* __restrict__ mag,
    int snap_start, int ring_size,
    int32_t*  __restrict__ cand_fo,
    uint8_t*  __restrict__ cand_to,
    uint8_t*  __restrict__ cand_ts,
    uint8_t*  __restrict__ cand_fs,
    int16_t*  __restrict__ cand_score,
    uint32_t* __restrict__ cand_count,
    uint32_t max_cands,
    int num_bins, int num_blocks, int time_osr, int freq_osr, int min_score)
{
    int fo = blockIdx.x * blockDim.x + threadIdx.x;
    // Need 8 consecutive bins for the 8 FT8 tones.
    if (fo > num_bins - 8) return;

    int combo    = blockIdx.y;
    int time_off = combo / (time_osr * freq_osr);   // 0..29
    int sub      = combo % (time_osr * freq_osr);
    int time_sub = sub / freq_osr;                  // 0..time_osr-1
    int freq_sub = sub % freq_osr;                  // 0..freq_osr-1

    const int block_stride = time_osr * freq_osr * num_bins;
    const int sub_offset   = (time_sub * freq_osr + freq_sub) * num_bins;

    int score = 0, num_avg = 0;

    // Three sync groups at blocks 0-6, 36-42, 72-78.
    for (int m = 0; m < 3; ++m) {
        for (int k = 0; k < 7; ++k) {
            int block_rel = 36 * m + k;
            int block_abs = time_off + block_rel;
            if (block_abs >= num_blocks) break;

            int sm = kCostas[k];  // expected tone bin (0..7)

            // Address of this symbol row in the ring.
            int ring_slot = (snap_start + block_abs) % ring_size;
            const uint8_t* p = mag + (size_t)ring_slot * block_stride + sub_offset + fo;
            int cur = (int)p[sm];

            // Frequency neighbors.
            if (sm > 0) { score += cur - (int)p[sm - 1]; ++num_avg; }
            if (sm < 7) { score += cur - (int)p[sm + 1]; ++num_avg; }

            // Temporal neighbor: one block back (same ts, fs, fo, sm).
            if (k > 0 && block_abs > 0) {
                int rs = (snap_start + block_abs - 1) % ring_size;
                score += cur - (int)(mag[(size_t)rs * block_stride + sub_offset + fo + sm]);
                ++num_avg;
            }
            // Temporal neighbor: one block forward.
            if (k < 6 && block_abs + 1 < num_blocks) {
                int rs = (snap_start + block_abs + 1) % ring_size;
                score += cur - (int)(mag[(size_t)rs * block_stride + sub_offset + fo + sm]);
                ++num_avg;
            }
        }
    }

    if (num_avg > 0) score /= num_avg;
    if (score < min_score) return;

    uint32_t idx = atomicAdd(cand_count, 1u);
    if (idx < max_cands) {
        cand_fo[idx]    = fo;
        cand_to[idx]    = (uint8_t)time_off;
        cand_ts[idx]    = (uint8_t)time_sub;
        cand_fs[idx]    = (uint8_t)freq_sub;
        cand_score[idx] = (int16_t)(score > 32767 ? 32767 : score);
    }
}

void ft8_gpu_scan(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    int32_t*  cand_fo_d,
    uint8_t*  cand_to_d,
    uint8_t*  cand_ts_d,
    uint8_t*  cand_fs_d,
    int16_t*  cand_score_d,
    uint32_t* cand_count_d,
    uint32_t  max_cands,
    int num_bins, int num_blocks, int time_osr, int freq_osr, int min_score,
    cudaStream_t stream)
{
    cudaMemsetAsync(cand_count_d, 0, sizeof(uint32_t), stream);

    const int BLOCK_SZ = 256;
    int grid_x = (num_bins + BLOCK_SZ - 1) / BLOCK_SZ;
    int grid_y = 30 * time_osr * freq_osr;   // 480 combos for osr=4
    dim3 grid(grid_x, grid_y);

    ft8SyncScanKernel<<<grid, BLOCK_SZ, 0, stream>>>(
        mag_d, snap_start, ring_size,
        cand_fo_d, cand_to_d, cand_ts_d, cand_fs_d, cand_score_d, cand_count_d, max_cands,
        num_bins, num_blocks, time_osr, freq_osr, min_score);
}
