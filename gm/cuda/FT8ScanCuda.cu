#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/FT8ScanCuda.h"

// FT8 Costas tone pattern (3 groups of 7 sync symbols).
static __constant__ uint8_t kCostas[7] = {3, 1, 4, 0, 6, 5, 2};

// One thread per (time_sub, freq_sub, time_offset, freq_offset).
//
// Grid:
//   blockIdx.x * blockDim.x + threadIdx.x = freq_offset (fo)
//   blockIdx.y = sub * 30 + time_offset
//                sub      = time_sub * freq_osr + freq_sub  (0..15, SLOW axis)
//                time_off = 0..29                            (FAST axis)
//
// "sub" is the slow axis so adjacent blockIdx.y values share the same sub_offset.
// Blocks with the same sub and adjacent time_off access the same cache lines
// in the ring (~86% ring-slot overlap) → significant L2 reuse between blocks
// co-scheduled on the same SM.
//
// The mag ring has ring_size "block slots"; each slot is:
//   time_osr * freq_osr * num_bins bytes
// addressed as:
//   mag[ring_slot * block_stride + (time_sub * freq_osr + freq_sub) * num_bins + freq_offset]
//
// Shared memory layout: NUM_SLOTS × WINDOW bytes, where
//   NUM_SLOTS = 21  (3 Costas groups × 7 symbols)
//   WINDOW    = blockDim.x + 8  (thread fo range plus max tone offset)
// All threads cooperatively load the 21 windows in a coalesced stride-1 pattern,
// then compute the Costas score entirely from shared memory (no further global reads).
// Score metric: max-log 8-FSK — per symbol, expected tone minus best competitor tone,
// averaged over valid Costas symbols.  Units: 0.5 dB per LSB (matches uint8 encoding).

#define NUM_COSTAS_SLOTS 21

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
    int num_bins, int num_blocks, int time_osr, int freq_osr, float min_score,
    const uint8_t* __restrict__ block_active,
    bool legacy_costas)
{
    // Chi pre-filter: skip entire blocks with no candidate bins.
    if (block_active && !block_active[blockIdx.x]) return;

    const int BLOCK_SZ = blockDim.x;
    const int WINDOW   = BLOCK_SZ + 8;  // covers tone offsets 0..7
    const int fo_base  = blockIdx.x * BLOCK_SZ;
    const int fo       = fo_base + threadIdx.x;

    // sub varies slowly → same sub_offset for all blocks in the same group of 30.
    int combo    = blockIdx.y;
    int sub      = combo / 30;          // 0..time_osr*freq_osr-1 (slow axis)
    int time_off = combo % 30;          // 0..29                  (fast axis)
    int time_sub = sub / freq_osr;
    int freq_sub = sub % freq_osr;

    const int block_stride = time_osr * freq_osr * num_bins;
    const int sub_offset   = sub * num_bins;

    // Shared memory: NUM_COSTAS_SLOTS windows, each WINDOW bytes.
    // smem[slot * WINDOW + threadIdx.x + tone_off] = mag at (ring_slot, sub, fo_base + threadIdx.x + tone_off).
    extern __shared__ uint8_t smem[];

    // Cooperatively load all 21 windows into shared memory.
    // Total bytes = 21 * WINDOW (≈ 5544 for BLOCK_SZ=256).
    // Each thread loads ceil(21*WINDOW / BLOCK_SZ) ≈ 22 bytes in a stride-BLOCK_SZ pattern.
    const int SMEM_TOTAL = NUM_COSTAS_SLOTS * WINDOW;
    for (int i = threadIdx.x; i < SMEM_TOTAL; i += BLOCK_SZ) {
        int slot_idx  = i / WINDOW;
        int fo_off    = i % WINDOW;
        int m         = slot_idx / 7;
        int k         = slot_idx % 7;
        int block_abs = time_off + 36 * m + k;
        uint8_t val   = 0;
        if (block_abs < num_blocks) {
            int ring_slot  = (snap_start + block_abs) % ring_size;
            int global_fo  = fo_base + fo_off;
            if (global_fo < num_bins)
                val = mag[(size_t)ring_slot * block_stride + sub_offset + global_fo];
        }
        smem[i] = val;
    }
    __syncthreads();

    // Threads outside the valid bin range helped load smem; now they exit.
    if (fo > num_bins - 8) return;

    if (legacy_costas) {
        // Legacy metric: freq neighbors ±1 tone + temporal neighbors ±1 block, averaged.
        int score = 0, num_avg = 0;
        for (int m = 0; m < 3; ++m) {
            for (int k = 0; k < 7; ++k) {
                int block_abs = time_off + 36 * m + k;
                if (block_abs >= num_blocks) break;
                int sm  = kCostas[k];
                int si  = (m * 7 + k) * WINDOW + threadIdx.x;
                int cur = (int)smem[si + sm];
                if (sm > 0) { score += cur - (int)smem[si + sm - 1]; ++num_avg; }
                if (sm < 7) { score += cur - (int)smem[si + sm + 1]; ++num_avg; }
                if (k > 0 && block_abs > 0) {
                    int si_prev = (m * 7 + k - 1) * WINDOW + threadIdx.x;
                    score += cur - (int)smem[si_prev + sm]; ++num_avg;
                }
                if (k < 6 && block_abs + 1 < num_blocks) {
                    int si_next = (m * 7 + k + 1) * WINDOW + threadIdx.x;
                    score += cur - (int)smem[si_next + sm]; ++num_avg;
                }
            }
        }
        if (num_avg == 0) return;
        score /= num_avg;
        if (score < (int)min_score) return;
        uint32_t idx = atomicAdd(cand_count, 1u);
        if (idx < max_cands) {
            cand_fo[idx]    = fo;
            cand_to[idx]    = (uint8_t)time_off;
            cand_ts[idx]    = (uint8_t)time_sub;
            cand_fs[idx]    = (uint8_t)freq_sub;
            cand_score[idx] = (int16_t)(score > 32767 ? 32767 : score);
        }
    } else {
        // Max-log 8-FSK Costas score: expected tone minus best competitor, per symbol.
        int score = 0, num_valid = 0;
        for (int m = 0; m < 3; ++m) {
            for (int k = 0; k < 7; ++k) {
                int block_abs = time_off + 36 * m + k;
                if (block_abs >= num_blocks) break;
                int sm = kCostas[k];
                int si = (m * 7 + k) * WINDOW + threadIdx.x;
                int cur = (int)smem[si + sm];
                int best_other = 0;
                for (int j = 0; j < 8; ++j) {
                    if (j == sm) continue;
                    int v = (int)smem[si + j];
                    if (v > best_other) best_other = v;
                }
                score += cur - best_other;
                ++num_valid;
            }
        }
        if (num_valid == 0) return;
        float fscore = (float)score / (float)num_valid;
        if (fscore < min_score) return;
        uint32_t idx = atomicAdd(cand_count, 1u);
        if (idx < max_cands) {
            cand_fo[idx]    = fo;
            cand_to[idx]    = (uint8_t)time_off;
            cand_ts[idx]    = (uint8_t)time_sub;
            cand_fs[idx]    = (uint8_t)freq_sub;
            int16_t s16     = (fscore >= 32767.0f) ? 32767 : (int16_t)fscore;
            cand_score[idx] = s16;
        }
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
    int num_bins, int num_blocks, int time_osr, int freq_osr, float min_score,
    cudaStream_t stream,
    const uint8_t* block_active_d,
    bool legacy_costas)
{
    cudaMemsetAsync(cand_count_d, 0, sizeof(uint32_t), stream);

    const int BLOCK_SZ   = 256;
    const int smem_bytes = NUM_COSTAS_SLOTS * (BLOCK_SZ + 8);  // 21 * 264 = 5544 bytes
    int grid_x = (num_bins + BLOCK_SZ - 1) / BLOCK_SZ;
    int grid_y = 30 * time_osr * freq_osr;   // 480 combos for osr=4
    dim3 grid(grid_x, grid_y);

    ft8SyncScanKernel<<<grid, BLOCK_SZ, smem_bytes, stream>>>(
        mag_d, snap_start, ring_size,
        cand_fo_d, cand_to_d, cand_ts_d, cand_fs_d, cand_score_d, cand_count_d, max_cands,
        num_bins, num_blocks, time_osr, freq_osr, min_score, block_active_d, legacy_costas);
}
