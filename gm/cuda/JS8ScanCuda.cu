#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/JS8ScanCuda.h"

// JS8 Normal (Mode A) Costas sync pattern — ORIGINAL type, all 3 blocks identical.
// FT8 uses {3, 1, 4, 0, 6, 5, 2}; JS8 Normal uses {4, 2, 5, 6, 1, 3, 0}.
static __constant__ uint8_t kCostas[7] = {4, 2, 5, 6, 1, 3, 0};

// Kernel is structurally identical to ft8SyncScanKernel; only kCostas differs.
// See FT8ScanCuda.cu for full comments on the grid layout and shared-memory scheme.

#define NUM_COSTAS_SLOTS 21

__global__ void js8SyncScanKernel(
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
    bool legacy_costas)
{
    const int BLOCK_SZ = blockDim.x;
    const int WINDOW   = BLOCK_SZ + 8;
    const int fo_base  = blockIdx.x * BLOCK_SZ;
    const int fo       = fo_base + threadIdx.x;

    int combo    = blockIdx.y;
    int sub      = combo / 30;
    int time_off = combo % 30;
    int time_sub = sub / freq_osr;
    int freq_sub = sub % freq_osr;

    const int block_stride = time_osr * freq_osr * num_bins;
    const int sub_offset   = sub * num_bins;

    extern __shared__ uint8_t smem[];

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

    if (fo > num_bins - 8) return;

    if (legacy_costas) {
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

void js8_gpu_scan(
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
    bool legacy_costas)
{
    cudaMemsetAsync(cand_count_d, 0, sizeof(uint32_t), stream);

    const int BLOCK_SZ   = 256;
    const int smem_bytes = NUM_COSTAS_SLOTS * (BLOCK_SZ + 8);
    int grid_x = (num_bins + BLOCK_SZ - 1) / BLOCK_SZ;
    int grid_y = 30 * time_osr * freq_osr;
    dim3 grid(grid_x, grid_y);

    js8SyncScanKernel<<<grid, BLOCK_SZ, smem_bytes, stream>>>(
        mag_d, snap_start, ring_size,
        cand_fo_d, cand_to_d, cand_ts_d, cand_fs_d, cand_score_d, cand_count_d, max_cands,
        num_bins, num_blocks, time_osr, freq_osr, min_score, legacy_costas);
}
