#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/FT8SyndromeScan.h"

// Data symbols per FT8 message (58 × 3 bits = 174 = FTX_LDPC_N).
#define NUM_DATA_SLOTS 58

// Sync symbols per FT8 message: 3 groups × 7 symbols = 21.
#define NUM_SYNC_SLOTS 21

// FT8 Costas array: expected tone index at each position within a sync group.
// Repeats for all 3 sync groups (offsets 0-6, 36-42, 72-78 from message start).
static __constant__ uint8_t kFT8_Costas_d[7] = {3, 1, 4, 0, 6, 5, 2};

// Inverse of kFT8_Gray_map_d = {0,1,3,2,5,6,4,7}: tone index -> Gray-coded symbol.
// Derivation: kFT8_Gray_map_d[j] = t means tone t carries symbol j, so inv[t] = j.
// Verified against FT8SoftCuda.cu LLR sign convention:
//   bit k*3+0 = (j>>2)&1, bit k*3+1 = (j>>1)&1, bit k*3+2 = j&1.
static __constant__ uint8_t kFT8_inv_Gray_map_d[8] = {0, 1, 3, 2, 6, 4, 5, 7};

// For each of the 174 codeword bits, the 3 LDPC check equations it participates in.
// 1-indexed (same convention as ft8_lib/ft8/constants.c kFTX_LDPC_Mn).
// Duplicated from libft8 because .cu files cannot link against libft8.a.
static __constant__ uint8_t kFTX_LDPC_Mn_d[174][3] = {
    { 16, 45, 73 }, { 25, 51, 62 }, { 33, 58, 78 }, { 1,  44, 45 },
    { 2,  7,  61 }, { 3,  6,  54 }, { 4,  35, 48 }, { 5,  13, 21 },
    { 8,  56, 79 }, { 9,  64, 69 }, { 10, 19, 66 }, { 11, 36, 60 },
    { 12, 37, 58 }, { 14, 32, 43 }, { 15, 63, 80 }, { 17, 28, 77 },
    { 18, 74, 83 }, { 22, 53, 81 }, { 23, 30, 34 }, { 24, 31, 40 },
    { 26, 41, 76 }, { 27, 57, 70 }, { 29, 49, 65 }, { 3,  38, 78 },
    { 5,  39, 82 }, { 46, 50, 73 }, { 51, 52, 74 }, { 55, 71, 72 },
    { 44, 67, 72 }, { 43, 68, 78 }, { 1,  32, 59 }, { 2,  6,  71 },
    { 4,  16, 54 }, { 7,  65, 67 }, { 8,  30, 42 }, { 9,  22, 31 },
    { 10, 18, 76 }, { 11, 23, 82 }, { 12, 28, 61 }, { 13, 52, 79 },
    { 14, 50, 51 }, { 15, 81, 83 }, { 17, 29, 60 }, { 19, 33, 64 },
    { 20, 26, 73 }, { 21, 34, 40 }, { 24, 27, 77 }, { 25, 55, 58 },
    { 35, 53, 66 }, { 36, 48, 68 }, { 37, 46, 75 }, { 38, 45, 47 },
    { 39, 57, 69 }, { 41, 56, 62 }, { 20, 49, 53 }, { 46, 52, 63 },
    { 45, 70, 75 }, { 27, 35, 80 }, { 1,  15, 30 }, { 2,  68, 80 },
    { 3,  36, 51 }, { 4,  28, 51 }, { 5,  31, 56 }, { 6,  20, 37 },
    { 7,  40, 82 }, { 8,  60, 69 }, { 9,  10, 49 }, { 11, 44, 57 },
    { 12, 39, 59 }, { 13, 24, 55 }, { 14, 21, 65 }, { 16, 71, 78 },
    { 17, 30, 76 }, { 18, 25, 80 }, { 19, 61, 83 }, { 22, 38, 77 },
    { 23, 41, 50 }, { 7,  26, 58 }, { 29, 32, 81 }, { 33, 40, 73 },
    { 18, 34, 48 }, { 13, 42, 64 }, { 5,  26, 43 }, { 47, 69, 72 },
    { 54, 55, 70 }, { 45, 62, 68 }, { 10, 63, 67 }, { 14, 66, 72 },
    { 22, 60, 74 }, { 35, 39, 79 }, { 1,  46, 64 }, { 1,  24, 66 },
    { 2,  5,  70 }, { 3,  31, 65 }, { 4,  49, 58 }, { 1,  4,  5  },
    { 6,  60, 67 }, { 7,  32, 75 }, { 8,  48, 82 }, { 9,  35, 41 },
    { 10, 39, 62 }, { 11, 14, 61 }, { 12, 71, 74 }, { 13, 23, 78 },
    { 11, 35, 55 }, { 15, 16, 79 }, { 7,  9,  16 }, { 17, 54, 63 },
    { 18, 50, 57 }, { 19, 30, 47 }, { 20, 64, 80 }, { 21, 28, 69 },
    { 22, 25, 43 }, { 13, 22, 37 }, { 2,  47, 51 }, { 23, 54, 74 },
    { 26, 34, 72 }, { 27, 36, 37 }, { 21, 36, 63 }, { 29, 40, 44 },
    { 19, 26, 57 }, { 3,  46, 82 }, { 14, 15, 58 }, { 33, 52, 53 },
    { 30, 43, 52 }, { 6,  9,  52 }, { 27, 33, 65 }, { 25, 69, 73 },
    { 38, 55, 83 }, { 20, 39, 77 }, { 18, 29, 56 }, { 32, 48, 71 },
    { 42, 51, 59 }, { 28, 44, 79 }, { 34, 60, 62 }, { 31, 45, 61 },
    { 46, 68, 77 }, { 6,  24, 76 }, { 8,  10, 78 }, { 40, 41, 70 },
    { 17, 50, 53 }, { 42, 66, 68 }, { 4,  22, 72 }, { 36, 64, 81 },
    { 13, 29, 47 }, { 2,  8,  81 }, { 56, 67, 73 }, { 5,  38, 50 },
    { 12, 38, 64 }, { 59, 72, 80 }, { 3,  26, 79 }, { 45, 76, 81 },
    { 1,  65, 74 }, { 7,  18, 77 }, { 11, 56, 59 }, { 14, 39, 54 },
    { 16, 37, 66 }, { 10, 28, 55 }, { 15, 60, 70 }, { 17, 25, 82 },
    { 20, 30, 31 }, { 12, 67, 68 }, { 23, 75, 80 }, { 27, 32, 62 },
    { 24, 69, 75 }, { 19, 21, 71 }, { 34, 53, 61 }, { 35, 46, 47 },
    { 33, 59, 76 }, { 40, 43, 83 }, { 41, 42, 63 }, { 49, 75, 83 },
    { 20, 44, 48 }, { 42, 49, 57 }
};

// One thread per (time_sub, freq_sub, time_offset, freq_offset).
//
// Grid layout:
//   blockIdx.x * blockDim.x + threadIdx.x = freq_offset within [bin_start, bin_start+scan_bins)
//   blockIdx.y = sub * time_off_range + time_off
//                sub      = time_sub * freq_osr + freq_sub  (slow axis)
//                time_off = 0..time_off_range-1              (fast axis)
//
// Shared memory layout: (NUM_DATA_SLOTS + NUM_SYNC_SLOTS) × WINDOW bytes.
//   slot_idx  0..28 → data group 1: time_off+7  .. time_off+35
//   slot_idx 29..57 → data group 2: time_off+43 .. time_off+71
//   slot_idx 58..64 → sync group 1: time_off+0  .. time_off+6
//   slot_idx 65..71 → sync group 2: time_off+36 .. time_off+42
//   slot_idx 72..78 → sync group 3: time_off+72 .. time_off+78
//
// A candidate must pass BOTH the syndrome check (hard-decision LDPC parity, data tones)
// AND the Costas sync match (soft argmax vs. known Costas pattern, sync tones).
__global__ void ft8SyndromeScanKernel(
    const uint8_t* __restrict__ mag,
    int snap_start, int ring_size,
    int32_t*  __restrict__ cand_fo,
    uint8_t*  __restrict__ cand_to,
    uint8_t*  __restrict__ cand_ts,
    uint8_t*  __restrict__ cand_fs,
    uint32_t* __restrict__ cand_count,
    uint32_t max_cands,
    int total_bins, int scan_bins, int bin_start,
    int num_blocks, int time_osr, int freq_osr,
    int max_syn_errors, int min_sync_matches,
    int time_off_range)
{
    const int BLOCK_SZ    = blockDim.x;
    const int WINDOW      = BLOCK_SZ + 8;
    const int TOTAL_SLOTS = NUM_DATA_SLOTS + NUM_SYNC_SLOTS;  // 79
    const int fo_base     = bin_start + blockIdx.x * BLOCK_SZ;  // absolute bin
    const int fo          = fo_base + (int)threadIdx.x;

    int combo    = blockIdx.y;
    int sub      = combo / time_off_range;
    int time_off = combo % time_off_range;
    int time_sub = sub / freq_osr;
    int freq_sub = sub % freq_osr;

    const int block_stride = time_osr * freq_osr * total_bins;
    const int sub_offset   = sub * total_bins;

    extern __shared__ uint8_t smem[];  // TOTAL_SLOTS * WINDOW bytes

    // Cooperatively load all 79 symbol windows (58 data + 21 sync) into shared memory.
    const int SMEM_TOTAL = TOTAL_SLOTS * WINDOW;
    for (int i = (int)threadIdx.x; i < SMEM_TOTAL; i += BLOCK_SZ) {
        int slot_idx = i / WINDOW;
        int fo_off   = i % WINDOW;
        int sym_off;
        if      (slot_idx < 29) sym_off = slot_idx + 7;           // data group 1: +7..+35
        else if (slot_idx < 58) sym_off = slot_idx - 29 + 43;     // data group 2: +43..+71
        else {
            int s = slot_idx - 58;                                 // 0..20
            sym_off = (s < 7) ? s : (s < 14) ? (s - 7 + 36) : (s - 14 + 72);
        }
        int block_abs = time_off + sym_off;
        uint8_t val   = 0;
        if (block_abs < num_blocks) {
            int ring_slot = (snap_start + block_abs) % ring_size;
            int global_fo = fo_base + fo_off;
            if (global_fo < total_bins)
                val = mag[(size_t)ring_slot * block_stride + sub_offset + global_fo];
        }
        smem[i] = val;
    }
    __syncthreads();

    // Threads beyond the scan window helped load smem; exit now.
    if (fo >= bin_start + scan_bins - 7) return;

    // --- Syndrome check (data tones) ---
    // Accumulate LDPC parity into 3 packed uint32_t registers (83 check bits total).
    //   syn0: checks  0-31    syn1: checks 32-63    syn2: checks 64-82
    uint32_t syn0 = 0, syn1 = 0, syn2 = 0;

    for (int k = 0; k < NUM_DATA_SLOTS; ++k) {
        int si = k * WINDOW + (int)threadIdx.x;

        uint8_t best = smem[si];
        int best_t = 0;
        for (int t = 1; t < 8; ++t) {
            uint8_t v = smem[si + t];
            if (v > best) { best = v; best_t = t; }
        }

        uint8_t j = kFT8_inv_Gray_map_d[best_t];
        for (int b = 0; b < 3; ++b) {
            if ((j >> (2 - b)) & 1) {
                int bit_idx = k * 3 + b;
                for (int q = 0; q < 3; ++q) {
                    int chk = (int)kFTX_LDPC_Mn_d[bit_idx][q] - 1;
                    if      (chk < 32) syn0 ^= 1u << chk;
                    else if (chk < 64) syn1 ^= 1u << (chk - 32);
                    else               syn2 ^= 1u << (chk - 64);
                }
            }
        }
    }

    int fails = __popc(syn0) + __popc(syn1) + __popc(syn2 & 0x0007FFFFu);
    if (fails > max_syn_errors) return;

    // --- Costas sync match (sync tones) ---
    // For each of the 21 sync positions, check whether the argmax tone matches
    // the expected Costas pattern.  Positions zeroed by out-of-range boundary
    // will argmax to tone 0, which the Costas pattern rarely expects.
    int sync_matches = 0;
    const int SYNC_BASE = NUM_DATA_SLOTS * WINDOW;
    for (int s = 0; s < NUM_SYNC_SLOTS; ++s) {
        int si = SYNC_BASE + s * WINDOW + (int)threadIdx.x;
        uint8_t best = smem[si];
        int best_t = 0;
        for (int t = 1; t < 8; ++t) {
            if (smem[si + t] > best) { best = smem[si + t]; best_t = t; }
        }
        if (best_t == (int)kFT8_Costas_d[s % 7])
            ++sync_matches;
    }
    if (sync_matches < min_sync_matches) return;

    uint32_t idx = atomicAdd(cand_count, 1u);
    if (idx < max_cands) {
        cand_fo[idx] = fo;
        cand_to[idx] = (uint8_t)time_off;
        cand_ts[idx] = (uint8_t)time_sub;
        cand_fs[idx] = (uint8_t)freq_sub;
    }
}

void ft8_syndrome_scan(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    int32_t*  cand_fo_d,
    uint8_t*  cand_to_d,
    uint8_t*  cand_ts_d,
    uint8_t*  cand_fs_d,
    uint32_t* cand_count_d,
    uint32_t  max_cands,
    int total_bins, int scan_bins, int bin_start,
    int num_blocks, int time_osr, int freq_osr,
    int max_syn_errors, int min_sync_matches,
    cudaStream_t stream)
{
    // FT8 message spans 79 symbols (NUM_DATA_SLOTS + NUM_SYNC_SLOTS).
    // Search all starting positions where the full message fits in the window.
    const int FT8_SYMBOLS = NUM_DATA_SLOTS + NUM_SYNC_SLOTS;  // 79
    const int time_off_range = (num_blocks > FT8_SYMBOLS) ? (num_blocks - FT8_SYMBOLS + 1) : 1;

    const int BLOCK_SZ   = 256;
    const int smem_bytes = FT8_SYMBOLS * (BLOCK_SZ + 8);  // 79 * 264 = 20856 bytes
    int grid_x = (scan_bins + BLOCK_SZ - 1) / BLOCK_SZ;
    int grid_y = time_off_range * time_osr * freq_osr;
    dim3 grid(grid_x, grid_y);

    ft8SyndromeScanKernel<<<grid, BLOCK_SZ, smem_bytes, stream>>>(
        mag_d, snap_start, ring_size,
        cand_fo_d, cand_to_d, cand_ts_d, cand_fs_d, cand_count_d, max_cands,
        total_bins, scan_bins, bin_start,
        num_blocks, time_osr, freq_osr, max_syn_errors, min_sync_matches,
        time_off_range);
}
