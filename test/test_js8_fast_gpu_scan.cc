// Test: js8_fast_gpu_scan with N=128 ring and MODIFIED Costas arrays
//
// Verifies that the kernel detects the MODIFIED Costas pattern in a 128-slot
// ring with cap_blocks=108. This is the invariant for all non-Normal JS8 modes
// (Fast, Slow, Turbo, Ultra).
//
// Signal planted at: base_fo=32, time_off=0, sub=0 (time_sub=0, freq_sub=0).
// Expected: at least one candidate with fo=32, to=0, ts=0, fs=0.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include "gm/cuda/JS8FastScanCuda.h"

static constexpr int     N          = 128;
static constexpr int     NUM_BINS   = 256;
static constexpr int     TIME_OSR   = 2;
static constexpr int     FREQ_OSR   = 2;
static constexpr int     CAP_BLOCKS = 108;
static constexpr int     BASE_FO    = 32;
static constexpr uint8_t SIGNAL     = 200;
static constexpr uint8_t NOISE      = 10;

// MODIFIED Costas arrays — must match kCostasModified in JS8FastScanCuda.cu
static constexpr int kCostas[3][7] = {
    {0, 6, 2, 3, 5, 4, 1},
    {1, 5, 0, 2, 3, 6, 4},
    {2, 5, 0, 6, 4, 1, 3},
};

int main() {
    const int    BLOCK_STRIDE = TIME_OSR * FREQ_OSR * NUM_BINS;
    const size_t RING_BYTES   = (size_t)N * BLOCK_STRIDE;

    std::vector<uint8_t> ring_h(RING_BYTES, NOISE);

    // Plant MODIFIED Costas tones at time_off=0, sub=0
    for (int m = 0; m < 3; ++m) {
        for (int k = 0; k < 7; ++k) {
            int    block_abs = 36 * m + k;
            int    ring_slot = block_abs % N;
            int    tone_bin  = BASE_FO + kCostas[m][k];
            size_t idx       = (size_t)ring_slot * BLOCK_STRIDE + 0 * NUM_BINS + tone_bin;
            ring_h[idx] = SIGNAL;
        }
    }

    uint8_t* mag_d = nullptr;
    cudaMalloc(&mag_d, RING_BYTES);
    cudaMemcpy(mag_d, ring_h.data(), RING_BYTES, cudaMemcpyHostToDevice);

    static constexpr uint32_t MAX_CANDS = 4096;
    int32_t*  cand_fo_d    = nullptr;
    uint8_t*  cand_to_d    = nullptr;
    uint8_t*  cand_ts_d    = nullptr;
    uint8_t*  cand_fs_d    = nullptr;
    int16_t*  cand_score_d = nullptr;
    uint32_t* cand_count_d = nullptr;
    cudaMalloc(&cand_fo_d,    MAX_CANDS * sizeof(int32_t));
    cudaMalloc(&cand_to_d,    MAX_CANDS * sizeof(uint8_t));
    cudaMalloc(&cand_ts_d,    MAX_CANDS * sizeof(uint8_t));
    cudaMalloc(&cand_fs_d,    MAX_CANDS * sizeof(uint8_t));
    cudaMalloc(&cand_score_d, MAX_CANDS * sizeof(int16_t));
    cudaMalloc(&cand_count_d, sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // min_score=100: noise positions score 0 (cur==best_other), signal scores ~190.
    // Only the planted Costas position at fo=BASE_FO should exceed the threshold.
    js8_fast_gpu_scan(
        mag_d,
        /*snap_start=*/0, /*ring_size=*/N,
        cand_fo_d, cand_to_d, cand_ts_d, cand_fs_d, cand_score_d, cand_count_d,
        MAX_CANDS,
        NUM_BINS, CAP_BLOCKS, TIME_OSR, FREQ_OSR,
        /*min_score=*/100.0f,
        stream,
        /*legacy_costas=*/false);

    cudaStreamSynchronize(stream);

    uint32_t count_h = 0;
    cudaMemcpy(&count_h, cand_count_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Clamp to what was actually stored (kernel caps at MAX_CANDS)
    uint32_t stored = std::min(count_h, MAX_CANDS);
    std::vector<int32_t> fo_h(stored);
    std::vector<uint8_t> to_h(stored), ts_h(stored), fs_h(stored);
    std::vector<int16_t> score_h(stored);
    if (stored > 0) {
        cudaMemcpy(fo_h.data(),    cand_fo_d,    stored * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(to_h.data(),    cand_to_d,    stored * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(ts_h.data(),    cand_ts_d,    stored * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(fs_h.data(),    cand_fs_d,    stored * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(score_h.data(), cand_score_d, stored * sizeof(int16_t), cudaMemcpyDeviceToHost);
    }

    printf("js8_fast_gpu_scan N=128 MODIFIED Costas: %u candidates (stored %u)\n",
           count_h, stored);
    for (uint32_t i = 0; i < stored; ++i)
        printf("  cand[%u]: fo=%d to=%d ts=%d fs=%d score=%d\n",
               i, fo_h[i], to_h[i], ts_h[i], fs_h[i], (int)score_h[i]);

    bool found = false;
    for (uint32_t i = 0; i < stored; ++i) {
        if (fo_h[i] == BASE_FO && to_h[i] == 0 && ts_h[i] == 0 && fs_h[i] == 0) {
            printf("PASS: fo=%d to=0 ts=0 fs=0 score=%d\n", BASE_FO, (int)score_h[i]);
            found = true;
            break;
        }
    }

    if (!found) {
        printf("FAIL: planted signal at fo=%d (to=0 ts=0 fs=0) not found among %u candidates\n",
               BASE_FO, stored);
    }

    cudaFree(mag_d);
    cudaFree(cand_fo_d); cudaFree(cand_to_d); cudaFree(cand_ts_d);
    cudaFree(cand_fs_d); cudaFree(cand_score_d); cudaFree(cand_count_d);
    cudaStreamDestroy(stream);

    return found ? 0 : 1;
}
