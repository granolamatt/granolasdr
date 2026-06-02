// Validate js8_fast_gpu_scan() kernel with a synthetic MODIFIED Costas signal.
//
// Builds a 116-slot mag ring (ring_size=116, num_blocks=108, time_osr=2,
// freq_osr=2, num_bins=512) in host memory, injects a MODIFIED Costas pattern
// at fo=100 / time_off=0 / sub=0, copies to GPU, runs the scan kernel, and
// asserts at least one candidate is returned.
//
// This test exercises the GPU kernel directly (not the full pipeline), so it
// catches regressions in JS8FastScanCuda.cu without needing real RF captures.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "gm/cuda/JS8FastScanCuda.h"

static constexpr int RING_SIZE  = 116;
static constexpr int NUM_BLOCKS = 108;
static constexpr int TIME_OSR   = 2;
static constexpr int FREQ_OSR   = 2;
static constexpr int NUM_BINS   = 512;
static constexpr int FO_SIGNAL  = 100;  // inject Costas at this bin offset

static const int kCostasModified[3][7] = {
    {0, 6, 2, 3, 5, 4, 1},
    {1, 5, 0, 2, 3, 6, 4},
    {2, 5, 0, 6, 4, 1, 3},
};

int main() {
    const size_t block_stride = (size_t)TIME_OSR * FREQ_OSR * NUM_BINS;
    const size_t ring_bytes   = (size_t)RING_SIZE * block_stride;

    // Build synthetic ring on host: background=5, signal bins=200.
    std::vector<uint8_t> ring(ring_bytes, 5);

    // Inject MODIFIED Costas at time_off=0, sub=0 (ts=0, fs=0).
    const int sub        = 0;
    const int time_off   = 0;
    const int sub_offset = sub * NUM_BINS;

    for (int m = 0; m < 3; ++m) {
        for (int k = 0; k < 7; ++k) {
            int block_abs = time_off + 36 * m + k;
            if (block_abs >= NUM_BLOCKS) break;
            int ring_slot = block_abs % RING_SIZE;
            int tone      = kCostasModified[m][k];
            // Set signal tone to 200, adjacent tones to 0 so score is unambiguous.
            for (int j = 0; j < 8; ++j) {
                ring[ring_slot * block_stride + sub_offset + FO_SIGNAL + j] =
                    (j == tone) ? 200 : 0;
            }
        }
    }

    // Upload ring to GPU.
    uint8_t* mag_d = nullptr;
    cudaMalloc(&mag_d, ring_bytes);
    cudaMemcpy(mag_d, ring.data(), ring_bytes, cudaMemcpyHostToDevice);

    // Allocate candidate output buffers.
    static constexpr uint32_t CAND_MAX = 2048;
    int32_t*  cand_fo_d    = nullptr;
    uint8_t*  cand_to_d    = nullptr;
    uint8_t*  cand_ts_d    = nullptr;
    uint8_t*  cand_fs_d    = nullptr;
    int16_t*  cand_score_d = nullptr;
    uint32_t* cand_count_d = nullptr;

    cudaMalloc(&cand_fo_d,    CAND_MAX * sizeof(int32_t));
    cudaMalloc(&cand_to_d,    CAND_MAX * sizeof(uint8_t));
    cudaMalloc(&cand_ts_d,    CAND_MAX * sizeof(uint8_t));
    cudaMalloc(&cand_fs_d,    CAND_MAX * sizeof(uint8_t));
    cudaMalloc(&cand_score_d, CAND_MAX * sizeof(int16_t));
    cudaMalloc(&cand_count_d, sizeof(uint32_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    js8_fast_gpu_scan(
        mag_d, /*snap_start=*/0, RING_SIZE,
        cand_fo_d, cand_to_d, cand_ts_d, cand_fs_d, cand_score_d, cand_count_d, CAND_MAX,
        NUM_BINS, NUM_BLOCKS, TIME_OSR, FREQ_OSR, /*min_score=*/1.0f,
        stream);

    cudaStreamSynchronize(stream);

    cudaError_t kerr = cudaGetLastError();
    if (kerr != cudaSuccess) {
        fprintf(stderr, "FAIL: CUDA error after scan: %s\n", cudaGetErrorString(kerr));
        return 1;
    }

    uint32_t count = 0;
    cudaMemcpy(&count, cand_count_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    printf("js8_fast_gpu_scan: %u candidate(s) found\n", count);

    if (count == 0) {
        fprintf(stderr, "FAIL: no candidates found for synthetic MODIFIED Costas signal\n");
        return 1;
    }

    // Read back first candidate (atomicAdd order, not sorted by score).
    int32_t fo0 = 0;
    int16_t sc0 = 0;
    cudaMemcpy(&fo0, cand_fo_d,    sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sc0, cand_score_d, sizeof(int16_t), cudaMemcpyDeviceToHost);
    printf("First candidate: fo=%d score=%d (signal injected at fo=%d)\n", fo0, (int)sc0, FO_SIGNAL);

    cudaFree(mag_d);
    cudaFree(cand_fo_d); cudaFree(cand_to_d); cudaFree(cand_ts_d);
    cudaFree(cand_fs_d); cudaFree(cand_score_d); cudaFree(cand_count_d);
    cudaStreamDestroy(stream);

    printf("PASS\n");
    return 0;
}
