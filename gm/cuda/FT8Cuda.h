#ifndef _GM_CUDA_FT8CUDA_H_
#define _GM_CUDA_FT8CUDA_H_

#include <atomic>
#include <functional>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cufft.h>
#include "gm/cuda/HostCuda.h"
#include "gm/cuda/FT8ScanCuda.h"
#include "gm/cuda/FT8SoftCuda.h"
#include "gm/cuda/FT8LdpcCuda.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

// Per-epoch GPU candidate scan results, indexed by decode slot (0..BUFFERS-1).
struct GpuScanResult {
    uint32_t              count;
    std::vector<int32_t>  fo;
    std::vector<uint8_t>  to;
    std::vector<uint8_t>  ts;
    std::vector<uint8_t>  fs;
    std::vector<int16_t>  score;
    std::vector<float>    log174;   // FTX_LDPC_N floats per candidate
    std::vector<uint8_t>  x_hat;   // FTX_LDPC_N bits per candidate (GPU LDPC output)
    std::vector<bool>     parity;  // parity check result per candidate
};

class FT8Cuda : public Thread {
public:
    FT8Cuda(gm::buffer::BufferPosition<std::complex<float>>* inP, bool enable_corpus = false);
    ~FT8Cuda();
    void run();
    void stop() {
        setRunning(false);
    }
    gm::buffer::BufferPosition<uint8_t>* getBuffer() {
        return &rt8BufferPosition;
    }
    // Returns the GPU scan result for the given decode slot (0..BUFFERS-1).
    const GpuScanResult& getGpuScanResult(int slot) const {
        return gpu_results[slot];
    }

private:
    cudaStream_t stream;
    cudaStream_t scan_stream;     // GPU sync score kernel
    cudaStream_t transfer_stream; // D2H mag snapshot (device ring → CPU decode slot)
    cudaStream_t ldpc_stream;     // GPU LDPC decode (low priority, NonBlocking)
    cudaEvent_t  ring_ready;      // fired when device ring D2D writes are committed
    cudaEvent_t  scan_done;       // fired when GPU scan kernel completes
    cudaEvent_t  ldpc_done;       // fired when LDPC batch kernel completes
    cufftHandle rplan;
    double lastepoch;
    const static int BUFFERS = 2;      // number of decode slots for ft8.cc
    const static int RING_BLOCKS = 200; // host rolling ring size in FT8 blocks

    std::ofstream fs;
    int buffer_number;
    uint64_t ring_write_idx;
    uint64_t last_trigger_second;
    std::thread last_snapshot_thread;

    gm::buffer::BufferPosition<uint8_t> rt8BufferPosition;

    gm::cuda::device::HostCuda cuda_h;
    std::vector<size_t> inShape;
    double freqsperbin;
    std::complex<float>* fftData_d;
    gm::buffer::BufferPosition<std::complex<float>>* inPos;
    std::complex<float> *inData_d;
    int doCopy(uint64_t now);
    std::complex<float>* demodData_d;
    std::complex<float>* demodFT8_d;
    std::complex<float>* demodShift_d;
    uint8_t* magFT8_d;
    uint8_t* magFT8;        // 1-byte dummy pinned alloc (keeps rt8BufferPosition pointer valid)

    // GPU-resident mag ring (RING_BLOCKS slots × block_bytes each).
    uint8_t* magFT8_ring_d;

    // Soft symbol LLR buffers (Phase 4).
    float* log174_d;   // device: FT8_GPU_CAND_MAX * kFtxLdpcN floats
    float* log174;     // pinned host staging: same size

    // GPU LDPC output buffers (device).
    uint8_t* x_hat_d;   // FT8_LDPC_BATCH × FTX_LDPC_N decoded bits
    bool*    parity_d;  // FT8_LDPC_BATCH parity flags

    // GPU candidate output buffers (device).
    int32_t*  gpu_cand_fo_d;
    uint8_t*  gpu_cand_to_d;
    uint8_t*  gpu_cand_ts_d;
    uint8_t*  gpu_cand_fs_d;
    int16_t*  gpu_cand_score_d;
    uint32_t* gpu_cand_count_d;

    // Host-side results per decode slot.
    GpuScanResult gpu_results[BUFFERS];

    std::vector<std::vector<uint32_t>> bins;
    size_t rfft_length;
    uint32_t nTune;
    uint32_t nChannels;
    double lastsecond;

    uint32_t buff_pos;

    // Corpus audio capture (enabled with --jtdx).
    bool enable_corpus;
    int  audio_ft8_bin;       // FT8 FFT bin for 20m FT8 dial (14.074 MHz)
    int  audio_ft8_bin_10m;   // FT8 FFT bin for 10m FT8 dial (28.074 MHz)
    int  audio_sample_rate;   // derived sample rate for the audio WAVs (Hz)
    static const int AUDIO_RING_SLOTS = 244; // ≥ 2 × FT8_CAPTURE_BLOCKS (120), prevents snapshot/write race
    std::complex<float>* audioBins_host;      // pinned ring: 20m, AUDIO_RING_SLOTS × FT8_AUDIO_BINS
    std::complex<float>* audioBins_host_10m;  // pinned ring: 10m, AUDIO_RING_SLOTS × FT8_AUDIO_BINS
};
}
}

#endif // _GM_CUDA_FT8CUDA_H_
