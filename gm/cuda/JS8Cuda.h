#pragma once
#include <atomic>
#include <functional>
#include <thread>
#include <cuda_runtime.h>
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/JS8ScanCuda.h"

namespace gm {
namespace cuda {

// JS8 Normal decode orchestrator.
//
// Reads from FT8Cuda's shared mag ring — no second RFFT or ring allocation.
// Runs its own cont scan loop (stride=6, ~1/sec) on a separate CUDA stream.
// For each scan batch with passing candidates, decode_callback is invoked
// from the worker thread with a ContScanResult containing pinned-host LLRs
// (log174) and candidate positions.  The callback (js8.cc) performs CPU BP
// decode, CRC-12, message extract, and ZMQ publish.
class JS8Cuda {
public:
    explicit JS8Cuda(FT8Cuda* ft8, float min_score = 5.0f);
    ~JS8Cuda();

    void setDecodeCallback(std::function<void(ContScanResult&)> cb) {
        decode_callback_ = std::move(cb);
    }

    void start();
    void stop();

private:
    FT8Cuda* ft8_;
    float    min_score_;

    cudaStream_t js8_scan_stream_{};

    static constexpr uint32_t CAND_MAX = JS8_GPU_CAND_MAX;

    // Scan candidate output buffers (device only; slots hold per-slot copies).
    int32_t*  cand_fo_d_{nullptr};
    uint8_t*  cand_to_d_{nullptr};
    uint8_t*  cand_ts_d_{nullptr};
    uint8_t*  cand_fs_d_{nullptr};
    int16_t*  cand_score_d_{nullptr};
    uint32_t* cand_count_d_{nullptr};
    float*    log174_d_{nullptr};

    // Double-buffered continuous scan slots.
    static constexpr int CONTINUOUS_SLOTS = 8;
    ContScanResult cont_slots_[CONTINUOUS_SLOTS];
    std::atomic<uint64_t> cont_write_idx_{0};
    std::atomic<uint64_t> cont_read_idx_{0};

    std::atomic<bool> running_{false};
    std::thread scan_thread_;
    std::thread worker_thread_;

    std::function<void(ContScanResult&)> decode_callback_;

    void allocSlots();
    void freeSlots();
    void scanLoop();
    void workerLoop();
};

} // namespace cuda
} // namespace gm
