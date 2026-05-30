#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <cuda_runtime.h>
#include <zmq.hpp>
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/JS8ScanCuda.h"
#include "gm/buffer/DeviceRingBuffer.h"

namespace gm {
namespace cuda {

// JS8 Normal decode orchestrator.
//
// Reads from the shared mag ring (DeviceRingBuffer) — no second RFFT.
// Runs its own cont scan loop (stride=6, ~1/sec) on a separate CUDA stream.
class JS8Cuda {
public:
    explicit JS8Cuda(const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring,
                     float min_score = 5.0f, int zmq_port = 0);
    ~JS8Cuda();

    void setDecodeCallback(std::function<void(ContScanResult&)> cb) {
        decode_callback_ = std::move(cb);
    }

    void start();
    void stop();

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring_;
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

    // Dispatch timestamp per slot (steady_clock ns), set by scanLoop, read by workerLoop.
    int64_t slot_dispatch_ns_[CONTINUOUS_SLOTS]{};

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mu_;

    void allocSlots();
    void freeSlots();
    void scanLoop();
    void workerLoop();
};

} // namespace cuda
} // namespace gm
