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

// Non-template base: exposes only what JS8 (the CPU decode class) needs.
// Allows JS8 to hold a JS8Cuda<N>* without knowing N.
class JS8CudaBase {
public:
    virtual ~JS8CudaBase() = default;
    virtual void setDecodeCallback(std::function<void(ContScanResult&)> cb) = 0;
    virtual void start() = 0;
    virtual void stop()  = 0;
};

// Scan kernel function pointer type — matches js8_gpu_scan and js8_fast_gpu_scan.
// Parameterizes which Costas pattern (ORIGINAL vs MODIFIED) the scan uses.
using JS8ScanFn = void(*)(
    const uint8_t*, int, int,
    int32_t*, uint8_t*, uint8_t*, uint8_t*,
    int16_t*, uint32_t*, uint32_t,
    int, int, int, int, float,
    cudaStream_t);

// JS8 decode orchestrator. N = ring depth (200 for Normal, 100 for Fast).
// scan_fn   : js8_gpu_scan (ORIGINAL Costas) or js8_fast_gpu_scan (MODIFIED Costas).
// time_osr  : time over-sampling ratio (4 for Normal, 2 for Fast).
// freq_osr  : freq over-sampling ratio (4 for Normal, 2 for Fast).
// cap_blocks: symbols+guard window in ring slots (106 for Normal, 100 for Fast).
template<int N>
class JS8Cuda : public JS8CudaBase {
public:
    explicit JS8Cuda(const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring,
                     float min_score, int zmq_port,
                     JS8ScanFn scan_fn,
                     int time_osr, int freq_osr, int cap_blocks);
    ~JS8Cuda() override;

    void setDecodeCallback(std::function<void(ContScanResult&)> cb) override {
        decode_callback_ = std::move(cb);
    }

    void start() override;
    void stop()  override;

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring_;
    float      min_score_;
    JS8ScanFn  scan_fn_;
    int        time_osr_;
    int        freq_osr_;
    int        cap_blocks_;

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

extern template class JS8Cuda<200>;
extern template class JS8Cuda<100>;

} // namespace cuda
} // namespace gm
