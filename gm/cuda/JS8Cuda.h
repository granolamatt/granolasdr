#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <thread>
#include <cuda_runtime.h>
#include <zmq.hpp>
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/JS8ScanCuda.h"
#include "gm/cuda/JS8FastScanCuda.h"
#include "gm/buffer/DeviceRingBuffer.h"

namespace gm {
namespace cuda {

// Non-template base: exposes only what JS8 (the CPU decode class) needs.
class JS8CudaBase {
public:
    virtual ~JS8CudaBase() = default;
    virtual void setDecodeCallback(std::function<void(ContScanResult&)> cb) = 0;
    virtual void start() = 0;
    virtual void stop()  = 0;
    // Per-candidate freq/time refine (fallback). Fills log174[174] from the
    // retained complex frame with continuous alignment + kJS8Refine LLRs.
    // Returns false when no complex ring is attached (e.g. Fast) or the frame
    // has aged out. See FT8Cuda::refineCandidate for the shared design.
    virtual bool refineCandidate(int32_t fo, int to, uint64_t snap_start,
                                 float* log174) = 0;
};

// Scan kernel function pointer — matches js8_gpu_scan and js8_fast_gpu_scan.
using JS8ScanFn = void(*)(
    const uint8_t*, int, int,
    int32_t*, uint8_t*, uint8_t*, uint8_t*,
    int16_t*, uint32_t*, uint32_t,
    int, int, int, int, float,
    cudaStream_t,
    bool);

// JS8 decode orchestrator. N = ring depth (200 for Normal, 100 for Fast).
// scan_fn   : js8_gpu_scan (ORIGINAL Costas) or js8_fast_gpu_scan (MODIFIED Costas).
// time_osr  : time over-sampling ratio (4 for Normal, 2 for Fast).
// freq_osr  : freq over-sampling ratio (4 for Normal, 2 for Fast).
// cap_blocks: symbols+guard window in ring slots (106 for Normal, 100 for Fast).
template<int N>
class JS8Cuda : public JS8CudaBase {
public:
    // cplx_ring + slot_cplx_idx (optional): MagBlock's complex-composite
    // retention ring for the refine fallback (Normal/N=200 only). Null disables.
    explicit JS8Cuda(const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring,
                     float min_score, int zmq_port,
                     JS8ScanFn scan_fn,
                     int time_osr, int freq_osr, int cap_blocks,
                     const char* label = "JS8",
                     bool legacy_costas = false,
                     const gm::buffer::DeviceRingBuffer<std::complex<float>,
                           gm::buffer::kComplexCompositeBlocks>* cplx_ring = nullptr,
                     const uint64_t* slot_cplx_idx = nullptr);
    ~JS8Cuda() override;

    void setDecodeCallback(std::function<void(ContScanResult&)> cb) override {
        decode_callback_ = std::move(cb);
    }

    void start() override;
    void stop()  override;

    bool refineCandidate(int32_t fo, int to, uint64_t snap_start,
                         float* log174) override;

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring_;
    float       min_score_;
    bool        legacy_costas_;

    // Complex-composite retention for the refine fallback (null = disabled).
    const gm::buffer::DeviceRingBuffer<std::complex<float>,
          gm::buffer::kComplexCompositeBlocks>* cplx_ring_{nullptr};
    const uint64_t* slot_cplx_idx_{nullptr};
    std::vector<std::complex<float>> refine_host_;   // D2H raw frame (~41 MB)
    std::vector<std::complex<float>> refine_decim_;  // decimated baseband frame
    JS8ScanFn   scan_fn_;
    int         time_osr_;
    int         freq_osr_;
    int         cap_blocks_;
    const char* label_;

    cudaStream_t js8_scan_stream_{};

    static constexpr uint32_t CAND_MAX = JS8_GPU_CAND_MAX;

    int32_t*  cand_fo_d_{nullptr};
    uint8_t*  cand_to_d_{nullptr};
    uint8_t*  cand_ts_d_{nullptr};
    uint8_t*  cand_fs_d_{nullptr};
    int16_t*  cand_score_d_{nullptr};
    uint32_t* cand_count_d_{nullptr};
    float*    log174_d_{nullptr};

    static constexpr int CONTINUOUS_SLOTS = 8;
    ContScanResult cont_slots_[CONTINUOUS_SLOTS];
    std::atomic<uint64_t> cont_write_idx_{0};
    std::atomic<uint64_t> cont_read_idx_{0};

    std::atomic<bool> running_{false};
    std::thread scan_thread_;
    std::thread worker_thread_;

    std::function<void(ContScanResult&)> decode_callback_;

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
extern template class JS8Cuda<128>;

} // namespace cuda
} // namespace gm
