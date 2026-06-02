#ifndef _GM_CUDA_FT8CUDA_H_
#define _GM_CUDA_FT8CUDA_H_

#include <atomic>
#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <cuda.h>
#include <zmq.hpp>
#include "gm/cuda/FT8ScanCuda.h"
#include "gm/cuda/FT8SoftCuda.h"
#include "gm/cuda/ChiFilter.h"
#include "gm/Thread.h"
#include "gm/buffer/DeviceRingBuffer.h"

namespace gm {
namespace cuda {

// Per-slot continuous scan candidate cap.
static const uint32_t CONT_CAND_MAX = 1000;

// One slot in the continuous Costas scan pipeline.
struct ContScanResult {
    uint32_t* count_d{nullptr};
    int32_t*  fo_d{nullptr};
    uint8_t*  to_d{nullptr};
    uint8_t*  ts_d{nullptr};
    uint8_t*  fs_d{nullptr};
    int16_t*  score_d{nullptr};
    float*    log174_d{nullptr};

    uint32_t* count{nullptr};
    int32_t*  fo{nullptr};
    uint8_t*  to{nullptr};
    uint8_t*  ts{nullptr};
    uint8_t*  fs{nullptr};
    int16_t*  score{nullptr};
    float*    log174{nullptr};

    double      timestamp{0.0};
    uint64_t    snap_start{0};  // ring block index at window start (set by launchContScan)
    cudaEvent_t event{};
    std::atomic<bool> dispatched{false};
};

class FT8Cuda : public Thread {
public:
    FT8Cuda(const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring,
            float min_score = 5.0f,
            const std::string& tag = "EPOCH",
            int zmq_port = 0,
            float cfar_multiplier = 0.5f);
    ~FT8Cuda();

    void run();
    void stop() { setRunning(false); }

    void setDecodeCallback(std::function<void(ContScanResult&)> cb);
    void startContinuousScan();

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring_;
    std::string tag_;
    float       min_score_;
    float       cfar_multiplier_;

    static constexpr int CONTINUOUS_SLOTS = 8;
    int cont_stride_{6};

    cudaStream_t cont_scan_stream_{};
    uint8_t*     block_active_d_{nullptr};   // chi pre-filter output [ceil(num_bins/256)]
    uint32_t*    active_blocks_d_{nullptr};  // device counter: # active blocks per scan
    uint32_t*    active_blocks_h_{nullptr};  // pinned host mirror

    // Continuous scan state
    ContScanResult cont_slots_[CONTINUOUS_SLOTS];
    std::atomic<uint64_t> cont_write_idx_{0};
    std::atomic<uint64_t> cont_read_idx_{0};
    std::atomic<bool>     cont_scan_active_{false};
    std::thread           cont_worker_thread_;
    std::function<void(ContScanResult&)> decode_callback_;

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mu_;

    void pubJson(const char* topic, const char* json);
    void allocContSlots();
    void freeContSlots();
    void contWorker();
    void launchContScan(uint64_t wi);
};

} // namespace cuda
} // namespace gm

#endif // _GM_CUDA_FT8CUDA_H_
