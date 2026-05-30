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
#include "gm/cuda/FT8LdpcCuda.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/buffer/DeviceRingBuffer.h"

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
    cudaEvent_t event{};
    std::atomic<bool> dispatched{false};
};

class FT8Cuda : public Thread {
public:
    FT8Cuda(const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring,
            float min_score = 5.0f,
            const std::string& tag = "EPOCH",
            int zmq_port = 0);
    ~FT8Cuda();

    void run();
    void stop() { setRunning(false); }

    gm::buffer::BufferPosition<uint8_t>* getBuffer() { return &rt8BufferPosition_; }

    const GpuScanResult& getGpuScanResult(int slot) const { return gpu_results_[slot]; }

    void setDecodeCallback(std::function<void(ContScanResult&)> cb);
    void startContinuousScan();

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring_;
    std::string tag_;
    float       min_score_;

    static constexpr int BUFFERS          = 2;
    static constexpr int CONTINUOUS_SLOTS = 8;
    int cont_stride_{6};

    cudaStream_t scan_stream_{};
    cudaStream_t transfer_stream_{};
    cudaStream_t ldpc_stream_{};
    cudaStream_t cont_scan_stream_{};
    cudaEvent_t  scan_done_{};
    cudaEvent_t  ldpc_done_{};

    gm::buffer::BufferPosition<uint8_t> rt8BufferPosition_;

    // Soft LLR buffers (epoch path)
    float*    log174_d_{nullptr};
    float*    log174_{nullptr};

    // GPU LDPC output (epoch path)
    uint8_t* x_hat_d_{nullptr};
    bool*    parity_d_{nullptr};

    // GPU candidate buffers (epoch path)
    int32_t*  gpu_cand_fo_d_{nullptr};
    uint8_t*  gpu_cand_to_d_{nullptr};
    uint8_t*  gpu_cand_ts_d_{nullptr};
    uint8_t*  gpu_cand_fs_d_{nullptr};
    int16_t*  gpu_cand_score_d_{nullptr};
    uint32_t* gpu_cand_count_d_{nullptr};

    uint8_t* magFT8_dummy_{nullptr};  // 1-byte pinned dummy for rt8BufferPosition_

    GpuScanResult gpu_results_[BUFFERS];

    int  buffer_number_{0};
    int  consecutive_timeouts_{0};

    // Epoch snapshot state
    uint64_t     last_trigger_second_{0};
    std::thread  last_snapshot_thread_;

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
    void launchEpochScan(uint64_t wi, double seconds);
};

} // namespace cuda
} // namespace gm

#endif // _GM_CUDA_FT8CUDA_H_
