#ifndef _GM_CUDA_FT8CUDA_H_
#define _GM_CUDA_FT8CUDA_H_

#include <atomic>
#include <cmath>
#include <complex>
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
#include "gm/Thread.h"
#include "gm/buffer/DeviceRingBuffer.h"

namespace gm {
namespace cuda {

// Per-slot continuous scan candidate cap.
static const uint32_t CONT_CAND_MAX = 100000;

// One slot in the continuous Costas scan pipeline.
struct ContScanResult {
    uint32_t* count_d{nullptr};
    int32_t*  fo_d{nullptr};
    uint8_t*  to_d{nullptr};
    uint8_t*  ts_d{nullptr};
    uint8_t*  fs_d{nullptr};
    int16_t*  score_d{nullptr};
    float*    log174_d{nullptr};
    uint8_t*  x_hat_d{nullptr};
    bool*     parity_d{nullptr};

    uint32_t* count{nullptr};
    int32_t*  fo{nullptr};
    uint8_t*  to{nullptr};
    uint8_t*  ts{nullptr};
    uint8_t*  fs{nullptr};
    int16_t*  score{nullptr};
    float*    log174{nullptr};
    uint8_t*  x_hat{nullptr};
    bool*     parity{nullptr};

    double      timestamp{0.0};
    uint64_t    snap_start{0};  // ring block index at window start (set by launchContScan)
    cudaEvent_t event{};
    std::atomic<bool> dispatched{false};
};

class FT8Cuda : public Thread {
public:
    // cplx_ring + slot_cplx_idx (optional): MagBlock's complex-composite retention
    // ring and its per-mag-slot map, for the per-candidate refine fallback. Null
    // disables refine (decode path unchanged).
    FT8Cuda(const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring,
            float min_score = 5.0f,
            const std::string& tag = "EPOCH",
            int zmq_port = 0,
            bool legacy_costas = false,
            const gm::buffer::DeviceRingBuffer<std::complex<float>,
                  gm::buffer::kComplexCompositeBlocks>* cplx_ring = nullptr,
            const uint64_t* slot_cplx_idx = nullptr);
    ~FT8Cuda();

    void run();
    void stop() { setRunning(false); }

    void setDecodeCallback(std::function<void(ContScanResult&)> cb);
    void startContinuousScan();

    // Per-candidate refine (fallback): D2H the candidate's complex frame from the
    // retention ring, downconvert+decimate, fine freq/time align, re-extract LLRs.
    // Fills log174[FTX_LDPC_N]. Returns false if refine is disabled or the frame
    // has aged out of the retention window. Caller runs LDPC/CRC on the LLRs.
    bool refineCandidate(int32_t fo, int to, uint64_t snap_start, float* log174);

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring_;
    std::string tag_;
    float       min_score_;
    bool        legacy_costas_;

    // Complex-composite retention for the refine fallback (null = disabled).
    const gm::buffer::DeviceRingBuffer<std::complex<float>,
          gm::buffer::kComplexCompositeBlocks>* cplx_ring_{nullptr};
    const uint64_t* slot_cplx_idx_{nullptr};
    std::vector<std::complex<float>> refine_host_;   // D2H raw frame (~41 MB)
    std::vector<std::complex<float>> refine_decim_;  // decimated baseband frame
    cudaStream_t refine_stream_{};                   // refine D2H (off the scan stream)

    static constexpr int CONTINUOUS_SLOTS = 8;
    int cont_stride_{6};

    cudaStream_t cont_scan_stream_{};

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
