#ifndef _GM_HF_FT8_H_
#define _GM_HF_FT8_H_

#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <zmq.hpp>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "ft8_lib/common/monitor.h"

// Forward-declare CUDA types so ft8.h doesn't pull in CUDA headers.
namespace gm { namespace cuda { class FT8Cuda; } }
namespace gm { namespace cuda { struct ContScanResult; } }

namespace gm {
namespace hf {

class FT8 : public Thread {

public:
    // ft8cuda is optional; non-null enables GPU candidate path.
    // use_gpu_ldpc enables the GPU QP-ADMM LDPC decoder (requires --gpu-ldpc flag).
    FT8(gm::buffer::BufferPosition<uint8_t>* inP,
        gm::cuda::FT8Cuda* ft8cuda = nullptr,
        int zmq_port = 5580,
        bool use_gpu_ldpc = false);
    ~FT8();
    void run();
    void stop() {
        setRunning(false);
    }
    // publish one decoded message; callable from any thread (zmq_mutex_ protected)
    void publishDecoded(const char* callsign, float freq_hz, float snr,
                        double unix_time, float time_offset);

    // Decode and publish candidates from the continuous Costas scan path.
    // Called from contWorker via the FT8Cuda decode callback.
    void decodeAndPublishContinuous(gm::cuda::ContScanResult& r);

    // Register callback invoked by publishDecoded for each decoded message.
    // Used by HFRx.cc to wire SSE broadcast into HFChannelizer.
    void setBroadcastCallback(std::function<void(const char*, float, float, double)> cb) {
        broadcast_callback_ = std::move(cb);
    }

private:
    monitor_t mon;
    gm::buffer::BufferPosition<uint8_t>* inPos;
    gm::cuda::FT8Cuda* ft8cuda;
    bool use_gpu_ldpc;
    int zmq_port_;

    zmq::context_t zmq_ctx;
    zmq::socket_t  zmq_pub;
    std::mutex     zmq_mutex_;
    FILE*          timing_log_;
    std::function<void(const char*, float, float, double)> broadcast_callback_;

    // Dedup for continuous-scan stdout prints: key=(text|freq_bucket) → last print time.
    // Suppresses re-printing the same decode within 20 seconds.
    std::unordered_map<std::string, double> cont_dedup_;
};

}
}

#endif // _GM_HF_FT8_H_