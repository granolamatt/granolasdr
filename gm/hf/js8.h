#pragma once
#include <functional>
#include <mutex>
#include <string>
#include <unordered_set>
#include <zmq.hpp>

namespace gm { namespace cuda { struct ContScanResult; } }

namespace gm {
namespace hf {

class JS8 {
public:
    explicit JS8(int zmq_port = 5590);
    ~JS8() = default;

    // Publish one decoded JS8 message; thread-safe.
    // from_call: the sending station's callsign (may be "" if not extractable).
    void publishDecoded(const char* text, const char* from_call, float freq_hz,
                        float snr, double unix_time, float time_offset);

    // Callback invoked by JS8Cuda's worker thread for each scan batch.
    // Does CPU BP decode → CRC-12 → message extract → ZMQ publish.
    void decodeAndPublishContinuous(gm::cuda::ContScanResult& r);

    // Optional SSE broadcast callback (same signature as FT8's).
    void setBroadcastCallback(std::function<void(const char*, float, float, double)> cb) {
        broadcast_callback_ = std::move(cb);
    }

private:
    int zmq_port_;
    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mutex_;
    std::function<void(const char*, float, float, double)> broadcast_callback_;

    // Per-epoch dedup: cleared when JS8 epoch (floor(unix/15)) rolls over.
    std::unordered_set<std::string> seen_this_epoch_;
    uint64_t last_epoch_ = 0;
};

} // namespace hf
} // namespace gm
