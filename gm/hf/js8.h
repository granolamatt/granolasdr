#pragma once
#include <mutex>
#include <string>
#include <unordered_set>
#include <zmq.hpp>

namespace gm { namespace cuda { struct ContScanResult; } }
namespace gm { namespace cuda { class JS8CudaBase; } }

namespace gm {
namespace hf {

class JS8 {
public:
    // symbol_period: seconds per symbol (0.160f Normal, 0.100f Fast).
    // cycle_secs   : TX cycle length for dedup epoch (15.0f Normal, 10.0f Fast).
    // time_osr     : time over-sampling ratio used in the scan (4 Normal, 2 Fast).
    // rfft_size    : FFT length that produced freq-offset bins (1048576 Normal, 655360 Fast).
    // mode_name    : value for "mode" field in ZMQ JSON ("JS8" Normal, "JS8-FAST" Fast).
    JS8(gm::cuda::JS8CudaBase* js8cuda, int zmq_port = 5590,
        float symbol_period = 0.160f, float cycle_secs = 15.0f,
        int time_osr = 4, int rfft_size = 1048576,
        const char* mode_name = "JS8");
    ~JS8() = default;

    // Publish one decoded JS8 message; thread-safe.
    // from_call: the sending station's callsign (may be "" if not extractable).
    void publishDecoded(const char* text, const char* from_call, float freq_hz,
                        float snr, double unix_time, float time_offset);

    // Callback invoked by JS8Cuda's worker thread for each scan batch.
    // Does CPU BP decode → CRC-12 → message extract → ZMQ publish.
    void decodeAndPublishContinuous(gm::cuda::ContScanResult& r);

private:
    int         zmq_port_;
    float       symbol_period_;
    float       cycle_secs_;
    int         time_osr_;
    int         rfft_size_;
    std::string mode_name_;

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mutex_;

    // Per-epoch dedup: cleared when JS8 epoch rolls over.
    std::unordered_set<std::string> seen_this_epoch_;
    uint64_t last_epoch_ = 0;
};

} // namespace hf
} // namespace gm
