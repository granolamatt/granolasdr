#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <zmq.hpp>
#include "wsdict.h"

namespace gm { namespace cuda { struct ContScanResult; } }
namespace gm { namespace cuda { class JS8CudaBase; } }

namespace gm {
namespace hf {

class JS8 {
public:
    // symbol_period: seconds per ring block (0.160f Normal, 0.100f Fast).
    // cycle_secs   : TX cycle length for dedup epoch (15.0f Normal, 10.0f Fast).
    // time_osr     : time over-sampling ratio used in the scan (4 Normal, 2 Fast).
    // rfft_size    : FFT length that produced freq-offset bins (32768 Normal, 20480 Fast).
    // mode_name    : label for log/JSON output ("JS8", "JS8 Fast", "JS8 Slow").
    // wsdict_port  : wsdict server port (0 = disabled).
    JS8(gm::cuda::JS8CudaBase* js8cuda, int zmq_port = 5590,
        float symbol_period = 0.160f, float cycle_secs = 15.0f,
        int time_osr = 4, int rfft_size = 32768,
        const char* mode_name = "JS8",
        int wsdict_port = 0);
    ~JS8() = default;

    void publishDecoded(const char* text, const char* from_call, float freq_hz,
                        float snr, double unix_time, float time_offset);

    void decodeAndPublishContinuous(gm::cuda::ContScanResult& r);

private:
    int         zmq_port_;
    float       symbol_period_;
    float       cycle_secs_;
    int         time_osr_;
    int         rfft_size_;
    const char* mode_name_;

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mutex_;

    std::unique_ptr<WsDictClient> ws_client_;
    std::string wsdict_key_;          // e.g. "granolasdr:js8:heard:" (callsign appended per publish)
    std::mutex  ws_mutex_;

    std::unordered_set<std::string> seen_this_epoch_;
    uint64_t last_epoch_ = 0;
};

} // namespace hf
} // namespace gm
