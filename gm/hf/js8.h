#pragma once
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <zmq.hpp>
#include "wsdict.h"
#include "gm/hf/llr_capture.h"

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

    // Enable/configure the OSD fallback (off by default).  Runs only on SP
    // parity failures whose sync score >= score_floor, deduped by frequency bin,
    // capped at max_per_cycle per scan.  soft_thresh > 0 adds a soft-distance
    // gate on top of CRC-12 (0 = rely on CRC alone).
    void setOsdConfig(bool enable, int order, float score_floor,
                      int max_per_cycle, float soft_thresh = 0.0f);

    // Enable the per-candidate refine fallback (needs the complex retention ring;
    // Normal only). Runs after OSD fails on a strong candidate.
    void setRefineEnabled(bool enable) { refine_enable_ = enable; }

private:
    // CRC-12 + dedup + publish for one candidate; `info` is the 87-bit message
    // (codeword + 87).  Shared by the SP-converged ("pass") and OSD ("osd") paths.
    // Captures labeled LLR training data on success.  Returns true if CRC passed
    // (a real decode), even if a duplicate suppressed re-publishing.
    bool publishCandidate(gm::cuda::ContScanResult& r, uint32_t i,
                          const uint8_t* info, const float* llr, double unix_now,
                          const char* status, const float* osd_dist);

    gm::cuda::JS8CudaBase* js8cuda_;   // for the refine fallback
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

    LlrCapture llr_capture_;

    // OSD fallback config (see setOsdConfig). Disabled by default.
    bool  osd_enable_        = false;
    int   osd_order_         = 2;
    float osd_score_floor_   = 0.0f;   // Costas sync score; tighter than min_score
    int   osd_max_per_cycle_ = 64;
    float osd_soft_thresh_   = 0.0f;   // 0 = gate on CRC-12 only

    // Per-candidate refine fallback (see setRefineEnabled). Off by default.
    bool  refine_enable_ = false;
};

} // namespace hf
} // namespace gm
