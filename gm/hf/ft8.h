#ifndef _GM_HF_FT8_H_
#define _GM_HF_FT8_H_

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <zmq.hpp>
#include "gm/Thread.h"
#include "gm/hf/llr_capture.h"

// Forward-declare CUDA types so ft8.h doesn't pull in CUDA headers.
namespace gm { namespace cuda { class FT8Cuda; } }
namespace gm { namespace cuda { struct ContScanResult; } }
class WsDictClient;

namespace gm {
namespace hf {

class FT8 : public Thread {

public:
    FT8(gm::cuda::FT8Cuda* ft8cuda,
        int zmq_port = 5580,
        int wsdict_port = 0,
        const char* label = "FT8",   // decode-stats tag (e.g. AB-test "FT8-NORM")
        bool log_timing = true);     // false for a parallel/test chain (no CSV)
    ~FT8();
    void run();
    void stop() { setRunning(false); }

    // Decode and publish candidates from the continuous Costas scan path.
    // Called from contWorker via the FT8Cuda decode callback.
    void decodeAndPublishContinuous(gm::cuda::ContScanResult& r);

    // Enable/configure the OSD fallback (off by default).  Runs only on BP
    // failures whose sync score >= score_floor, deduped by frequency bin, capped
    // at max_per_cycle per scan.  soft_thresh > 0 adds a soft-distance gate on top
    // of CRC-14 (0 = rely on CRC alone).
    void setOsdConfig(bool enable, int order, float score_floor,
                      int max_per_cycle, float soft_thresh = 0.0f);

    // Enable the per-candidate refine fallback (needs the complex retention ring).
    // Refine runs on its own worker thread (off the scan callback's critical
    // path), so this also spins that thread up on first enable.
    void setRefineEnabled(bool enable) {
        refine_enable_ = enable;
        if (enable && !refine_thread_.joinable()) {
            refine_run_.store(true);
            refine_thread_ = std::thread([this] { refineWorker(); });
        }
    }

private:
    gm::cuda::FT8Cuda* ft8cuda_;
    int zmq_port_;
    std::string label_{"FT8"};   // decode-stats tag

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mutex_;
    FILE*          timing_log_;
    std::unique_ptr<WsDictClient> ws_client_;

    struct WindowSpot {
        float  snr;
        float  freq_hz;
        double unix_time;
        float  time_offset;
    };

    std::unordered_map<std::string, WindowSpot> window_buf_;
    std::mutex   window_mu_;
    double       window_start_{0.0};

    LlrCapture   llr_capture_;

    // OSD fallback config (see setOsdConfig). Disabled by default.
    bool  osd_enable_        = false;
    int   osd_order_         = 2;
    float osd_score_floor_   = 0.0f;   // Costas sync score; tighter than min_score
    int   osd_max_per_cycle_ = 64;
    float osd_soft_thresh_   = 0.0f;   // 0 = gate on CRC-14 only

    // Per-candidate freq/time refine fallback (see setRefineEnabled). When BP+OSD
    // both fail on a strong-Costas candidate, the scan callback enqueues a job;
    // the refine worker thread re-extracts LLRs from the retained complex frame
    // with continuous alignment and retries BP+OSD — fully off the scan worker's
    // critical path.  Requires MagBlock's complex ring (Normal only); off by default.
    bool  refine_enable_ = false;

    struct RefineJob {
        int32_t  fo;
        int      to;
        int      ts;
        uint64_t snap_start;
        int16_t  score;
        double   timestamp;
    };
    std::deque<RefineJob>   refine_q_;
    std::mutex              refine_mu_;
    std::condition_variable refine_cv_;
    std::thread             refine_thread_;
    std::atomic<bool>       refine_run_{false};
    static constexpr size_t kRefineQueueMax = 32;   // drop-oldest safety valve

    std::mutex decode_mu_;    // guards ftx_message_decode (mutates global hashtable)
    std::mutex capture_mu_;   // guards llr_capture_.write across threads

    void refineWorker();
    // Shared publish/capture helpers (called from the scan callback and the
    // refine worker), so both paths funnel through the same window + capture logic.
    void publishSpot(const char* text, int32_t fo, int to, int ts,
                     int16_t score, double timestamp);
    void captureCand(const char* status, const float* llr, int32_t fo, int to,
                     int ts, int16_t score, double timestamp,
                     const char* text, const float* dist);

    void publishDecoded(const char* callsign, float freq_hz, float snr,
                        double unix_time, float time_offset);
    void flushWindow(double now);
};

}
}

#endif // _GM_HF_FT8_H_
