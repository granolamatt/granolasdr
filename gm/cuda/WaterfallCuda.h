#pragma once
#include <cuda_runtime.h>
#include <string>
#include "gm/Thread.h"
#include "gm/buffer/DeviceRingBuffer.h"
#include "gm/hf/ft8_capture.h"  // FT8_TIME_OSR

namespace gm {
namespace cuda {

// WaterfallCuda: publishes ROWS_PER_SLOT pre-colored RGBA rows to wsdict
// on every MagBlock ring write, using all FT8_TIME_OSR time sub-arrays
// within the slot.  Each sub-array represents a 40 ms time step, giving
// FT8_TIME_OSR × ring_rate ≈ 25 rows/second at 6.25 slots/s.
//
// Output format per wsdict message: FT8_TIME_OSR × out_bins × 4 uint8 RGBA,
// rows in chronological order (row 0 = oldest, row 3 = newest).
//
// Hz-to-bin for the 1,048,576-bin / 6.5536 MHz ring: bin = round(hz / 6.25)
//
// Templated on ring depth N so the same block serves the FT8/JS8 composite ring
// (N=200) and the CW composite ring (N=128).  key_prefix routes each instance to
// its own wsdict key family (granolasdr:waterfall: vs granolasdr:cwwaterfall:);
// full_rate_hz is only used for the startup Hz/px log line.
template<int N = 200>
class WaterfallCuda : public Thread {
public:
    static constexpr int DEFAULT_OUT_BINS = 2048;
    static constexpr int WS_PORT          = 8765;
    static constexpr int RING_KEYS        = 8;
    static constexpr int ROWS_PER_SLOT    = FT8_TIME_OSR;  // 4

    // Colormap: pixel = (mag - noise_ref + wf_offset) * wf_gain.  noise_ref is a
    // low percentile of the slot magnitudes, EMA-tracked, so CW and FT8 waterfalls
    // (different absolute magnitude scales) self-align and share WF_GAIN/WF_OFFSET.
    WaterfallCuda(const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring,
                  int bin_start, int bin_end,
                  int out_bins  = DEFAULT_OUT_BINS,
                  int ws_port   = WS_PORT,
                  const char* key_prefix = "granolasdr:waterfall:",
                  float full_rate_hz = 6553600.0f,
                  int min_publish_ms = 0);   // 0 = publish every ring write; >0 throttles
    ~WaterfallCuda();

    void run();
    void stop() { setRunning(false); }

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring_;
    int         bin_start_;
    int         bin_end_;
    int         out_bins_;
    int         ws_port_;
    float       wf_gain_{6.0f};    // WF_GAIN: contrast (dB-above-noise → brightness)
    int         wf_offset_{0};     // WF_OFFSET: noise-floor lift, uint8 units
    int         noise_pct_{25};    // WF_NOISE_PCT: percentile used as noise floor
    float       noise_ref_{160.0f};// auto-tracked noise floor
    bool        noise_init_{false};
    std::string key_prefix_;
    float       full_rate_hz_;
    int         min_publish_ms_;

    cudaStream_t stream_{};
    cudaEvent_t  ready_{};

    unsigned int* hist_d_{nullptr};   // device: 256-bin magnitude histogram
    unsigned int  hist_h_[256]{};     // host mirror for the percentile estimate

    // Device/pinned output: ROWS_PER_SLOT × out_bins × 4 bytes (RGBA).
    uint8_t* rgba_d_{nullptr};
    uint8_t* rgba_h_{nullptr};
};

} // namespace cuda
} // namespace gm
