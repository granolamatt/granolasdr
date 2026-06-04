#pragma once
#include <cuda_runtime.h>
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
class WaterfallCuda : public Thread {
public:
    static constexpr int DEFAULT_OUT_BINS = 2048;
    static constexpr int WS_PORT          = 8765;
    static constexpr int RING_KEYS        = 8;
    static constexpr int ROWS_PER_SLOT    = FT8_TIME_OSR;  // 4

    // wf_floor / wf_ceil: uint8 magnitude window for colormap normalization.
    // Values outside this range are clamped. Typical HF noise floor ≈ 200,
    // strong signals ≈ 240; start with floor=195 ceil=248 and adjust as needed.
    WaterfallCuda(const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring,
                  int bin_start, int bin_end,
                  int out_bins  = DEFAULT_OUT_BINS,
                  int ws_port   = WS_PORT,
                  uint8_t wf_floor = 195,
                  uint8_t wf_ceil  = 248);
    ~WaterfallCuda();

    void run();
    void stop() { setRunning(false); }

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring_;
    int     bin_start_;
    int     bin_end_;
    int     out_bins_;
    int     ws_port_;
    uint8_t wf_floor_;
    uint8_t wf_ceil_;

    cudaStream_t stream_{};
    cudaEvent_t  ready_{};

    // Device/pinned output: ROWS_PER_SLOT × out_bins × 4 bytes (RGBA).
    uint8_t* rgba_d_{nullptr};
    uint8_t* rgba_h_{nullptr};
};

} // namespace cuda
} // namespace gm
