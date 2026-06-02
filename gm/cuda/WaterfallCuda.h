#pragma once
#include <mutex>
#include <cuda_runtime.h>
#include <zmq.hpp>
#include "gm/Thread.h"
#include "gm/buffer/DeviceRingBuffer.h"

namespace gm {
namespace cuda {

// WaterfallCuda: graph block that decimates the mag ring to a per-slot waterfall
// row and publishes it to ZMQ "waterfall" as a binary uint8 blob.
//
// Constructed with a bin range [bin_start, bin_end) into the 1M-bin ring,
// which maps to out_bins output pixels via linear average pooling.
// Runs every ring write — one waterfall row per ring slot.
//
// Hz to bin: bin = round(hz / 6.25) for the 1,048,576-bin / 6.5 MHz ring.
class WaterfallCuda : public Thread {
public:
    static constexpr int DEFAULT_OUT_BINS = 2048;

    WaterfallCuda(const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring,
                  int bin_start, int bin_end,
                  int out_bins = DEFAULT_OUT_BINS,
                  int zmq_port = 0);
    ~WaterfallCuda();

    void run();
    void stop() { setRunning(false); }

private:
    const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring_;
    int bin_start_;
    int bin_end_;
    int out_bins_;

    cudaStream_t stream_{};
    cudaEvent_t  ready_{};

    uint8_t* out_d_{nullptr};
    uint8_t* out_h_{nullptr};

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mu_;
};

} // namespace cuda
} // namespace gm
