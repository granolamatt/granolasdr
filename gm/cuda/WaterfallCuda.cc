#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include "gm/cuda/WaterfallCuda.h"
#include "gm/cuda/WaterfallKernel.h"
#include "gm/cuda/HostCuda.h"
#include "gm/hf/ft8_capture.h"

namespace gm {
namespace cuda {

WaterfallCuda::WaterfallCuda(
    const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring,
    int bin_start, int bin_end, int out_bins, int zmq_port)
    : ring_(ring)
    , bin_start_(bin_start)
    , bin_end_(bin_end)
    , out_bins_(out_bins)
    , zmq_ctx_(1)
    , zmq_pub_(zmq_ctx_, ZMQ_PUB)
{
    if (zmq_port > 0)
        zmq_pub_.connect("tcp://localhost:" + std::to_string(zmq_port));

    cuda_check_error(cudaStreamCreate(&stream_));
    cuda_check_error(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));
    cuda_check_error(cudaMalloc((void**)&out_d_, out_bins_));
    cuda_check_error(cudaHostAlloc((void**)&out_h_, out_bins_, cudaHostAllocDefault));

    float hz_per_bin = 6553600.0f / (float)ring_.num_bins;
    printf("Waterfall: bins [%d, %d) = %.0f–%.0f Hz, %d pixels (%.1f Hz/px)\n",
           bin_start_, bin_end_,
           bin_start_ * hz_per_bin, bin_end_ * hz_per_bin,
           out_bins_,
           (float)(bin_end_ - bin_start_) * hz_per_bin / out_bins_);
}

WaterfallCuda::~WaterfallCuda()
{
    if (out_d_) cudaFree(out_d_);
    if (out_h_) cudaFreeHost(out_h_);
    cudaEventDestroy(ready_);
    cudaStreamDestroy(stream_);
}

void WaterfallCuda::run()
{
    uint64_t last_wi = 0;
    bool pending = false;

    while (isRunning()) {
        // Deliver completed frame if one is in flight.
        if (pending && cudaEventQuery(ready_) == cudaSuccess) {
            std::lock_guard<std::mutex> lk(zmq_mu_);
            zmq::message_t topic("waterfall", 9);
            zmq::message_t data(out_h_, out_bins_);
            zmq_pub_.send(topic, zmq::send_flags::sndmore);
            zmq_pub_.send(data,  zmq::send_flags::none);
            pending = false;
        }

        uint64_t wi = ring_.write_idx.load(std::memory_order_acquire);
        if (wi <= last_wi || pending) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        last_wi = wi;

        // Sub-band 0 of the slot just written.
        uint64_t slot_idx = (wi - 1) % 200;
        const uint8_t* slot_base = ring_.base_d + slot_idx * ring_.slot_bytes;

        cudaStreamWaitEvent(stream_, ring_.ready, 0);
        waterfall_decimate(slot_base, bin_start_, bin_end_,
                           out_d_, out_bins_, stream_);
        cudaMemcpyAsync(out_h_, out_d_, out_bins_,
                        cudaMemcpyDeviceToHost, stream_);
        cudaEventRecord(ready_, stream_);
        pending = true;
    }
}

} // namespace cuda
} // namespace gm
