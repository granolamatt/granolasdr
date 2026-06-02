#pragma once
#include <complex>
#include <mutex>
#include <vector>
#include <cuda.h>
#include <cufft.h>
#include <zmq.hpp>
#include "gm/cuda/HostCuda.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/buffer/DeviceRingBuffer.h"

namespace gm {
namespace cuda {

// MagBlock: RFFT + |·|² + uint8 decimation → DeviceRingBuffer + waterfall ZMQ.
// N = ring depth (number of slots). Normal uses N=200; Fast uses N=116.
template<int N>
class MagBlock : public Thread {
public:
    static constexpr int WATERFALL_BINS = 2048;

    explicit MagBlock(gm::buffer::BufferPosition<std::complex<float>>* inP,
                      int rfft_len, int time_osr, int freq_osr,
                      int zmq_port = 0);
    ~MagBlock();

    void run();
    void stop() { setRunning(false); }

    const gm::buffer::DeviceRingBuffer<uint8_t, N>& getRing() const {
        return ring_;
    }

private:
    gm::buffer::BufferPosition<std::complex<float>>* inPos_;
    std::vector<size_t>  inShape_;
    std::complex<float>* inData_d_;

    cudaStream_t    stream_{};
    cudaStream_t    waterfall_stream_{};
    cudaEvent_t     waterfall_ready_{};
    cufftHandle     rplan_{};
    gm::cuda::device::HostCuda cuda_h_;

    size_t   rfft_length_{0};
    int      time_osr_{0};
    int      freq_osr_{0};
    float    bin_hz_{0.0f};
    uint32_t buff_pos_{0};

    std::complex<float>* demodData_d_{nullptr};
    std::complex<float>* demodFT8_d_{nullptr};
    std::complex<float>* demodShift_d_{nullptr};
    uint8_t*             magFT8_d_{nullptr};

    // Waterfall
    uint8_t* waterfall_d_{nullptr};
    uint8_t* waterfall_host_{nullptr};

    gm::buffer::DeviceRingBuffer<uint8_t, N> ring_;

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mu_;

    void pubBin(const char* topic, const void* data, size_t len);
    int  doCopy(uint64_t now);
};

extern template class MagBlock<200>;
extern template class MagBlock<116>;

} // namespace cuda
} // namespace gm
