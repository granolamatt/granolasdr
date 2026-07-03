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

// MagBlock: RFFT + |·|² + uint8 decimation → DeviceRingBuffer.
// Owns the mag ring that FT8Cuda, JS8Cuda, and WaterfallCuda read from.
//
// N        : ring depth (slots). Normal=200, Fast/Slow/Turbo/Ultra=128.
// rfft_len : FFT length. Normal=65536 (6.25 Hz/bin), Fast=40960 (10 Hz/bin).
// time_osr : time over-sampling ratio. Normal=4, Fast=2.
// freq_osr : frequency over-sampling ratio. Normal=4, Fast=2.
template<int N>
class MagBlock : public Thread {
public:
    // retain_complex: also keep a rolling ring of the raw complex composite (one
    // slot per input block) so consumers can re-read a full frame with phase for
    // per-candidate refine.  Off by default (extra VRAM); enable on the Normal ring.
    explicit MagBlock(gm::buffer::BufferPosition<std::complex<float>>* inP,
                      int rfft_len, int time_osr, int freq_osr,
                      int zmq_port = 0, bool retain_complex = false);
    ~MagBlock();

    void run();
    void stop() { setRunning(false); }

    const gm::buffer::DeviceRingBuffer<uint8_t, N>& getRing() const {
        return ring_;
    }

    // Complex-composite retention ring (valid only when retain_complex).  Depth
    // in input blocks; one FT8/JS8 frame is 79*32 = 2528 blocks (~12.6 s).
    static constexpr int kComplexBlocks = gm::buffer::kComplexCompositeBlocks;
    const gm::buffer::DeviceRingBuffer<std::complex<float>, kComplexBlocks>&
    getComplexRing() const { return complex_ring_; }
    // Per-mag-slot snapshot of the complex ring block index of that slot's window
    // start.  Index by (mag write_idx % N).  Maps a candidate's mag-slot time to
    // its complex-frame start; the refine's fine time search absorbs any slop.
    const uint64_t* getSlotCplxIdx() const { return slot_cplx_idx_; }
    bool retainsComplex() const { return retain_complex_; }

private:
    gm::buffer::BufferPosition<std::complex<float>>* inPos_;
    std::vector<size_t>  inShape_;
    std::complex<float>* inData_d_;

    cudaStream_t    stream_{};
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

    gm::buffer::DeviceRingBuffer<uint8_t, N> ring_;

    // Complex-composite retention (retain_complex only).
    bool     retain_complex_{false};
    gm::buffer::DeviceRingBuffer<std::complex<float>, kComplexBlocks> complex_ring_;
    uint64_t cplx_wi_{0};                 // complex ring write index (input blocks)
    uint64_t slot_cplx_idx_[N] = {};      // mag-slot -> complex block of window start

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mu_;

    void pubBin(const char* topic, const void* data, size_t len);
    int  doCopy(uint64_t now);
};

extern template class MagBlock<200>;
extern template class MagBlock<128>;

} // namespace cuda
} // namespace gm
