#pragma once
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include "gm/buffer/DeviceRingBuffer.h"

namespace gm {
namespace cuda {

// CWSkimmerCuda: reads the CW MagBlock ring (50 Hz/bin, 5 ms rows over the
// 819.2 kHz CW composite), detects keyed CW carriers (per-bin peak-vs-average
// over a sliding window), extracts each active bin's magnitude envelope, and
// runs the pure-CPU cw_morse decoder.  A detected MagBlock bin maps back to RF
// via kCWBands (bin k -> composite bin c = k/2 -> band offset -> Hz).
//
// Block contract: owns its own cudaStream_t and worker thread; reads the ring
// via write_idx + ready (never blocks the writer).  Detection + decode feed a
// CwTracker (gm/hf/cw_track) that keeps one Track per carrier across windows —
// following drift, suppressing sidebands, confirming a callsign only when the
// same carrier decodes it repeatedly.  Still Phase 2/3: ZMQ/wsdict publish, RBN.
template<int N>
class CWSkimmerCuda {
public:
    explicit CWSkimmerCuda(const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring,
                           const char* label = "CW");
    ~CWSkimmerCuda();

    void start();
    void stop();

private:
    void   worker();
    double binToHz(int magbin) const;   // demap MagBlock bin -> RF Hz (0 = none)

    const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring_;
    const char*          label_;
    cudaStream_t         stream_{};
    std::thread          thread_;
    std::atomic<bool>    running_{false};
    std::vector<uint8_t> host_win_;     // WSLOTS * slot_bytes snapshot
    int                  content_bins_{0}; // 2·Σ CW bw: MagBlock bins carrying signal
};

extern template class CWSkimmerCuda<256>;

} // namespace cuda
} // namespace gm
