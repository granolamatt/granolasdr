#include "gm/cuda/CWSkimmerCuda.h"
#include "gm/hf/cw_bands.h"
#include "gm/hf/cw_morse.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace gm {
namespace cuda {

template<int N>
CWSkimmerCuda<N>::CWSkimmerCuda(const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring,
                               const char* label)
    : ring_(ring), label_(label) {
    cudaStreamCreate(&stream_);
    uint32_t total = 0;
    for (int i = 0; i < kNumCWBands; ++i) total += kCWBands[i].bw;
    // 50 Hz MagBlock bins, 2 per 100 Hz composite bin -> 2·Σ bw carry signal.
    content_bins_ = std::min((int)(2 * total), (int)ring_.num_bins);
}

template<int N>
CWSkimmerCuda<N>::~CWSkimmerCuda() {
    stop();
    if (stream_) cudaStreamDestroy(stream_);
}

template<int N>
void CWSkimmerCuda<N>::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread(&CWSkimmerCuda<N>::worker, this);
}

template<int N>
void CWSkimmerCuda<N>::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}

template<int N>
double CWSkimmerCuda<N>::binToHz(int magbin) const {
    const int c = magbin / 2;            // 50 Hz MagBlock bin -> 100 Hz composite bin
    int cum = 0;
    for (int b = 0; b < kNumCWBands; ++b) {
        if (c < cum + (int)kCWBands[b].bw)
            return (double)(kCWBands[b].wb_start + (c - cum)) * 100.0;
        cum += (int)kCWBands[b].bw;
    }
    return 0.0;
}

template<int N>
void CWSkimmerCuda<N>::worker() {
    const int    TOSR       = 4;                         // CW time_osr (rows/slot)
    const int    NB         = (int)ring_.num_bins;       // 16384
    const int    WSLOTS     = std::min(100, N - 2);      // window depth (~2 s)
    const int    WROWS      = WSLOTS * TOSR;             // envelope length
    const size_t slot_elems = ring_.slot_bytes;         // bytes == elems (uint8)
    const float  SNR_THRESH = std::getenv("CW_SNR") ? std::stof(std::getenv("CW_SNR")) : 28.0f;
    const int    STRIDE     = 25;                        // every ~0.5 s (20 ms/slot)

    host_win_.resize((size_t)WSLOTS * slot_elems);
    std::vector<float> col(WROWS);
    std::vector<float> snr(content_bins_, 0.0f);
    std::unordered_map<int, std::string> last_txt;       // crude per-bin dedup
    gm::hf::CwMorse dec(5.0f, 20.0f);
    uint64_t last = 0;

    auto val = [&](int s, int row, int bin) -> uint8_t {
        return host_win_[(size_t)s * slot_elems + (size_t)row * NB + bin];
    };

    while (running_.load(std::memory_order_acquire)) {
        uint64_t wi = ring_.write_idx.load(std::memory_order_acquire);
        if (wi < (uint64_t)WSLOTS + 1 || wi <= last || wi % STRIDE != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        last = wi;

        // Snapshot the last WSLOTS slots [wi-WSLOTS, wi) to host.
        const uint64_t start = wi - WSLOTS;
        cudaStreamWaitEvent(stream_, ring_.ready, 0);
        for (int s = 0; s < WSLOTS; ++s) {
            cudaMemcpyAsync(&host_win_[(size_t)s * slot_elems],
                            ring_.slot(start + s), slot_elems,
                            cudaMemcpyDeviceToHost, stream_);
        }
        cudaStreamSynchronize(stream_);

        // Per-bin detection: peak vs average over the window (keyed CW -> high gap).
        for (int bin = 0; bin < content_bins_; ++bin) {
            int peak = 0; long sum = 0;
            for (int s = 0; s < WSLOTS; ++s)
                for (int r = 0; r < TOSR; ++r) {
                    int v = val(s, r, bin);
                    sum += v; if (v > peak) peak = v;
                }
            snr[bin] = (float)peak - (float)sum / WROWS;
        }

        // Peak-pick: local maximum over ±2 bins (min ~200 Hz separation), SNR gate.
        int emitted = 0;
        for (int bin = 2; bin < content_bins_ - 2 && emitted < 64; ++bin) {
            const float s = snr[bin];
            if (s < SNR_THRESH) continue;
            if (s < snr[bin-1] || s < snr[bin+1] || s < snr[bin-2] || s < snr[bin+2]) continue;

            int idx = 0;
            for (int sl = 0; sl < WSLOTS; ++sl)
                for (int r = 0; r < TOSR; ++r) col[idx++] = (float)val(sl, r, bin);

            std::string txt = dec.decode(col.data(), WROWS);
            int nchar = 0; for (char c : txt) if (c != ' ') ++nchar;
            if (nchar < 4) continue;
            if (last_txt[bin] == txt) continue;   // unchanged since last window: skip
            last_txt[bin] = txt;

            const double hz = binToHz(bin);
            if (hz <= 0.0) continue;
            printf("[%s] %9.3f kHz  ~%2.0f wpm  snr %2.0f  %s\n",
                   label_, hz / 1000.0, dec.wpm(), s, txt.c_str());
            ++emitted;
        }
    }
}

template class CWSkimmerCuda<128>;

} // namespace cuda
} // namespace gm
