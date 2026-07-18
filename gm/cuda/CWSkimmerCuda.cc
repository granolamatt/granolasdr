#include "gm/cuda/CWSkimmerCuda.h"
#include "gm/hf/cw_bands.h"
#include "gm/hf/cw_morse.h"
#include "gm/hf/cw_track.h"
#include "gm/hf/decode_stats.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

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
    const int    WSLOTS     = std::min(250, N - 2);      // window depth (~5 s @ 20ms/slot)
    const int    WROWS      = WSLOTS * TOSR;             // envelope length
    const size_t slot_elems = ring_.slot_bytes;         // bytes == elems (uint8)
    // strtof (not stof) so a non-numeric CW_SNR falls back to the default instead
    // of throwing std::invalid_argument -> uncaught -> std::terminate.
    const float  SNR_THRESH = [] {
        const char* e = std::getenv("CW_SNR");
        char* end = nullptr;
        float v = e ? std::strtof(e, &end) : 0.0f;
        return (e && end != e) ? v : 6.0f;      // absolute floor (adaptive gate rules above it)
    }();
    const int    STRIDE     = 25;                        // every ~0.5 s (20 ms/slot)

    host_win_.resize((size_t)WSLOTS * slot_elems);
    std::vector<float> col(WROWS);
    std::vector<float> dbuf_;                     // detrend scratch for decodeActive
    std::vector<float> snr(content_bins_, 0.0f);
    gm::hf::CwMorse dec(5.0f, 20.0f);
    uint64_t last = 0;

    // Per-signal tracker: keeps one Track per carrier across windows, following
    // drift, suppressing sidebands, and confirming a callsign only when the SAME
    // carrier decodes it in >=confirm windows.  This replaces the old global
    // callsign-keyed Heard map, which cross-confirmed one-off misdecodes from
    // unrelated bins on a crowded band (cwtest2.dat).
    gm::hf::CwTracker::Config tcfg;
    tcfg.snr_thresh = SNR_THRESH;
    if (const char* d = std::getenv("CW_DRIFT")) tcfg.drift_bins = tcfg.merge_bins = std::max(0, std::atoi(d));
    if (const char* c = std::getenv("CW_CONFIRM")) tcfg.confirm = std::max(1, std::atoi(c));
    tcfg.adapt_k = 2.7f;   // adaptive gate on by default (median-relative); CW_ADAPT_K=0 for fixed
    if (const char* k = std::getenv("CW_ADAPT_K")) tcfg.adapt_k = std::max(0.0f, std::strtof(k, nullptr));
    gm::hf::CwTracker tracker(tcfg);

    auto val = [&](int s, int row, int bin) -> uint8_t {
        return host_win_[(size_t)s * slot_elems + (size_t)row * NB + bin];
    };

    // Extract a bin's magnitude envelope, trim the leading/trailing silence to the
    // active span (above the mid level), and decode.  Trimming is essential: a long
    // key-up region ahead of the transmission lets cw_morse's AGC normalize noise up
    // to signal level and merge the whole window into one mark.
    auto decodeActive = [&](int bin) -> std::string {
        int idx = 0;
        for (int sl = 0; sl < WSLOTS; ++sl)
            for (int r = 0; r < TOSR; ++r) col[idx++] = (float)val(sl, r, bin);
        float cmin = col[0], cmax = col[0];
        for (int i = 1; i < WROWS; ++i) {
            if (col[i] < cmin) cmin = col[i];
            if (col[i] > cmax) cmax = col[i];
        }
        const float act = cmin + 0.5f * (cmax - cmin);
        int a = 0, b = WROWS - 1;
        while (a < WROWS && col[a] < act) ++a;
        while (b > a && col[b] < act) --b;
        const int PAD = 4;                       // ~20 ms so the edge element isn't clipped
        a = (a > PAD) ? a - PAD : 0;
        b = (b + PAD < WROWS) ? b + PAD : WROWS - 1;
        const int wlen = b - a + 1;
        if (wlen < 8) return "";                 // too short to be a callsign
        // Detrend: subtract the noise floor (median of the span) so key-up gaps go to
        // ~0.  Real CW mag is log-compressed uint8 (noise ~0.7*peak), so without this
        // the decoder's AGC amplifies the char/word gaps up to signal level and merges
        // the transmission into one mark.
        dbuf_.assign(col.begin() + a, col.begin() + a + wlen);
        std::vector<float> tmp(dbuf_);
        std::nth_element(tmp.begin(), tmp.begin() + wlen / 2, tmp.end());
        const float floor = tmp[wlen / 2];
        bool any = false;
        for (float& v : dbuf_) { v -= floor; if (v < 0.0f) v = 0.0f; else if (v > 0.0f) any = true; }
        if (!any) return "";                     // detrended to silence (median==peak): skip
        return dec.decode(dbuf_.data(), wlen);
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
        { std::lock_guard<std::mutex> lk(ring_.ready_mu);
          cudaStreamWaitEvent(stream_, ring_.ready, 0); }
        for (int s = 0; s < WSLOTS; ++s) {
            cudaMemcpyAsync(&host_win_[(size_t)s * slot_elems],
                            ring_.slot(start + s), slot_elems,
                            cudaMemcpyDeviceToHost, stream_);
        }
        cudaStreamSynchronize(stream_);

        // Per-bin detection: (80th percentile - median) of the bin's magnitude
        // distribution over the window.  A keyed CW carrier is bimodal — key-down
        // sits well above the key-up noise floor — so its high percentile beats its
        // median.  Steady noise (unimodal) and continuous carriers/birdies (high but
        // flat) both give ~0.  Robust to single spikes, unlike peak-mean.
        int gmin = 255, gmax = 0; double gsum = 0;
        unsigned int hist[256];
        const int med_t = WROWS / 2, p80_t = (WROWS * 4) / 5;
        for (int bin = 0; bin < content_bins_; ++bin) {
            for (int h = 0; h < 256; ++h) hist[h] = 0;
            for (int s = 0; s < WSLOTS; ++s)
                for (int r = 0; r < TOSR; ++r) {
                    int v = val(s, r, bin);
                    hist[v]++;
                    if (v < gmin) gmin = v; if (v > gmax) gmax = v; gsum += v;
                }
            int cum = 0, p50 = 0, p80 = 0; bool g50 = false;
            for (int h = 0; h < 256; ++h) {
                cum += hist[h];
                if (!g50 && cum >= med_t) { p50 = h; g50 = true; }
                if (cum >= p80_t)         { p80 = h; break; }
            }
            snr[bin] = (float)(p80 - p50);
        }

        // CW_DEBUG: report the ring magnitude range + the strongest bins, so a
        // dead/clamped scale or an absent signal is visible without a decode.
        if (std::getenv("CW_DEBUG")) {
            int passed = 0;
            for (int b = 0; b < content_bins_; ++b) if (snr[b] >= SNR_THRESH) ++passed;
            // top bin by SNR
            int top = 0; for (int b = 1; b < content_bins_; ++b) if (snr[b] > snr[top]) top = b;
            fprintf(stderr,
                "[CWdbg] wi=%llu mag[min/mean/max]=%d/%.0f/%d  bins=%d thr=%.0f passed=%d  "
                "top bin=%d (%.3f kHz) snr=%.0f\n",
                (unsigned long long)wi, gmin, gsum / ((double)content_bins_ * WROWS), gmax,
                content_bins_, SNR_THRESH, passed, top, binToHz(top) / 1000.0, snr[top]);
            // coarse envelope of the top bin (50 cols, '#'=above mid, '.'=below)
            int emin = 255, emax = 0;
            for (int sl = 0; sl < WSLOTS; ++sl) for (int r = 0; r < TOSR; ++r) {
                int v = val(sl, r, top); if (v < emin) emin = v; if (v > emax) emax = v;
            }
            int mid = (emin + emax) / 2; char line[64]; int li = 0;
            const int step = WROWS / 50 ? WROWS / 50 : 1;
            for (int i = 0; i < WROWS && li < 50; i += step) {
                int sl = i / TOSR, r = i % TOSR;
                line[li++] = (val(sl, r, top) > mid) ? '#' : '.';
            }
            line[li] = 0;
            fprintf(stderr, "[CWdbg]   top env [%d..%d]: %s\n", emin, emax, line);
            // Unconditional decode of the strongest bin (bypasses the gate + filters)
            // so we can see what cw_morse actually returns even when nothing emits.
            std::string dtxt = decodeActive(top);
            fprintf(stderr, "[CWdbg]   top decode (~%.0f wpm): \"%s\"\n",
                    dec.wpm(), dtxt.c_str());
        }

        // Advance the per-signal tracker over this window's detection metric.  It
        // associates peaks to carriers, follows drift, and returns only the calls
        // freshly confirmed (same carrier, >=confirm windows).  decodeActive sets
        // dec's speed as a side effect, so wpm() is read right after decode(bin).
        auto spots = tracker.step(
            wi, snr.data(), content_bins_,
            [&](int bin){ return decodeActive(bin); },
            [&](int)    { return dec.wpm(); },
            [&](int bin){ return binToHz(bin); });
        for (const auto& sp : spots) {
            printf("[%s] %9.3f kHz  ~%2.0f wpm  snr %2.0f  %s%s\n",
                   label_, sp.hz / 1000.0, (double)sp.wpm, (double)sp.snr, sp.call.c_str(),
                   sp.cq ? "  CQ" : "");
            gm::hf::decode_stats_count(label_);
        }
    }
}

template class CWSkimmerCuda<256>;

} // namespace cuda
} // namespace gm
