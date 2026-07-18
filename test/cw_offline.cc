// cw_offline.cc — offline CW skimmer harness.
//
// Reads a --record-cw GNLH capture (819.2 kHz complex CW composite) and runs the
// CW MagBlock front end on the CPU (16384-pt FFT, time_osr=4 -> 5 ms hop, no
// window, the EXACT magKernel uint8 scaling), then feeds the SAME per-bin
// detection metric (p80-p50) + CwTracker + CwMorse that the live CWSkimmerCuda
// uses (gm/cuda/CWSkimmerCuda.cc). Replays any (start,dur) window so the detector
// can be tuned against a real capture with no CUDA and no radio.
//
// Usage: cw_offline <capture.dat> [start_sec] [dur_sec]
//   env CW_SNR / CW_DRIFT / CW_CONFIRM — same knobs as hf_rx.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "ft8_lib/fft/kiss_fft.h"
}
#include "gm/hf/cw_bands.h"     // kCWBands / kNumCWBands (global namespace)
#include "gm/hf/cw_morse.h"
#include "gm/hf/cw_track.h"

// --- geometry, matched to the CW MagBlock<256> + CWSkimmerCuda worker ----------
static constexpr int    RFFT   = 16384;         // CW MagBlock FFT
static constexpr int    TOSR   = 4;             // time_osr: rows per slot
static constexpr int    HOP    = RFFT / TOSR;   // 4096 samples = 5 ms @ 819.2 kHz
static constexpr int    NB     = RFFT;          // bins per row
static constexpr int    WSLOTS = 250;           // detection window depth (~5 s)
static constexpr int    WROWS  = WSLOTS * TOSR; // 1000
static constexpr int    STRIDE = 25;            // run detection every ~0.5 s
static constexpr double SR     = 819200.0;

// GNLH header: <IIIIQII> magic,ver,block_samples,element_bytes,interval_ns,sr,rx_sr
static constexpr uint32_t GNLH_MAGIC = 0x474E4C48;
#pragma pack(push, 1)
struct GnlhHeader { uint32_t magic, ver, block_samples, element_bytes;
                    uint64_t interval_ns; uint32_t sr, rx_sr; };
#pragma pack(pop)

// magKernel (gm/cuda/HostCuda.cu:83-86), verbatim.
static inline uint8_t mag_u8(const std::complex<float>& z) {
    float mag = std::abs(z) / 400e6f;
    float db  = 10.0f * std::log10(1e-12f + mag);
    int scaled = (int)(2 * db + 240);
    return (uint8_t)(scaled < 0 ? 0 : (scaled > 255 ? 255 : scaled));
}

// binToHz (CWSkimmerCuda::binToHz), verbatim.
static double binToHz(int magbin) {
    const int c = magbin / 2;
    int cum = 0;
    for (int b = 0; b < kNumCWBands; ++b) {
        if (c < cum + (int)kCWBands[b].bw)
            return (double)(kCWBands[b].wb_start + (c - cum)) * 100.0;
        cum += (int)kCWBands[b].bw;
    }
    return 0.0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <capture.dat> [start_sec] [dur_sec]\n", argv[0]); return 2; }
    const char* path = argv[1];
    const double start_sec = argc > 2 ? atof(argv[2]) : 0.0;
    const double dur_sec   = argc > 3 ? atof(argv[3]) : 60.0;

    uint32_t total_bw = 0;
    for (int i = 0; i < kNumCWBands; ++i) total_bw += kCWBands[i].bw;
    const int content_bins = std::min(2 * (int)total_bw, NB);   // 9400

    // --- read the requested window ---------------------------------------------
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    GnlhHeader h;
    if (fread(&h, sizeof h, 1, f) != 1 || h.magic != GNLH_MAGIC || h.element_bytes != 8) {
        fprintf(stderr, "bad GNLH header (magic=%#x eb=%u)\n", h.magic, h.element_bytes); return 1; }
    if (h.sr != (uint32_t)SR)
        fprintf(stderr, "WARNING: sr=%u, expected %.0f (CW composite)\n", h.sr, SR);

    const size_t a = (size_t)(start_sec * h.sr);
    const size_t nsamp = (size_t)(dur_sec * h.sr);
    std::vector<std::complex<float>> x(nsamp);
    if (fseek(f, (long)(sizeof(GnlhHeader) + a * 8), SEEK_SET) != 0) { fprintf(stderr, "seek failed\n"); return 1; }
    size_t got = fread(x.data(), 8, nsamp, f);
    fclose(f);
    x.resize(got);
    const int nslots = (int)((got >= (size_t)((TOSR - 1) * HOP + RFFT))
                             ? (got - ((TOSR - 1) * HOP + RFFT)) / RFFT : 0);
    printf("%s: %.0f..%.0fs  %d slots (%.1fs), %d content bins @ %.0f Hz\n",
           path, start_sec, start_sec + dur_sec, nslots, nslots * RFFT / SR,
           content_bins, SR / RFFT);
    if (nslots < WSLOTS + STRIDE) { fprintf(stderr, "window too short for detection\n"); return 1; }

    // --- CPU MagBlock front end -> uint8 mags[nslots][TOSR][content_bins] -------
    std::vector<uint8_t> mags((size_t)nslots * TOSR * content_bins);
    kiss_fft_cfg cfg = kiss_fft_alloc(RFFT, 0, nullptr, nullptr);
    std::vector<kiss_fft_cpx> in(RFFT), out(RFFT);
    for (int s = 0; s < nslots; ++s) {
        for (int r = 0; r < TOSR; ++r) {
            const size_t base = (size_t)s * RFFT + (size_t)r * HOP;
            for (int i = 0; i < RFFT; ++i) { in[i].r = x[base + i].real(); in[i].i = x[base + i].imag(); }
            kiss_fft(cfg, in.data(), out.data());
            uint8_t* row = &mags[(((size_t)s * TOSR + r) * content_bins)];
            for (int bin = 0; bin < content_bins; ++bin)
                row[bin] = mag_u8({out[bin].r, out[bin].i});
        }
    }
    kiss_fft_free(cfg);

    // --- detection + tracker + decode, mirroring CWSkimmerCuda::worker ----------
    auto env_f = [](const char* n, float d) { const char* e = std::getenv(n); char* p = nullptr;
        float v = e ? std::strtof(e, &p) : 0; return (e && p != e) ? v : d; };
    const float SNR_THRESH = env_f("CW_SNR", 6.0f);   // floor; adaptive gate rules above it
    gm::hf::CwTracker::Config tcfg; tcfg.snr_thresh = SNR_THRESH;
    tcfg.adapt_k = 2.7f;                               // adaptive gate default (matches CWSkimmerCuda)
    if (const char* d = std::getenv("CW_DRIFT"))   tcfg.drift_bins = tcfg.merge_bins = std::max(0, atoi(d));
    if (const char* c = std::getenv("CW_CONFIRM")) tcfg.confirm = std::max(1, atoi(c));
    if (const char* k = std::getenv("CW_ADAPT_K")) tcfg.adapt_k = std::max(0.0f, std::strtof(k, nullptr));
    gm::hf::CwTracker tracker(tcfg);
    gm::hf::CwMorse dec(5.0f, 20.0f);

    std::vector<float> snr(content_bins, 0.0f);
    std::vector<float> col(WROWS), dbuf;
    int winStart = 0;
    auto valw = [&](int ws, int r, int bin) -> uint8_t {
        return mags[(((size_t)(winStart + ws) * TOSR + r) * content_bins + bin)];
    };
    // decodeActive(bin): CWSkimmerCuda.cc:93-129, verbatim logic.
    auto decodeActive = [&](int bin) -> std::string {
        int idx = 0;
        for (int sl = 0; sl < WSLOTS; ++sl)
            for (int r = 0; r < TOSR; ++r) col[idx++] = (float)valw(sl, r, bin);
        float cmin = col[0], cmax = col[0];
        for (int i = 1; i < WROWS; ++i) { if (col[i] < cmin) cmin = col[i]; if (col[i] > cmax) cmax = col[i]; }
        const float act = cmin + 0.5f * (cmax - cmin);
        int a2 = 0, b2 = WROWS - 1;
        while (a2 < WROWS && col[a2] < act) ++a2;
        while (b2 > a2 && col[b2] < act) --b2;
        const int PAD = 4;
        a2 = (a2 > PAD) ? a2 - PAD : 0;
        b2 = (b2 + PAD < WROWS) ? b2 + PAD : WROWS - 1;
        const int wlen = b2 - a2 + 1;
        if (wlen < 8) return "";
        dbuf.assign(col.begin() + a2, col.begin() + a2 + wlen);
        std::vector<float> tmp(dbuf);
        std::nth_element(tmp.begin(), tmp.begin() + wlen / 2, tmp.end());
        const float floor = tmp[wlen / 2];
        bool any = false;
        for (float& v : dbuf) { v -= floor; if (v < 0.0f) v = 0.0f; else if (v > 0.0f) any = true; }
        if (!any) return "";
        return dec.decode(dbuf.data(), wlen);
    };

    std::vector<std::string> all_calls;
    int windows = 0;
    const int med_t = WROWS / 2, p80_t = (WROWS * 4) / 5;
    for (int s = WSLOTS; s < nslots; ++s) {
        if (s % STRIDE != 0) continue;
        winStart = s - WSLOTS;
        ++windows;
        unsigned hist[256];
        for (int bin = 0; bin < content_bins; ++bin) {
            for (int hh = 0; hh < 256; ++hh) hist[hh] = 0;
            for (int ws = 0; ws < WSLOTS; ++ws)
                for (int r = 0; r < TOSR; ++r) hist[valw(ws, r, bin)]++;
            int cum = 0, p50 = 0, p80 = 0; bool g50 = false;
            for (int hh = 0; hh < 256; ++hh) {
                cum += hist[hh];
                if (!g50 && cum >= med_t) { p50 = hh; g50 = true; }
                if (cum >= p80_t) { p80 = hh; break; }
            }
            snr[bin] = (float)(p80 - p50);
        }
        auto spots = tracker.step((uint64_t)s, snr.data(), content_bins,
            [&](int bin) { return decodeActive(bin); },
            [&](int)     { return dec.wpm(); },
            [&](int bin) { return binToHz(bin); });
        const double t = start_sec + (double)s * RFFT / SR;
        for (const auto& sp : spots) {
            printf("[CW] t=%6.0fs  %9.3f kHz  ~%2.0f wpm  snr %2.0f  %s%s\n",
                   t, sp.hz / 1000.0, sp.wpm, sp.snr, sp.call.c_str(), sp.cq ? "  CQ" : "");
            all_calls.push_back(sp.call);
        }
    }

    std::sort(all_calls.begin(), all_calls.end());
    int uniq = (int)(std::unique(all_calls.begin(), all_calls.end()) - all_calls.begin());
    printf("\n%d detection windows, gate floor=%.0f adapt_k=%.1f -> %d spots, %d unique calls\n",
           windows, SNR_THRESH, tcfg.adapt_k, (int)all_calls.size(), uniq);
    return 0;
}
