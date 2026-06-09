#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <complex>
#include <thread>
#include <zmq.hpp>

#include "gm/rx888/rx888.h"
#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/MagBlock.h"
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/JS8Cuda.h"
#include "gm/cuda/JS8ScanCuda.h"
#include "gm/cuda/JS8FastScanCuda.h"
#include "gm/cuda/WaterfallCuda.h"
#include "gm/hf/ft8.h"
#include "gm/hf/js8.h"
#include "gm/buffer/BufferFile.h"
#include "gm/hf/hf_bands.h"
#include "wsdict_server.h"

static constexpr int kProxyXSubPort = 5599;  // producers connect here
static constexpr int kProxyXPubPort = 5600;  // consumers subscribe here
static constexpr int kWsDictPort    = 8765;  // wsdict WebSocket server

// Normal ring: 65536-pt FFT, 6.25 Hz/bin, 0.16s/block
static constexpr int   kNormalRfftLen  = 65536;
static constexpr int   kNormalTimeOsr  = 4;
static constexpr int   kNormalFreqOsr  = 4;
static constexpr int   kNormalCapBlks  = 108;   // max_block_abs = 29+72+6 = 107
static constexpr float kNormalSymPer   = 0.160f; // seconds per ring block
static constexpr float kNormalCycleSec = 15.0f;

// Fast ring: 40960-pt FFT, 10 Hz/bin, 0.10s/block
static constexpr int   kFastRfftLen   = 40960;
static constexpr int   kFastTimeOsr   = 2;
static constexpr int   kFastFreqOsr   = 2;
static constexpr int   kFastCapBlks   = 108;
static constexpr float kFastSymPer    = 0.100f;
static constexpr float kFastCycleSec  = 10.0f;

// Slow ring: 131072-pt FFT, 3.125 Hz/bin, 0.32s/block  (409600/3.125=131072 exactly)
static constexpr int   kSlowRfftLen   = 131072;
static constexpr int   kSlowTimeOsr   = 2;
static constexpr int   kSlowFreqOsr   = 2;
static constexpr int   kSlowCapBlks   = 108;
static constexpr float kSlowSymPer    = 0.320f;
static constexpr float kSlowCycleSec  = 30.0f;

// Turbo ring: 20480-pt FFT (5×2^12), 20 Hz/bin, 0.05s/block
static constexpr int   kTurboRfftLen   = 20480;
static constexpr int   kTurboTimeOsr   = 2;
static constexpr int   kTurboFreqOsr   = 2;
static constexpr int   kTurboCapBlks   = 108;
static constexpr float kTurboSymPer    = 0.050f;
static constexpr float kTurboCycleSec  = 6.0f;

// Ultra ring: 13107-pt FFT (3×17×257; Bluestein), 31.25 Hz/bin, ~0.032s/block
static constexpr int   kUltraRfftLen   = 13107;
static constexpr int   kUltraTimeOsr   = 2;
static constexpr int   kUltraFreqOsr   = 2;
static constexpr int   kUltraCapBlks   = 108;
static constexpr float kUltraSymPer    = 0.032001f;
static constexpr float kUltraCycleSec  = 4.0f;

// Hz to bin index for the 65,536-bin / 409.6 kHz composite ring.
static int hzToBin(float hz) {
    return (int)(hz / 6.25f + 0.5f);
}

// Runs zmq_proxy(XSUB, XPUB) — blocks forever, call in a detached thread.
static void runProxy() {
    zmq::context_t ctx(1);
    zmq::socket_t xsub(ctx, ZMQ_XSUB);
    xsub.bind("tcp://*:" + std::to_string(kProxyXSubPort));
    zmq::socket_t xpub(ctx, ZMQ_XPUB);
    xpub.bind("tcp://*:" + std::to_string(kProxyXPubPort));
    zmq_proxy(xsub.handle(), xpub.handle(), nullptr);
}

static void runPipeline(gm::buffer::BufferPosition<std::complex<float>>& buf,
                        float min_score, bool enable_js8, bool enable_js8_fast,
                        bool enable_js8_slow, bool enable_js8_turbo, bool enable_js8_ultra,
                        int wf_bin_start, int wf_bin_end,
                        bool legacy_costas, uint8_t wf_floor, uint8_t wf_ceil) {

    // RAII order: MagBlocks own ring memory; all readers hold const refs.
    // C++ destroys in reverse declaration order (readers before rings).
    gm::cuda::MagBlock<200> magblock(&buf,
        kNormalRfftLen, kNormalTimeOsr, kNormalFreqOsr, kProxyXSubPort);
    magblock.start();

    gm::cuda::FT8Cuda ft8channel(magblock.getRing(), min_score, "EPOCH", kProxyXSubPort, legacy_costas);
    ft8channel.start();

    gm::hf::FT8 ft8(&ft8channel, kProxyXSubPort, kWsDictPort);
    ft8.start();

    gm::cuda::WaterfallCuda waterfall(magblock.getRing(),
                                      wf_bin_start, wf_bin_end,
                                      gm::cuda::WaterfallCuda::DEFAULT_OUT_BINS,
                                      kWsDictPort, wf_floor, wf_ceil);
    waterfall.start();

    std::unique_ptr<gm::cuda::JS8Cuda<200>> js8channel;
    std::unique_ptr<gm::hf::JS8>            js8_obj;
    if (enable_js8) {
        js8channel = std::make_unique<gm::cuda::JS8Cuda<200>>(
            magblock.getRing(), min_score, kProxyXSubPort,
            js8_gpu_scan, kNormalTimeOsr, kNormalFreqOsr, kNormalCapBlks, "JS8",
            legacy_costas);
        js8_obj = std::make_unique<gm::hf::JS8>(
            js8channel.get(), kProxyXSubPort,
            kNormalSymPer, kNormalCycleSec, kNormalTimeOsr, kNormalRfftLen,
            "JS8", kWsDictPort);
    }

    std::unique_ptr<gm::cuda::MagBlock<128>>  magblock_fast;
    std::unique_ptr<gm::cuda::JS8Cuda<128>>   js8fast_channel;
    std::unique_ptr<gm::hf::JS8>              js8fast_obj;
    if (enable_js8_fast) {
        magblock_fast = std::make_unique<gm::cuda::MagBlock<128>>(
            &buf, kFastRfftLen, kFastTimeOsr, kFastFreqOsr, 0);
        magblock_fast->start();

        js8fast_channel = std::make_unique<gm::cuda::JS8Cuda<128>>(
            magblock_fast->getRing(), min_score, kProxyXSubPort,
            js8_fast_gpu_scan, kFastTimeOsr, kFastFreqOsr, kFastCapBlks, "JS8 Fast",
            legacy_costas);
        js8fast_obj = std::make_unique<gm::hf::JS8>(
            js8fast_channel.get(), kProxyXSubPort,
            kFastSymPer, kFastCycleSec, kFastTimeOsr, kFastRfftLen,
            "JS8 Fast", kWsDictPort);
    }

    std::unique_ptr<gm::cuda::MagBlock<128>>  magblock_slow;
    std::unique_ptr<gm::cuda::JS8Cuda<128>>   js8slow_channel;
    std::unique_ptr<gm::hf::JS8>              js8slow_obj;
    if (enable_js8_slow) {
        magblock_slow = std::make_unique<gm::cuda::MagBlock<128>>(
            &buf, kSlowRfftLen, kSlowTimeOsr, kSlowFreqOsr, 0);
        magblock_slow->start();

        js8slow_channel = std::make_unique<gm::cuda::JS8Cuda<128>>(
            magblock_slow->getRing(), min_score, kProxyXSubPort,
            js8_fast_gpu_scan, kSlowTimeOsr, kSlowFreqOsr, kSlowCapBlks, "JS8 Slow",
            legacy_costas);
        js8slow_obj = std::make_unique<gm::hf::JS8>(
            js8slow_channel.get(), kProxyXSubPort,
            kSlowSymPer, kSlowCycleSec, kSlowTimeOsr, kSlowRfftLen,
            "JS8 Slow", kWsDictPort);
    }

    std::unique_ptr<gm::cuda::MagBlock<128>>  magblock_turbo;
    std::unique_ptr<gm::cuda::JS8Cuda<128>>   js8turbo_channel;
    std::unique_ptr<gm::hf::JS8>              js8turbo_obj;
    if (enable_js8_turbo) {
        magblock_turbo = std::make_unique<gm::cuda::MagBlock<128>>(
            &buf, kTurboRfftLen, kTurboTimeOsr, kTurboFreqOsr, 0);
        magblock_turbo->start();

        js8turbo_channel = std::make_unique<gm::cuda::JS8Cuda<128>>(
            magblock_turbo->getRing(), min_score, kProxyXSubPort,
            js8_fast_gpu_scan, kTurboTimeOsr, kTurboFreqOsr, kTurboCapBlks, "JS8 Turbo",
            legacy_costas);
        js8turbo_obj = std::make_unique<gm::hf::JS8>(
            js8turbo_channel.get(), kProxyXSubPort,
            kTurboSymPer, kTurboCycleSec, kTurboTimeOsr, kTurboRfftLen,
            "JS8 Turbo", kWsDictPort);
    }

    std::unique_ptr<gm::cuda::MagBlock<128>>  magblock_ultra;
    std::unique_ptr<gm::cuda::JS8Cuda<128>>   js8ultra_channel;
    std::unique_ptr<gm::hf::JS8>              js8ultra_obj;
    if (enable_js8_ultra) {
        magblock_ultra = std::make_unique<gm::cuda::MagBlock<128>>(
            &buf, kUltraRfftLen, kUltraTimeOsr, kUltraFreqOsr, 0);
        magblock_ultra->start();

        js8ultra_channel = std::make_unique<gm::cuda::JS8Cuda<128>>(
            magblock_ultra->getRing(), min_score, kProxyXSubPort,
            js8_fast_gpu_scan, kUltraTimeOsr, kUltraFreqOsr, kUltraCapBlks, "JS8 Ultra",
            legacy_costas);
        js8ultra_obj = std::make_unique<gm::hf::JS8>(
            js8ultra_channel.get(), kProxyXSubPort,
            kUltraSymPer, kUltraCycleSec, kUltraTimeOsr, kUltraRfftLen,
            "JS8 Ultra", kWsDictPort);
    }

    while (true) {
        usleep(1000000);
    }
}

int main(int argc, char* argv[]) {

    bool        enable_js8       = false;
    bool        enable_js8_fast  = false;
    bool        enable_js8_slow  = false;
    bool        enable_js8_turbo = false;
    bool        enable_js8_ultra = false;
    bool        legacy_costas   = true;
    uint8_t     wf_floor        = 170;   // tune: raise to darken noise floor
    uint8_t     wf_ceil         = 210;   // tune: lower to saturate signals sooner
    float       min_score       = -1.0f;  // sentinel: resolved after flag parsing
    // HFChannelizer packs FT8/JS8 sub-band windows into composite 0–211 kHz;
    // bins above that are zero-filled.  Show only the live content region.
    static constexpr float kCompositeContentHz = 211000.0f;  // 2110 bins × 100 Hz
    float       wf_center_hz  = kCompositeContentHz / 2.0f;  // 45000 Hz
    float       wf_bw_hz      = kCompositeContentHz;          // 90000 Hz
    std::string record_file;
    std::string playback_file;
    int         zoom_band_start = -1;
    int         zoom_band_end   = -1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--min-score") == 0 && i + 1 < argc) {
            min_score = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--waterfall-center-hz") == 0 && i + 1 < argc) {
            wf_center_hz = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--waterfall-bw-hz") == 0 && i + 1 < argc) {
            wf_bw_hz = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--record") == 0 && i + 1 < argc) {
            record_file = argv[++i];
        } else if (strcmp(argv[i], "--playback") == 0 && i + 1 < argc) {
            playback_file = argv[++i];
        } else if (strcmp(argv[i], "--js8") == 0) {
            enable_js8 = true;
        } else if (strcmp(argv[i], "--js8-fast") == 0) {
            enable_js8_fast = true;
        } else if (strcmp(argv[i], "--js8-slow") == 0) {
            enable_js8_slow = true;
        } else if (strcmp(argv[i], "--js8-turbo") == 0) {
            enable_js8_turbo = true;
        } else if (strcmp(argv[i], "--js8-ultra") == 0) {
            enable_js8_ultra = true;
        } else if (strcmp(argv[i], "--max-log-costas") == 0) {
            legacy_costas = false;
        } else if (strcmp(argv[i], "--zoom-band") == 0 && i + 2 < argc) {
            zoom_band_start = std::stoi(argv[++i]);
            zoom_band_end   = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--wf-floor") == 0 && i + 1 < argc) {
            wf_floor = (uint8_t)std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--wf-ceil") == 0 && i + 1 < argc) {
            wf_ceil  = (uint8_t)std::stoi(argv[++i]);
        }
    }

    if (min_score < 0.0f)
        min_score = legacy_costas ? 5.0f : 3.0f;

    printf("Costas metric: %s  min_score=%.1f\n",
           legacy_costas ? "legacy (freq+temporal neighbor)" : "max-log 8-FSK",
           (double)min_score);

    int wf_bin_start = hzToBin(wf_center_hz - wf_bw_hz / 2.0f);
    int wf_bin_end   = hzToBin(wf_center_hz + wf_bw_hz / 2.0f);
    wf_bin_start     = std::max(0, wf_bin_start);
    wf_bin_end       = std::min(wf_bin_end, (int)65536);
    if (wf_bin_start >= wf_bin_end) {
        fprintf(stderr, "[WF] waterfall range invalid (center=%.0f bw=%.0f); showing full content band\n",
                wf_center_hz, wf_bw_hz);
        wf_bin_start = 0;
        wf_bin_end   = hzToBin(kCompositeContentHz);
    }

    if (zoom_band_start >= 0) {
        int cum_bins[kNumHFBands + 1];
        cum_bins[0] = 0;
        for (int b = 0; b < kNumHFBands; ++b)
            cum_bins[b + 1] = cum_bins[b] + kHFBands[b].bw * 16;

        zoom_band_start = std::max(0, std::min(zoom_band_start, kNumHFBands - 1));
        zoom_band_end   = std::max(zoom_band_start, std::min(zoom_band_end, kNumHFBands));
        wf_bin_start = cum_bins[zoom_band_start];
        wf_bin_end   = cum_bins[zoom_band_end];
        printf("Waterfall: --zoom-band %d-%d  (%s–%s)  bins [%d, %d)\n",
               zoom_band_start, zoom_band_end,
               kHFBands[zoom_band_start].name,
               kHFBands[std::min(zoom_band_end, kNumHFBands - 1)].name,
               wf_bin_start, wf_bin_end);
    } else {
        printf("Waterfall: center=%.0f Hz  bw=%.0f Hz  bins [%d, %d)\n",
               wf_center_hz, wf_bw_hz, wf_bin_start, wf_bin_end);
    }

    std::thread(runProxy).detach();
    printf("ZMQ proxy: XSUB tcp://*:%d  XPUB tcp://*:%d\n",
           kProxyXSubPort, kProxyXPubPort);

    if (wsdict_server_start(kWsDictPort, "control", nullptr) != 0) {
        fprintf(stderr, "Failed to start wsdict server on port %d\n", kWsDictPort);
        return 1;
    }
    printf("wsdict server: http://localhost:%d  (WebSocket at /ws)\n", kWsDictPort);

    if (!playback_file.empty()) {
        gm::buffer::BufferFile<std::complex<float>> playback(playback_file);
        playback.start();
        runPipeline(*playback.getBuffer(), min_score, enable_js8, enable_js8_fast,
                    enable_js8_slow, enable_js8_turbo, enable_js8_ultra,
                    wf_bin_start, wf_bin_end, legacy_costas, wf_floor, wf_ceil);
    } else {
        gm::rx888::rx888 mydsp;
        mydsp.start_card();

        gm::cuda::HFChannelizer channelizer(mydsp.getRxBufferPosition(), kWsDictPort);
        channelizer.start();

        std::unique_ptr<gm::buffer::BufferFile<std::complex<float>>> recorder;
        if (!record_file.empty()) {
            recorder = std::make_unique<gm::buffer::BufferFile<std::complex<float>>>(
                channelizer.getBuffer(), record_file, channelizer.getBufferFileParams());
            recorder->start();
        }

        runPipeline(*channelizer.getBuffer(), min_score, enable_js8, enable_js8_fast,
                    enable_js8_slow, enable_js8_turbo, enable_js8_ultra,
                    wf_bin_start, wf_bin_end, legacy_costas, wf_floor, wf_ceil);
    }

    return 0;
}
