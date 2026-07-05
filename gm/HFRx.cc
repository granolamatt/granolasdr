#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <complex>
#include <thread>
#include <chrono>
#include <zmq.hpp>

#include "gm/rx888/rx888.h"
#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/MagBlock.h"
#include "gm/cuda/CWSkimmerCuda.h"
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/JS8Cuda.h"
#include "gm/cuda/JS8ScanCuda.h"
#include "gm/cuda/JS8FastScanCuda.h"
#include "gm/cuda/WaterfallCuda.h"
#include "gm/hf/cw_bands.h"
#include "gm/hf/ft8.h"
#include "gm/hf/js8.h"
#include "gm/buffer/BufferFile.h"
#include "gm/hf/hf_bands.h"
#include "wsdict_server.h"
#include "tci_server.h"

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
                        bool legacy_costas,
                        gm::buffer::BufferFile<std::complex<float>>* playback = nullptr,
                        gm::buffer::BufferPosition<std::complex<float>>* cwbuf = nullptr) {

    // RAII order: MagBlocks own ring memory; all readers hold const refs.
    // C++ destroys in reverse declaration order (readers before rings).
    gm::cuda::MagBlock<200> magblock(&buf,
        kNormalRfftLen, kNormalTimeOsr, kNormalFreqOsr, kProxyXSubPort,
        /*retain_complex=*/true);   // FT8/JS8 Normal refine reads the complex ring
    magblock.start();

    gm::cuda::FT8Cuda ft8channel(magblock.getRing(), min_score, "EPOCH", kProxyXSubPort, legacy_costas,
                                 &magblock.getComplexRing(), magblock.getSlotCplxIdx());
    ft8channel.start();

    gm::hf::FT8 ft8(&ft8channel, kProxyXSubPort, kWsDictPort);
    ft8.start();

    gm::cuda::WaterfallCuda<200> waterfall(magblock.getRing(),
                                      wf_bin_start, wf_bin_end,
                                      gm::cuda::WaterfallCuda<200>::DEFAULT_OUT_BINS,
                                      kWsDictPort);
    waterfall.start();

    // OSD LDPC fallback, ON by default (score_floor 6).  FT8 and JS8 are
    // configured independently.  Tunables (optional, per mode):
    //   <MODE>_OSD                "0" disables; unset/any other value enables (default ON)
    //   <MODE>_OSD_SCORE_FLOOR    min Costas sync score to attempt OSD (default 6)
    //   <MODE>_OSD_MAX            max OSD attempts per scan cycle       (default 64)
    // where <MODE> is FT8 or JS8.  See gm/hf/{ft8,js8}.cc:setOsdConfig.  Gating
    // bounds the OSD-on-noise false-accept rate (~0.045%/attempt); CRC is final.
    struct OsdEnv { bool enable; float floor; int max; };
    auto read_osd_env = [](const char* en, const char* fl, const char* mx) -> OsdEnv {
        const char* e = std::getenv(en);
        return { (e == nullptr) || (std::string(e) != "0"),  // default ON; <MODE>_OSD=0 disables
                 std::getenv(fl) ? std::stof(std::getenv(fl)) : 6.0f,
                 std::getenv(mx) ? std::stoi(std::getenv(mx)) : 64 };
    };
    auto log_osd = [](const char* mode, const OsdEnv& o) {
        if (o.enable)
            printf("%s OSD fallback: ENABLED (order=2, score_floor=%.1f, max_per_cycle=%d)\n",
                   mode, (double)o.floor, o.max);
        else
            printf("%s OSD fallback: disabled (%s_OSD=0)\n", mode, mode);
    };

    const OsdEnv ft8_osd = read_osd_env("FT8_OSD", "FT8_OSD_SCORE_FLOOR", "FT8_OSD_MAX");
    ft8.setOsdConfig(ft8_osd.enable, /*order=*/2, ft8_osd.floor, ft8_osd.max);
    log_osd("FT8", ft8_osd);

    // Per-candidate freq/time refine fallback (reads MagBlock's complex ring).
    // ON by default; set FT8_REFINE=0 to disable. Runs only after BP+OSD both fail.
    const char* ft8_refine_env = std::getenv("FT8_REFINE");
    const bool ft8_refine = (ft8_refine_env == nullptr) ||
                            (std::string(ft8_refine_env) != "0");
    ft8.setRefineEnabled(ft8_refine);
    printf("FT8 refine fallback: %s\n", ft8_refine ? "ENABLED"
                                                    : "disabled (FT8_REFINE=0)");

    const OsdEnv js8_osd = read_osd_env("JS8_OSD", "JS8_OSD_SCORE_FLOOR", "JS8_OSD_MAX");
    auto apply_osd = [&](gm::hf::JS8* j) {
        if (j) j->setOsdConfig(js8_osd.enable, /*order=*/2, js8_osd.floor, js8_osd.max);
    };
    log_osd("JS8", js8_osd);

    // JS8 Normal shares MagBlock<200> with FT8, so it refines from the same
    // complex ring.  ON by default; set JS8_REFINE=0 to disable (Normal only).
    const char* js8_refine_env = std::getenv("JS8_REFINE");
    const bool js8_refine = (js8_refine_env == nullptr) ||
                            (std::string(js8_refine_env) != "0");

    std::unique_ptr<gm::cuda::JS8Cuda<200>> js8channel;
    std::unique_ptr<gm::hf::JS8>            js8_obj;
    if (enable_js8) {
        js8channel = std::make_unique<gm::cuda::JS8Cuda<200>>(
            magblock.getRing(), min_score, kProxyXSubPort,
            js8_gpu_scan, kNormalTimeOsr, kNormalFreqOsr, kNormalCapBlks, "JS8",
            legacy_costas, &magblock.getComplexRing(), magblock.getSlotCplxIdx());
        js8_obj = std::make_unique<gm::hf::JS8>(
            js8channel.get(), kProxyXSubPort,
            kNormalSymPer, kNormalCycleSec, kNormalTimeOsr, kNormalRfftLen,
            "JS8", kWsDictPort);
        js8_obj->setRefineEnabled(js8_refine);
        printf("JS8 refine fallback: %s\n", js8_refine ? "ENABLED"
                                                       : "disabled (JS8_REFINE=0)");
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

    apply_osd(js8_obj.get());
    apply_osd(js8fast_obj.get());
    apply_osd(js8slow_obj.get());
    apply_osd(js8turbo_obj.get());
    apply_osd(js8ultra_obj.get());

    // CW skimmer (--cw): the CW composite (819.2 kHz, 9 CW sub-bands) feeds a
    // dedicated CW MagBlock at rfft=16384/osr4 -> 50 Hz/bin, 20 ms window, 5 ms
    // hop.  Phase -1: build the ring so CW signals are visible; CWSkimmerCuda +
    // gm::hf::CW decode land in Phase 1+.
    std::unique_ptr<gm::cuda::MagBlock<256>>       magblock_cw;
    std::unique_ptr<gm::cuda::CWSkimmerCuda<256>>  cw_skimmer;
    std::unique_ptr<gm::cuda::WaterfallCuda<256>>  cw_waterfall;
    if (cwbuf) {
        magblock_cw = std::make_unique<gm::cuda::MagBlock<256>>(
            cwbuf, /*rfft=*/16384, /*time_osr=*/4, /*freq_osr=*/1, /*zmq=*/0);
        magblock_cw->start();
        cw_skimmer = std::make_unique<gm::cuda::CWSkimmerCuda<256>>(
            magblock_cw->getRing(), "CW");
        cw_skimmer->start();

        // CW waterfall: the 9 packed CW sub-bands occupy the lower 2·Σbw MagBlock
        // bins (50 Hz/bin, 2 per 100 Hz composite bin).  Published on its own wsdict
        // key family so the dashboard can select CW mode.  Floor/ceil tuned for the
        // un-normalized CW composite (noise ≈168, carriers ≈196), not the FT8 scale.
        uint32_t cw_total = 0;
        for (int i = 0; i < kNumCWBands; ++i) cw_total += kCWBands[i].bw;
        const int cw_content_bins = (int)(2 * cw_total);   // ~9400
        cw_waterfall = std::make_unique<gm::cuda::WaterfallCuda<256>>(
            magblock_cw->getRing(), /*bin_start=*/0, /*bin_end=*/cw_content_bins,
            gm::cuda::WaterfallCuda<256>::DEFAULT_OUT_BINS, kWsDictPort,
            "granolasdr:cwwaterfall:", /*full_rate_hz=*/819200.0f,
            /*min_publish_ms=*/120);
        cw_waterfall->start();

        printf("CW skimmer: ring up (16384-pt FFT @ 819.2 kHz -> 50 Hz/bin, 20 ms window); "
               "decode active (CW_SNR to tune threshold); waterfall on cwwaterfall:*\n");
    }

    while (true) {
        // --playback-once: when the file is fully fed, let the pipeline drain the
        // trailing frames, flush captures/stdout, and exit cleanly.  (Blocks stay
        // parked on getPosition once the feed stops, so we exit rather than join.)
        if (playback && playback->finished()) {
            const int kDrainSec = 3;
            printf("Playback finished; draining %d s then exiting.\n", kDrainSec);
            fflush(stdout);
            std::this_thread::sleep_for(std::chrono::seconds(kDrainSec));
            fflush(stdout);
            fflush(stderr);
            std::exit(0);
        }
        usleep(200000);
    }
}

// CW-only playback: replay a --record-cw capture (819.2 kHz CW composite) through
// the CW MagBlock -> CWSkimmerCuda, with no FT8/JS8 chain.  This closes the offline
// iteration loop for the CW detector/decoder: capture live with --record-cw, then
// replay here under CW_SNR / CW_DEBUG to tune without the radio.  The capture loops
// at EOF (BufferFile default), so this runs until interrupted.
static void runCWPlayback(gm::buffer::BufferPosition<std::complex<float>>& cwbuf) {
    gm::cuda::MagBlock<256> magblock_cw(&cwbuf,
        /*rfft=*/16384, /*time_osr=*/4, /*freq_osr=*/1, /*zmq=*/0);
    magblock_cw.start();
    gm::cuda::CWSkimmerCuda<256> cw_skimmer(magblock_cw.getRing(), "CW");
    cw_skimmer.start();

    // CW waterfall over the replayed composite, so the dashboard (CW mode) confirms
    // the capture actually holds signals.
    uint32_t cw_total = 0;
    for (int i = 0; i < kNumCWBands; ++i) cw_total += kCWBands[i].bw;
    gm::cuda::WaterfallCuda<256> cw_waterfall(
        magblock_cw.getRing(), /*bin_start=*/0, /*bin_end=*/(int)(2 * cw_total),
        gm::cuda::WaterfallCuda<256>::DEFAULT_OUT_BINS, kWsDictPort,
        "granolasdr:cwwaterfall:", /*full_rate_hz=*/819200.0f,
        /*min_publish_ms=*/120);
    cw_waterfall.start();

    printf("CW playback: ring up (16384-pt FFT @ 819.2 kHz -> 50 Hz/bin, 20 ms window); "
           "decode active (CW_SNR / CW_DEBUG to tune); waterfall on cwwaterfall:*\n");
    while (true) {
        usleep(1000000);
    }
}

static void printUsage(const char* prog) {
    fprintf(stderr,
"Usage: %s [options]\n"
"\n"
"Input (default: live RX888):\n"
"  --playback <file>          Replay a --record capture, looping at EOF\n"
"  --playback-once <file>     Replay a capture once, then exit cleanly\n"
"  --playback-cw <file>       Replay a --record-cw capture into the CW skimmer only\n"
"  --record <file>            Record the FT8/JS8 composite (complex float) to <file>\n"
"  --record-cw <file>         Record the CW composite (819.2 kHz) to <file> (implies --cw)\n"
"\n"
"Decoders (FT8 always on; JS8 Normal/Fast/Slow + CW on by default):\n"
"  --no-js8                   Disable JS8 Normal\n"
"  --no-js8-fast              Disable JS8 Fast\n"
"  --no-js8-slow              Disable JS8 Slow\n"
"  --no-cw                    Disable the CW skimmer\n"
"  --js8-turbo                Enable JS8 Turbo (opt-in)\n"
"  --js8-ultra                Enable JS8 Ultra (opt-in)\n"
"\n"
"Decode tuning:\n"
"  --min-score <f>            Costas sync threshold (default 5.0 legacy / 3.0 max-log)\n"
"  --max-log-costas           Use the max-log 8-FSK Costas metric (default: legacy)\n"
"  env FT8_REFINE=0/JS8_REFINE=0     disable per-candidate refine (on by default)\n"
"\n"
"Waterfall:\n"
"  --waterfall-center-hz <f>  Composite center to display (default 45000)\n"
"  --waterfall-bw-hz <f>      Composite bandwidth to display (default 211000)\n"
"  --zoom-band <start> <end>  Zoom the waterfall to an HF band index range\n"
"  env WF_GAIN=<f>            Waterfall contrast (default 6)\n"
"  env WF_OFFSET=<n>          Waterfall noise-floor lift, may be negative (default 0)\n"
"\n"
"TCI (WSJT-X / JTDX / N1MM audio):\n"
"  --tci-port <n>             TCI server port, 0 = disabled (default 40001)\n"
"  --tci-gain <f>             TCI INT16 gain (default 1.0)\n"
"\n"
"  -h, --help                 Show this help and exit\n",
        prog);
}

// Consume the value that must follow a value-taking flag; usage+exit if absent.
static const char* argValue(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        fprintf(stderr, "Error: %s requires a value\n\n", argv[i]);
        printUsage(argv[0]);
        std::exit(1);
    }
    return argv[++i];
}

int main(int argc, char* argv[]) {

    // FT8 always on; JS8 Normal/Fast/Slow + CW on by default (--no-* to disable).
    // Turbo/Ultra remain opt-in.
    bool        enable_js8       = true;
    bool        enable_js8_fast  = true;
    bool        enable_js8_slow  = true;
    bool        enable_js8_turbo = false;
    bool        enable_js8_ultra = false;
    bool        enable_cw        = true;
    bool        legacy_costas   = true;
    int         tci_port        = 40001;  // 0 = disabled
    float       tci_gain        = 1.0f;
    float       min_score       = -1.0f;  // sentinel: resolved after flag parsing
    // HFChannelizer packs FT8/JS8 sub-band windows into composite 0–211 kHz;
    // bins above that are zero-filled.  Show only the live content region.
    static constexpr float kCompositeContentHz = 211000.0f;  // 2110 bins × 100 Hz
    float       wf_center_hz  = kCompositeContentHz / 2.0f;  // 45000 Hz
    float       wf_bw_hz      = kCompositeContentHz;          // 90000 Hz
    std::string record_file;
    std::string record_cw_file;
    std::string playback_file;
    std::string playback_cw_file;
    bool        playback_loop   = true;   // --playback loops; --playback-once does not
    int         zoom_band_start = -1;
    int         zoom_band_end   = -1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--min-score") == 0) {
            min_score = std::stof(argValue(i, argc, argv));
        } else if (strcmp(argv[i], "--waterfall-center-hz") == 0) {
            wf_center_hz = std::stof(argValue(i, argc, argv));
        } else if (strcmp(argv[i], "--waterfall-bw-hz") == 0) {
            wf_bw_hz = std::stof(argValue(i, argc, argv));
        } else if (strcmp(argv[i], "--record") == 0) {
            record_file = argValue(i, argc, argv);
        } else if (strcmp(argv[i], "--record-cw") == 0) {
            record_cw_file = argValue(i, argc, argv);
            enable_cw = true;   // the CW composite only exists when --cw is on
        } else if (strcmp(argv[i], "--playback") == 0) {
            playback_file = argValue(i, argc, argv);
            playback_loop = true;
        } else if (strcmp(argv[i], "--playback-once") == 0) {
            playback_file = argValue(i, argc, argv);
            playback_loop = false;
        } else if (strcmp(argv[i], "--playback-cw") == 0) {
            playback_cw_file = argValue(i, argc, argv);
        } else if (strcmp(argv[i], "--js8") == 0) {
            enable_js8 = true;          // (on by default; kept for compatibility)
        } else if (strcmp(argv[i], "--js8-fast") == 0) {
            enable_js8_fast = true;
        } else if (strcmp(argv[i], "--js8-slow") == 0) {
            enable_js8_slow = true;
        } else if (strcmp(argv[i], "--js8-turbo") == 0) {
            enable_js8_turbo = true;
        } else if (strcmp(argv[i], "--cw") == 0) {
            enable_cw = true;
        } else if (strcmp(argv[i], "--js8-ultra") == 0) {
            enable_js8_ultra = true;
        } else if (strcmp(argv[i], "--no-js8") == 0) {
            enable_js8 = false;
        } else if (strcmp(argv[i], "--no-js8-fast") == 0) {
            enable_js8_fast = false;
        } else if (strcmp(argv[i], "--no-js8-slow") == 0) {
            enable_js8_slow = false;
        } else if (strcmp(argv[i], "--no-cw") == 0) {
            enable_cw = false;
        } else if (strcmp(argv[i], "--max-log-costas") == 0) {
            legacy_costas = false;
        } else if (strcmp(argv[i], "--zoom-band") == 0) {
            zoom_band_start = std::stoi(argValue(i, argc, argv));
            zoom_band_end   = std::stoi(argValue(i, argc, argv));
        } else if (strcmp(argv[i], "--tci-port") == 0) {
            tci_port = std::stoi(argValue(i, argc, argv));
        } else if (strcmp(argv[i], "--tci-gain") == 0) {
            tci_gain = std::stof(argValue(i, argc, argv));
        } else {
            fprintf(stderr, "Error: unknown argument '%s'\n\n", argv[i]);
            printUsage(argv[0]);
            return 1;
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

    if (tci_port > 0) {
        if (tci_server_start((uint16_t)tci_port, tci_gain) != 0) {
            fprintf(stderr, "Failed to start TCI server on port %d\n", tci_port);
            return 1;
        }
        printf("TCI server: ws://localhost:%d  (WSJT-X, JTDX, N1MM)  gain=%.2f\n",
               tci_port, (double)tci_gain);
    }

    if (!playback_cw_file.empty()) {
        gm::buffer::BufferFile<std::complex<float>> playback(playback_cw_file);
        playback.start();
        runCWPlayback(*playback.getBuffer());
    } else if (!playback_file.empty()) {
        gm::buffer::BufferFile<std::complex<float>> playback(playback_file, playback_loop);
        playback.start();
        printf("Playback: %s (%s)\n", playback_file.c_str(),
               playback_loop ? "looping at EOF" : "once, then exit");
        runPipeline(*playback.getBuffer(), min_score, enable_js8, enable_js8_fast,
                    enable_js8_slow, enable_js8_turbo, enable_js8_ultra,
                    wf_bin_start, wf_bin_end, legacy_costas,
                    &playback);
    } else {
        gm::rx888::rx888 mydsp;
        mydsp.start_card();

        gm::cuda::HFChannelizer channelizer(mydsp.getRxBufferPosition(), kWsDictPort, enable_cw);
        channelizer.start();

        std::unique_ptr<gm::buffer::BufferFile<std::complex<float>>> recorder;
        if (!record_file.empty()) {
            recorder = std::make_unique<gm::buffer::BufferFile<std::complex<float>>>(
                channelizer.getBuffer(), record_file, channelizer.getBufferFileParams());
            recorder->start();
        }

        // Record the CW composite (819.2 kHz, 9 CW sub-bands) to its own file, in
        // the same BufferFile format as --record.  --record-cw forces --cw above.
        std::unique_ptr<gm::buffer::BufferFile<std::complex<float>>> recorder_cw;
        if (!record_cw_file.empty()) {
            recorder_cw = std::make_unique<gm::buffer::BufferFile<std::complex<float>>>(
                channelizer.getCWBuffer(), record_cw_file, channelizer.getCWBufferFileParams());
            recorder_cw->start();
            printf("CW record: %s  (819.2 kHz composite)\n", record_cw_file.c_str());
        }

        runPipeline(*channelizer.getBuffer(), min_score, enable_js8, enable_js8_fast,
                    enable_js8_slow, enable_js8_turbo, enable_js8_ultra,
                    wf_bin_start, wf_bin_end, legacy_costas,
                    /*playback=*/nullptr,
                    enable_cw ? channelizer.getCWBuffer() : nullptr);

        // TC5/TC6 shutdown order: stop channelizer workers before TCI server
        // so tciVfoWorker never polls a destroyed Rust global.
        channelizer.stop();
        channelizer.join();
        if (tci_port > 0) tci_server_stop();
    }

    return 0;
}
