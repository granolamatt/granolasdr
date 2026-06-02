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
#include "gm/cuda/WaterfallCuda.h"
#include "gm/hf/ft8.h"
#include "gm/hf/js8.h"
#include "gm/buffer/BufferFile.h"

static constexpr int kProxyXSubPort = 5599;  // producers connect here
static constexpr int kProxyXPubPort = 5600;  // consumers subscribe here

// Hz to bin index for the 1,048,576-bin / 6.553600 MHz ring.
static int hzToBin(float hz) {
    return (int)(hz / 6.25f + 0.5f);
}

// Runs zmq_proxy(XSUB, XPUB) — blocks forever, call in a detached thread.
static void runProxy() {
    zmq::context_t ctx(1);
    zmq::socket_t xsub(ctx, ZMQ_XSUB);
    xsub.bind("tcp://*:5599");
    zmq::socket_t xpub(ctx, ZMQ_XPUB);
    xpub.bind("tcp://*:5600");
    zmq_proxy(xsub.handle(), xpub.handle(), nullptr);
}

static void runPipeline(gm::buffer::BufferPosition<std::complex<float>>& buf,
                        float min_score, bool enable_js8,
                        int wf_bin_start, int wf_bin_end) {

    // RAII order: MagBlock owns ring memory; all readers hold const refs.
    // C++ destroys in reverse declaration order (readers before ring).
    gm::cuda::MagBlock magblock(&buf, kProxyXSubPort);
    magblock.start();

    gm::cuda::FT8Cuda ft8channel(magblock.getRing(), min_score, "EPOCH", kProxyXSubPort);
    ft8channel.start();

    gm::hf::FT8 ft8(&ft8channel, kProxyXSubPort);
    ft8.start();

    gm::cuda::WaterfallCuda waterfall(magblock.getRing(),
                                      wf_bin_start, wf_bin_end,
                                      gm::cuda::WaterfallCuda::DEFAULT_OUT_BINS,
                                      kProxyXSubPort);
    waterfall.start();

    std::unique_ptr<gm::cuda::JS8Cuda> js8channel;
    std::unique_ptr<gm::hf::JS8>       js8_obj;
    if (enable_js8) {
        js8channel = std::make_unique<gm::cuda::JS8Cuda>(
            magblock.getRing(), min_score, kProxyXSubPort);
        js8_obj = std::make_unique<gm::hf::JS8>(js8channel.get(), kProxyXSubPort);
    }

    while (true) {
        usleep(1000000);
    }
}

int main(int argc, char* argv[]) {

    bool        enable_js8    = false;
    std::string ctrl_host     = "127.0.0.1";
    int         ctrl_port     = 8080;
    float       min_score     = 3.0f;
    float       wf_start_hz   = 0.0f;
    float       wf_end_hz     = 6553600.0f;  // full band by default
    std::string record_file;
    std::string playback_file;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--control-host") == 0 && i + 1 < argc) {
            ctrl_host = argv[++i];
        } else if (strcmp(argv[i], "--control-port") == 0 && i + 1 < argc) {
            ctrl_port = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-score") == 0 && i + 1 < argc) {
            min_score = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--waterfall-start-hz") == 0 && i + 1 < argc) {
            wf_start_hz = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--waterfall-end-hz") == 0 && i + 1 < argc) {
            wf_end_hz = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--record") == 0 && i + 1 < argc) {
            record_file = argv[++i];
        } else if (strcmp(argv[i], "--playback") == 0 && i + 1 < argc) {
            playback_file = argv[++i];
        } else if (strcmp(argv[i], "--js8") == 0) {
            enable_js8 = true;
        }
    }

    int wf_bin_start = hzToBin(wf_start_hz);
    int wf_bin_end   = hzToBin(wf_end_hz);
    wf_bin_end       = std::min(wf_bin_end, (int)1048576);

    std::thread(runProxy).detach();
    printf("ZMQ proxy: XSUB tcp://*:%d  XPUB tcp://*:%d\n",
           kProxyXSubPort, kProxyXPubPort);

    if (!playback_file.empty()) {
        gm::buffer::BufferFile<std::complex<float>> playback(playback_file);
        playback.start();
        runPipeline(*playback.getBuffer(), min_score, enable_js8,
                    wf_bin_start, wf_bin_end);
    } else {
        gm::rx888::rx888 mydsp;
        mydsp.start_card();

        gm::cuda::HFChannelizer channelizer(mydsp.getRxBufferPosition(), ctrl_host, ctrl_port);
        channelizer.start();

        std::unique_ptr<gm::buffer::BufferFile<std::complex<float>>> recorder;
        if (!record_file.empty()) {
            recorder = std::make_unique<gm::buffer::BufferFile<std::complex<float>>>(
                channelizer.getBuffer(), record_file, channelizer.getBufferFileParams());
            recorder->start();
        }

        runPipeline(*channelizer.getBuffer(), min_score, enable_js8,
                    wf_bin_start, wf_bin_end);
    }

    return 0;
}
