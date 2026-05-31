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
#include "gm/hf/ft8.h"
#include "gm/hf/js8.h"
#include "gm/buffer/BufferFile.h"

static constexpr int kProxyXSubPort = 5599;  // producers connect here
static constexpr int kProxyXPubPort = 5600;  // consumers subscribe here

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
                        float min_score, bool enable_js8, bool enable_js8_fast) {

    // RAII order: MagBlock owns ring memory; FT8Cuda/JS8Cuda hold const refs.
    // C++ destroys in reverse declaration order (scanners before ring).
    gm::cuda::MagBlock<200> magblock(&buf, 1048576, 4, 4, kProxyXSubPort);
    magblock.start();

    gm::cuda::FT8Cuda ft8channel(magblock.getRing(), min_score, "EPOCH", kProxyXSubPort);
    ft8channel.start();

    gm::hf::FT8 ft8(&ft8channel, kProxyXSubPort);
    ft8.start();

    std::unique_ptr<gm::cuda::JS8Cuda<200>> js8channel;
    std::unique_ptr<gm::hf::JS8>            js8_obj;
    if (enable_js8) {
        js8channel = std::make_unique<gm::cuda::JS8Cuda<200>>(
            magblock.getRing(), min_score, kProxyXSubPort,
            js8_gpu_scan, 4, 4, 106);
        js8_obj = std::make_unique<gm::hf::JS8>(
            js8channel.get(), kProxyXSubPort, 0.160f, 15.0f, 4, 1048576, "JS8");
    }

    std::unique_ptr<gm::cuda::MagBlock<100>>  magblock_fast;
    std::unique_ptr<gm::cuda::JS8Cuda<100>>   js8fast_channel;
    std::unique_ptr<gm::hf::JS8>              js8fast_obj;
    if (enable_js8_fast) {
        magblock_fast = std::make_unique<gm::cuda::MagBlock<100>>(
            &buf, 655360, 2, 2, 0);
        magblock_fast->start();

        js8fast_channel = std::make_unique<gm::cuda::JS8Cuda<100>>(
            magblock_fast->getRing(), min_score, kProxyXSubPort,
            js8_fast_gpu_scan, 2, 2, 100);
        js8fast_obj = std::make_unique<gm::hf::JS8>(
            js8fast_channel.get(), kProxyXSubPort, 0.100f, 10.0f, 2, 655360, "JS8-FAST");
    }

    while (true) {
        usleep(1000000);
    }
}


int main(int argc, char* argv[]) {

    bool        enable_js8      = false;
    bool        enable_js8_fast = false;
    std::string ctrl_host       = "127.0.0.1";
    int         ctrl_port       = 8080;
    float       min_score       = 3.0f;
    std::string record_file;
    std::string playback_file;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--control-host") == 0 && i + 1 < argc) {
            ctrl_host = argv[++i];
        } else if (strcmp(argv[i], "--control-port") == 0 && i + 1 < argc) {
            ctrl_port = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-score") == 0 && i + 1 < argc) {
            min_score = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--record") == 0 && i + 1 < argc) {
            record_file = argv[++i];
        } else if (strcmp(argv[i], "--playback") == 0 && i + 1 < argc) {
            playback_file = argv[++i];
        } else if (strcmp(argv[i], "--js8") == 0) {
            enable_js8 = true;
        } else if (strcmp(argv[i], "--js8-fast") == 0) {
            enable_js8 = true;
            enable_js8_fast = true;
        }
    }

    std::thread(runProxy).detach();
    printf("ZMQ proxy: XSUB tcp://*:%d  XPUB tcp://*:%d\n",
           kProxyXSubPort, kProxyXPubPort);

    if (!playback_file.empty()) {
        gm::buffer::BufferFile<std::complex<float>> playback(playback_file);
        playback.start();
        runPipeline(*playback.getBuffer(), min_score, enable_js8, enable_js8_fast);
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

        runPipeline(*channelizer.getBuffer(), min_score, enable_js8, enable_js8_fast);
    }

    return 0;
}
