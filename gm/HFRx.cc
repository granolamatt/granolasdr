#include <stdio.h>
#include <unistd.h>
#include <cstring>
#include <complex>
#include <cmath>
#include <iostream>
#include <sstream>

#include "gm/rx888/rx888.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/HostCuda.h"
#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/FT8Cuda.h"
#include "gm/hf/ft8.h"


int main(int argc, char* argv[]) {

    bool enable_corpus = false;
    bool use_gpu_ldpc  = false;
    std::string ctrl_host = "127.0.0.1";
    int ctrl_port = 8080;
    float min_score = 3.0f;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--jtdx") == 0) {
            enable_corpus = true;
        } else if (strcmp(argv[i], "--gpu-ldpc") == 0) {
            use_gpu_ldpc = true;
        } else if (strcmp(argv[i], "--control-host") == 0 && i + 1 < argc) {
            ctrl_host = argv[++i];
        } else if (strcmp(argv[i], "--control-port") == 0 && i + 1 < argc) {
            ctrl_port = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--min-score") == 0 && i + 1 < argc) {
            min_score = std::stof(argv[++i]);
        }
    }

    gm::rx888::rx888 mydsp;
    mydsp.start_card();

    gm::cuda::HFChannelizer epochbuffer(mydsp.getRxBufferPosition(), ctrl_host, ctrl_port);
    epochbuffer.start();

    gm::cuda::FT8Cuda ft8channel(epochbuffer.getBuffer(), enable_corpus, min_score);
    ft8channel.start();

    gm::hf::FT8 ft8(ft8channel.getBuffer(), &ft8channel, 5580, use_gpu_ldpc);
    ft8channel.setDecodeCallback([&ft8](gm::cuda::ContScanResult& r) {
        ft8.decodeAndPublishContinuous(r);
    });
    ft8channel.startContinuousScan();

    // Wire SSE broadcast: FT8Cuda timing → HFChannelizer SSE, FT8 decode → HFChannelizer SSE.
    ft8channel.setTimingCallback([&epochbuffer](float scan_ms, float ldpc_ms, uint32_t n) {
        epochbuffer.broadcastTiming(scan_ms, ldpc_ms, n);
    });
    ft8.setBroadcastCallback([&epochbuffer](const char* call, float freq_hz, float snr, double unix_time) {
        epochbuffer.broadcastDecode(call, freq_hz, snr, unix_time);
    });
    ft8channel.setWaterfallCallback([&epochbuffer](const uint8_t* data, int len) {
        epochbuffer.broadcastWaterfall(data, len);
    });

    ft8.start();

    while (true) {
        usleep(1000000);
    }

    return 0;
}
