#include <unistd.h>
#include <cstring>

#include "gm/rx888/rx888.h"
#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/FileChannelizer.h"
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/JS8Cuda.h"
#include "gm/hf/ft8.h"
#include "gm/hf/js8.h"


// Wire FT8Cuda/FT8/JS8Cuda/JS8 on top of any channelizer that provides
// getBuffer(), broadcastTiming/Decode/Waterfall, and setTimingCallback.
template<typename Channelizer>
static void runPipeline(Channelizer& epochbuffer,
                        bool enable_corpus, float min_score, bool use_gpu_ldpc,
                        bool enable_js8) {

    gm::cuda::FT8Cuda ft8channel(epochbuffer.getBuffer(), enable_corpus, min_score);
    ft8channel.start();

    gm::hf::FT8 ft8(ft8channel.getBuffer(), &ft8channel, 5580, use_gpu_ldpc);
    ft8channel.setDecodeCallback([&ft8](gm::cuda::ContScanResult& r) {
        ft8.decodeAndPublishContinuous(r);
    });
    ft8channel.startContinuousScan();

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

    // JS8 objects declared at function scope so they outlive the loop below.
    std::unique_ptr<gm::hf::JS8>      js8_obj;
    std::unique_ptr<gm::cuda::JS8Cuda> js8channel;
    if (enable_js8) {
        js8_obj     = std::make_unique<gm::hf::JS8>(5590);
        js8channel  = std::make_unique<gm::cuda::JS8Cuda>(&ft8channel, min_score);
        js8channel->setDecodeCallback([&js8_obj](gm::cuda::ContScanResult& r) {
            js8_obj->decodeAndPublishContinuous(r);
        });
        js8channel->start();
    }

    while (true) {
        usleep(1000000);
    }
}


int main(int argc, char* argv[]) {

    bool        enable_corpus  = false;
    bool        use_gpu_ldpc   = false;
    bool        enable_js8     = false;
    std::string ctrl_host      = "127.0.0.1";
    int         ctrl_port      = 8080;
    float       min_score      = 3.0f;
    std::string record_file;
    std::string playback_file;

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
        } else if (strcmp(argv[i], "--record") == 0 && i + 1 < argc) {
            record_file = argv[++i];
        } else if (strcmp(argv[i], "--playback") == 0 && i + 1 < argc) {
            playback_file = argv[++i];
        } else if (strcmp(argv[i], "--js8") == 0) {
            enable_js8 = true;
        }
    }

    if (!playback_file.empty()) {
        gm::cuda::FileChannelizer epochbuffer(playback_file);
        epochbuffer.start();
        runPipeline(epochbuffer, enable_corpus, min_score, use_gpu_ldpc, enable_js8);
    } else {
        gm::rx888::rx888 mydsp;
        mydsp.start_card();

        gm::cuda::HFChannelizer epochbuffer(mydsp.getRxBufferPosition(), ctrl_host, ctrl_port);
        epochbuffer.start();

        if (!record_file.empty()) {
            epochbuffer.startRecording(record_file);
        }

        runPipeline(epochbuffer, enable_corpus, min_score, use_gpu_ldpc, enable_js8);
    }

    return 0;
}
