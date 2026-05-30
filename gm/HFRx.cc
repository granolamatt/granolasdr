#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <thread>
#include <zmq.hpp>

#include "gm/rx888/rx888.h"
#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/FileChannelizer.h"
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/JS8Cuda.h"
#include "gm/hf/ft8.h"
#include "gm/hf/js8.h"

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

// Thin publisher used in HFRx for components (FT8Cuda, JS8Cuda) whose
// callbacks are still wired here.  Thread-safe; each send is two frames:
// [topic, payload].
struct BusPub {
    zmq::context_t ctx{1};
    zmq::socket_t  pub{ctx, ZMQ_PUB};
    std::mutex     mu;

    BusPub() { pub.connect("tcp://localhost:5599"); }

    void sendJson(const char* topic, const char* json) {
        std::lock_guard<std::mutex> lk(mu);
        zmq::message_t t(topic, strlen(topic));
        zmq::message_t d(json,  strlen(json));
        pub.send(t, zmq::send_flags::sndmore);
        pub.send(d, zmq::send_flags::none);
    }
    void sendBin(const char* topic, const void* data, size_t len) {
        std::lock_guard<std::mutex> lk(mu);
        zmq::message_t t(topic, strlen(topic));
        zmq::message_t d(data, len);
        pub.send(t, zmq::send_flags::sndmore);
        pub.send(d, zmq::send_flags::none);
    }
};

template<typename Channelizer>
static void runPipeline(Channelizer& epochbuffer,
                        bool enable_corpus, float min_score, bool use_gpu_ldpc,
                        bool enable_js8) {

    BusPub bus;

    gm::cuda::FT8Cuda ft8channel(epochbuffer.getBuffer(), enable_corpus, min_score);
    ft8channel.start();

    gm::hf::FT8 ft8(ft8channel.getBuffer(), &ft8channel, kProxyXSubPort, use_gpu_ldpc);
    ft8channel.setDecodeCallback([&ft8](gm::cuda::ContScanResult& r) {
        ft8.decodeAndPublishContinuous(r);
    });
    ft8channel.startContinuousScan();

    ft8channel.setTimingCallback([&bus](float scan_ms, float ldpc_ms, uint32_t n) {
        char buf[128];
        snprintf(buf, sizeof(buf),
            "{\"scan_ms\":%.1f,\"ldpc_ms\":%.1f,\"n\":%u}", scan_ms, ldpc_ms, n);
        bus.sendJson("ft8/timing", buf);
    });
    ft8channel.setWaterfallCallback([&bus](const uint8_t* data, int len) {
        bus.sendBin("waterfall", data, (size_t)len);
    });

    ft8.start();

    std::unique_ptr<gm::hf::JS8>       js8_obj;
    std::unique_ptr<gm::cuda::JS8Cuda> js8channel;
    if (enable_js8) {
        js8_obj    = std::make_unique<gm::hf::JS8>(kProxyXSubPort);
        js8channel = std::make_unique<gm::cuda::JS8Cuda>(&ft8channel, min_score);
        js8channel->setDecodeCallback([&js8_obj](gm::cuda::ContScanResult& r) {
            js8_obj->decodeAndPublishContinuous(r);
        });
        js8channel->setTimingCallback([&bus](float scan_ms, float /*ldpc_ms*/, uint32_t n) {
            char buf[96];
            snprintf(buf, sizeof(buf), "{\"scan_ms\":%.1f,\"n\":%u}", scan_ms, n);
            bus.sendJson("js8/timing", buf);
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

    std::thread(runProxy).detach();
    printf("ZMQ proxy: XSUB tcp://*:%d  XPUB tcp://*:%d\n",
           kProxyXSubPort, kProxyXPubPort);

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
