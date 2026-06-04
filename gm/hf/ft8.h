#ifndef _GM_HF_FT8_H_
#define _GM_HF_FT8_H_

#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <zmq.hpp>
#include "gm/Thread.h"

// Forward-declare CUDA types so ft8.h doesn't pull in CUDA headers.
namespace gm { namespace cuda { class FT8Cuda; } }
namespace gm { namespace cuda { struct ContScanResult; } }
class WsDictClient;

namespace gm {
namespace hf {

class FT8 : public Thread {

public:
    FT8(gm::cuda::FT8Cuda* ft8cuda,
        int zmq_port = 5580,
        int wsdict_port = 0);
    ~FT8();
    void run();
    void stop() { setRunning(false); }

    // Decode and publish candidates from the continuous Costas scan path.
    // Called from contWorker via the FT8Cuda decode callback.
    void decodeAndPublishContinuous(gm::cuda::ContScanResult& r);

private:
    gm::cuda::FT8Cuda* ft8cuda_;
    int zmq_port_;

    zmq::context_t zmq_ctx_;
    zmq::socket_t  zmq_pub_;
    std::mutex     zmq_mutex_;
    FILE*          timing_log_;
    std::unique_ptr<WsDictClient> ws_client_;

    struct WindowSpot {
        float  snr;
        float  freq_hz;
        double unix_time;
        float  time_offset;
    };

    std::unordered_map<std::string, WindowSpot> window_buf_;
    std::mutex   window_mu_;
    double       window_start_{0.0};

    void publishDecoded(const char* callsign, float freq_hz, float snr,
                        double unix_time, float time_offset);
    void flushWindow(double now);
};

}
}

#endif // _GM_HF_FT8_H_
