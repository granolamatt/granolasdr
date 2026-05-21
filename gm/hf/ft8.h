#ifndef _GM_HF_FT8_H_
#define _GM_HF_FT8_H_

#include <cstdio>
#include <mutex>
#include <zmq.hpp>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "ft8_lib/common/monitor.h"

// Forward-declare CUDA types so ft8.h doesn't pull in CUDA headers.
namespace gm { namespace cuda { class FT8Cuda; } }

namespace gm {
namespace hf {

class FT8 : public Thread {

public:
    // ft8cuda is optional; non-null enables VALIDATE_GPU_CANDS comparison output.
    FT8(gm::buffer::BufferPosition<uint8_t>* inP,
        gm::cuda::FT8Cuda* ft8cuda = nullptr,
        int zmq_port = 5580);
    ~FT8();
    void run();
    void stop() {
        setRunning(false);
    }
    // publish one decoded message; callable from any thread (zmq_mutex_ protected)
    void publishDecoded(const char* callsign, float freq_hz, float snr,
                        double unix_time, float time_offset);

private:
    monitor_t mon;
    gm::buffer::BufferPosition<uint8_t>* inPos;
    gm::cuda::FT8Cuda* ft8cuda;

    zmq::context_t zmq_ctx;
    zmq::socket_t  zmq_pub;
    std::mutex     zmq_mutex_;
    FILE*          timing_log_;
};

}
}

#endif // _GM_HF_FT8_H_