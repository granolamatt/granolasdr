#ifndef _GM_HF_FT8_H_
#define _GM_HF_FT8_H_

#include <zmq.hpp>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "ft8_lib/common/monitor.h"

namespace gm {
namespace hf {

class FT8 : public Thread {

public:
    FT8(gm::buffer::BufferPosition<uint8_t>* inP, int zmq_port = 5580);
    ~FT8();
    void run();
    void stop() {
        setRunning(false);
    }
    // publish one decoded message; called from decode()
    void publishDecoded(const char* callsign, float freq_hz, float snr,
                        double unix_time, float time_offset);
private:
    monitor_t mon;
    gm::buffer::BufferPosition<uint8_t>* inPos;
    uint8_t* demodFT8;

    zmq::context_t zmq_ctx;
    zmq::socket_t  zmq_pub;
};

}
}

#endif // _GM_HF_FT8_H_