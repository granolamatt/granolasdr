#ifndef _GM_HF_FT8_H_
#define _GM_HF_FT8_H_

#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "ft8_lib/common/monitor.h"

namespace gm {
namespace hf {

class FT8 : public Thread {

public:
    FT8(gm::buffer::BufferPosition<std::complex<float>>* inP);
    ~FT8();
    void run();
    void stop() {
        setRunning(false);
    }
private:
    monitor_t mon;
    gm::buffer::BufferPosition<std::complex<float>>* inPos;
    std::complex<float>* demodFT8;

};

}
}

#endif // _GM_HF_FT8_H_