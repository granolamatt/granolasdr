#ifndef _GM_RX888_H_
#define _GM_RX888_H_

#include <string>
#include <complex>
#include <cmath>
#include "gm/Thread.h"
#include "gm/zmqcode/zmqworker.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace rx888 {

class rxTask;

class bladeRF {

private:
    std::string device_str;
    unsigned int rx_samplerate;
    unsigned int rx_frequency;
    struct bladerf *dev;
    unsigned int samplerate_actual;
    unsigned int frequency_actual;
    unsigned int bandwidth_actual;
    rxTask* rxThread;
    const static unsigned int block_size = 512;
    const static unsigned int num_xfers = 16;
    const static unsigned int timeout_ms = 1000;
    gm::buffer::BufferPosition<std::complex<short>> rxBufferPosition;
    struct bladerf* initialize_device();
    gm::zmqcode::func_t setRxGain();
    gm::zmqcode::func_t getRxFreq();
    gm::zmqcode::func_t setRxFreq();
    const static int N = 4096;
    int init_module(struct bladerf *dev, unsigned int samplerate, unsigned int frequency,
                    bladerf_module m);
public:
    rx888();
    ~rx888();
    const static int rx_buffer_size = 8192 * 1024 * 16;
    const static int tx_buffer_size = 8192 * 1024 * 16;
    int change_rx_freq(unsigned int freq);
    unsigned int get_rx_freq();
    int start_card();
    int stop_card();
    unsigned int getBlockSize() {
        return block_size;
    }
    std::complex<short>* getRxBuffer() {
        return rxBufferPosition.getBuffer();
    }
    struct bladerf* getDev() {
        return dev;
    }
    unsigned int getRxSampleRate() {
        return rx_samplerate;
    }
    void setRxSampleRate(unsigned int samplerate) {
        rx_samplerate = samplerate;
    }
    gm::buffer::BufferPosition<std::complex<short>>* getRxBufferPosition() {
        return &rxBufferPosition;
    }
};

class rxTask : public gm::Thread {
private:
    bladeRF* device;
    bool done;
public:
    rxTask(bladeRF* device);
    ~rxTask();
    void setDone() {
        done = true;
    }
    virtual void run();
};

}
}

#endif //_GM_RX888_H_