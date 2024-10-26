#ifndef _GM_RX888_H_
#define _GM_RX888_H_

#include <string>
#include <cmath>
#include <libsddc.h>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace rx888 {

class rx888 {

private:
    std::string device_str;
    struct sddc *dev;
    gm::buffer::BufferPosition<int16_t> rxBufferPosition;
    struct sddc* initialize_device();
    //gm::zmqcode::func_t setRxGain();
    const static int N = 4096;
    uint64_t position;
public:
    rx888();
    ~rx888();
    const static int NLARGE = 1048576;
    const static int BUFFERS = 16;
    const static uint32_t rx_samplerate = 140000000;

    const static int rx_buffer_size = NLARGE * BUFFERS; // Normally rx888 gets 65536 samples
    int start_card();
    int stop_card();
    int handle_samples(uint32_t data_size, const int16_t *data);
    int16_t* getRxBuffer() {
        return rxBufferPosition.getBuffer();
    }
    struct sddc* getDev() {
        return dev;
    }
    // unsigned int getRxSampleRate() {
    //     return rx_samplerate;
    // }
    // void setRxSampleRate(unsigned int samplerate) {
    //     rx_samplerate = samplerate;
    // }
    gm::buffer::BufferPosition<int16_t>* getRxBufferPosition() {
        return &rxBufferPosition;
    }
    
};

}
}

#endif //_GM_RX888_H_
