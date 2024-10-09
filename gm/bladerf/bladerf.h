#ifndef _GM_BLADERF_H_
#define _GM_BLADERF_H_

#include <string>
#include <complex>
#include <cmath>
#include <libbladeRF.h>
#include "gm/Thread.h"
#include "gm/zmqcode/zmqworker.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace bladerf {

class rxTask;
class txTask;

class bladeRF {

private:
    std::string device_str;
    unsigned int rx_samplerate;
    unsigned int rx_frequency;
    bool tx_enable;
    unsigned int tx_samplerate;
    unsigned int tx_frequency;
    struct bladerf *dev;
    unsigned int samplerate_actual;
    unsigned long int frequency_actual;
    unsigned int bandwidth_actual;
    txTask* txThread;
    rxTask* rxThread;
    const static unsigned int block_size = 512;
    const static unsigned int num_xfers = 16;
    const static unsigned int timeout_ms = 1000;
    gm::buffer::BufferPosition<std::complex<short>> rxBufferPosition;
    gm::buffer::BufferPosition<std::complex<short>> txBufferPosition;
    //std::complex<short> *rxBuffer;
    //std::complex<short> *txBuffer;
    struct bladerf* initialize_device();
    gm::zmqcode::func_t setRxGain();
    gm::zmqcode::func_t getRxFreq();
    gm::zmqcode::func_t setRxFreq();
    gm::zmqcode::func_t setRxCalibration();
    const static int N = 4096;
    int init_module(struct bladerf *dev, unsigned int samplerate, unsigned int frequency,
                    bladerf_module m);
// 	bladerf_loopback loopback;

// //	RawTunerStruct *tuner_info;
// 	unsigned int tx_repetitions;
// 	unsigned int block_size;

// 	/* Stream config */
// 	unsigned int num_xfers;
// 	unsigned int stream_buffer_count;
// 	unsigned int stream_buffer_size;    /* Units of samples */
// 	unsigned int timeout_ms;
public:
    bladeRF();
    ~bladeRF();
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
    std::complex<short>* getTxBuffer() {
        return txBufferPosition.getBuffer();
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
    gm::buffer::BufferPosition<std::complex<short>>* getTxBufferPosition() {
        return &txBufferPosition;
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

class txTask : public gm::Thread {
private:
    bladeRF* device;
    bool done;
public:
    txTask(bladeRF* device);
    ~txTask();
    void setDone() {
        done = true;
    }
    virtual void run();
};

}
}

#endif //_GM_BLADERF_H_