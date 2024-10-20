#include <string>
#include <string.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <libsddc.h>
#include "gm/rx888/rx888.h"
#include "gm/Thread.h"


namespace gm {
namespace rx888 {

static void count_bytes_callback(uint32_t data_size,
                                 const int16_t *data,
                                 void *context)
{
    rx888* module = (rx888*) context;
    module->handle_samples(data_size, data);
}

rx888::rx888() :
    rx_samplerate(140000000),
    device_str("\0"),
    dev(NULL),
    position(0),
    rxBufferPosition() {
	// gm::zmqcode::zmqWorker().getFuncMap().insert({"setRxGain", setRxGain()});
}

rx888::~rx888() {
}

// gm::zmqcode::func_t rx888::setRxGain() {
// 	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
// 		zmq::message_t message;
// 		int more;
// 		size_t more_size = sizeof (more);
// 		socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);		
// 		socket->send(*identity, ZMQ_SNDMORE);
// 		std::string func("NO ERROR SETRXGAIN");
// 		socket->send(func.c_str(), func.size());
// 	};
// }

int rx888::handle_samples(uint32_t data_size, const int16_t *data) {
    position += (uint64_t)data_size;
    memcpy(&rxBufferPosition.getBuffer()[(int)(position % rx_buffer_size)], data, data_size*sizeof(int16_t));
    //XXX handle the wrap??
    rxBufferPosition.setPosition(position);
    return 0;
}

struct sddc* rx888::initialize_device()
{
    sddc_t* sddc = NULL;
    try {

        sddc = sddc_open(rx_samplerate, count_bytes_callback, (void*)this);
        if (sddc == 0) {
            throw "Failed to init sddc module:\n";
        }

        if (sddc_set_sample_rate(sddc, rx_samplerate) < 0) {
            throw "Failed to set sample rate:\n";
        }

        if (sddc_set_async_params(sddc, 0, 0, sddc) < 0) {
            throw "Failed to setup callbacks:\n";
        }

        sddc_set_rf_mode(sddc, HF_MODE);

        int16_t* rxBuffer = (int16_t*)calloc(rx_buffer_size, sizeof(int16_t));
        rxBufferPosition.setBuffer(rxBuffer, {BUFFERS,NLARGE});

        if (sddc_start_streaming(sddc) < 0) {
            throw "ERROR - sddc_start_streaming() failed\n";
        }

    } catch (const char* error_string) {
        fprintf(stderr, "Got and exception %s\n", error_string);
        stop_card();
        sddc = NULL;
    }

    return sddc;
}


/* For the sake of simplicity, we'll just bounce back and forth between
 * RX and TX when doing both. */
int rx888::start_card()
{
    dev = initialize_device();
    if (dev == NULL) {
        printf("Cant initialize device \n");
        return -1;
    }

    printf("device initialized\n");
    printf("Starting RX task\n");
    printf("Running...\n");

    return 0;

}

int rx888::stop_card() {
    sddc_stop_streaming(dev);
    return 0;

}

}
}

