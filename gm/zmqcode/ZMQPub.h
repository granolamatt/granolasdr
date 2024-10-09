#ifndef _GM_ZMQCODE_ZMQPUB_H_
#define _GM_ZMQCODE_ZMQPUB_H_

//#include <zmq.hpp>
#include <complex>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace zmqcode {

class ZMQPub : public Runnable {
public :
    ZMQPub(gm::buffer::BufferPosition<float>* inP, float* fmsquelch, float* ave);
    ~ZMQPub();
    void run();
private :
    gm::buffer::BufferPosition<float>* inputPos;
    float* squelch;
    float* ave;
};

}
}

#endif //_GM_ZMQCODE_ZMQPUB_H_