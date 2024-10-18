#ifndef _GM_CUDA_HFCHANNELIZER_H_
#define _GM_CUDA_HFCHANNELIZER_H_

#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

template<class T>
class HFChannelizer : public Runnable {
public:
    HFChannelizer(gm::buffer::BufferPosition<T>* inP);
    ~HFChannelizer();
    void run();
    void stop() {
        setRunning(false);
    }
        
private:
    // bool running;
    std::vector<size_t> inShape;
    T* outData_d;
    T* inData;
    gm::buffer::BufferPosition<T>* inPos;
    int doCopy(uint64_t now);
};
}
}

#endif // _GM_CUDA_HFCHANNELIZER_H_
