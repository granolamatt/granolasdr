#ifndef _GM_CUDA_COPYEPOCH_H_
#define _GM_CUDA_COPYEPOCH_H_

#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

template<class T>
class CopyEpoch : public Runnable {
public:
    CopyEpoch(gm::buffer::BufferPosition<T>* inP);
    ~CopyEpoch();
    void run();
    void stop() {
        setRunning(false);
    }
    
    gm::buffer::BufferPosition<T>* getOutputBufferPos() {
        return &outPos;
    }
    
private:
    // bool running;
    std::vector<size_t> inShape;
    T* outData_d;
    T* inData;
    gm::buffer::BufferPosition<T>* inPos;
    gm::buffer::BufferPosition<T> outPos;
    int doCopy(uint64_t now);
};
}
}

#endif // _GM_CUDA_COPYEPOCH_H_
