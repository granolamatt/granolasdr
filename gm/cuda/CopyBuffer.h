#ifndef _GM_CUDA_COPYBUFFER_H_
#define _GM_CUDA_COPYBUFFER_H_

#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

template<class T>
class CopyBuffer : public Runnable {
public:
    CopyBuffer(gm::buffer::BufferPosition<T>* inP);
    ~CopyBuffer();
    void run();
    void stop() {
        setRunning(false);
    }
    
    gm::buffer::BufferPosition<T>* getOutputBufferPos() {
        return &outPos;
    }
    void setSize(int nsize) {
        outSize = nsize;
    }
    
private:
    // bool running;
    int outSize;
    int inSize;
    T* outData_d;
    T* inData;
    gm::buffer::BufferPosition<T>* inPos;
    gm::buffer::BufferPosition<T> outPos;
    int doCopy(long now, long length);
};
}
}

#endif // _GM_CUDA_COPYBUFFER_H_
