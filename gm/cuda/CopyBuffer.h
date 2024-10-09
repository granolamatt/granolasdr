#ifndef _GM_CUDA_COPYBUFFER_H_
#define _GM_CUDA_COPYBUFFER_H_

#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {
class CopyBuffer : public Runnable {
public:
    CopyBuffer(gm::buffer::BufferPosition<std::complex<short>>* inP);
    ~CopyBuffer();
    void run();
    void stop() {
        setRunning(false);
    }
    gm::buffer::BufferPosition<std::complex<short>>* getOutputBufferPos() {
        return &outPos;
    }
    void setSize(int nsize) {
        outSize = nsize;
    }
    
private:
    // bool running;
    int outSize;
    int inSize;
    int sampleSize;
    std::complex<short>* outData_d;
    std::complex<short>* inData;
    gm::buffer::BufferPosition<std::complex<short>>* inPos;
    gm::buffer::BufferPosition<std::complex<short>> outPos;
    int doCopy(long now, long length);
};
}
}

#endif // _GM_CUDA_COPYBUFFER_H_
