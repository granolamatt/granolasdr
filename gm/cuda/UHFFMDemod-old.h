#ifndef _GM_CUDA_UHFFMDEMOD_H_
#define _GM_CUDA_UHFFMDEMOD_H_

#include <cuda.h>
#include <cufft.h>
#include <thrust/complex.h>
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"
#include "gm/cuda/FMDemodLong.h"
#include "gm/zmqcode/zmqworker.h"

namespace gm {
namespace cuda {
class UHFFMDemod : public Runnable {
public:
    const static int NLARGE = (1048576);
    const static int NEPOCH = (NLARGE/2);
    const static int NSBUFFERS = 16;
    const static int NSTREAMS = 4;
    UHFFMDemod(gm::buffer::BufferPosition<std::complex<short>>* inP);
    ~UHFFMDemod();
    void run();
    void stop() {
        running = false;
    }
    gm::buffer::BufferPosition<std::complex<float>>* getOutputBufferPos() {
        return &outPos;
    }
private:
    thrust::complex<float>* fftData_d;
    thrust::complex<float>* fftDatao_d;
    thrust::complex<short>* inData_d;
    int dmaSize;
    bool running;
    long epoch;
    int streamNum;
    int bufferNum;
    thrust::complex<short>* inData;
    gm::buffer::BufferPosition<std::complex<short>>* inPos;
    gm::buffer::BufferPosition<std::complex<float>> outPos;
    CudaCopy* cudaCopy[NSBUFFERS];
    FMDemodLong* fmDemod[NSBUFFERS];
    cufftHandle plan[NSTREAMS];
    cufftHandle iplan[NSTREAMS];
    cudaStream_t stream[NSTREAMS];
    gm::zmqcode::func_t processSamples();
    gm::zmqcode::func_t setFreqRange();
    gm::zmqcode::func_t getFreqs();
};
}
}

#endif //_GM_CUDA_UHFFMDEMOD_H_
