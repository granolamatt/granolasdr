#ifndef _GM_CUDA_FREQSCAN_
#define _GM_CUDA_FREQSCAN_

#include <cuda.h>
#include <cufft.h>
#include <thrust/complex.h>
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"

namespace gm {
namespace cuda {
class FreqScan {
public:
    const static int NLARGE = 131072;
    const static int NSMALL = 4096;
    const static int NSCANS = 200;
    const static int NSBUFFERS = 8;
    const static int NSTREAMS = 4;
    FreqScan(gm::buffer::BufferPosition<std::complex<short>>* inP);
    ~FreqScan();
    long capture(long epoch);
    gm::buffer::BufferPosition<float>* getOutputBufferPos() {
        return &outPos;
    }
private:
    float* aveData_d;
    thrust::complex<float>* fftData_d;
    thrust::complex<short>* inData_d;
    int dmaSize;
//    long epoch;
    int streamNum;
    int bufferNum;
    int captureNum;
    thrust::complex<short>* inData;
    gm::buffer::BufferPosition<std::complex<short>>* inPos;
    gm::buffer::BufferPosition<float> outPos;
    CudaCopy* cudaCopy[NSBUFFERS];
    cufftHandle plan[NSTREAMS];
    cudaStream_t stream[NSTREAMS];
};
}
}

#endif //_GM_CUDA_FREQSCAN_