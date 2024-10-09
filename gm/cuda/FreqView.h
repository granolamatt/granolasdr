#ifndef _GM_CUDA_FREQVIEW_
#define _GM_CUDA_FREQVIEW_

#include <cuda.h>
#include <cufft.h>
#include <thrust/complex.h>
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"

namespace gm {
namespace cuda {
class FreqView {
public:
    const static int NLARGE = 262144;
    const static int NSBUFFERS = 32;
    const static int NSTREAMS = 4;
    FreqView(gm::buffer::BufferPosition<std::complex<short>>* inP);
    ~FreqView();
    long capture();
    gm::buffer::BufferPosition<std::complex<float>>* getOutputBufferPos() {
        return &outPos;
    }
private:
    thrust::complex<float>* fftData_d;
    thrust::complex<short>* inData_d;
    int dmaSize;
    long epoch;
    int streamNum;
    int bufferNum;
    thrust::complex<short>* inData;
    gm::buffer::BufferPosition<std::complex<short>>* inPos;
    gm::buffer::BufferPosition<std::complex<float>> outPos;
    CudaCopy* cudaCopy[NSBUFFERS];
    cufftHandle plan[NSTREAMS];
    cudaStream_t stream[NSTREAMS];
};
}
}

#endif //_GM_CUDA_FREQVIEW_