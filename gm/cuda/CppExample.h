#ifndef _GM_CUDA_CPPEXAMPLE_
#define _GM_CUDA_CPPEXAMPLE_

#include <cuda.h>
#include <cufft.h>
#include <complex.h>
#include <thrust/complex.h>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/HostCuda.h"

namespace gm {
namespace cuda {
class CppExample : public gm::Runnable {
public:
    const static int NLARGE = 262144;
    const static int NEPOCH = NLARGE/2;
    const static int NSBUFFERS = 32;
    const static int NSTREAMS = 4;
    __host__ void run();
    __device__ CppExample(thrust::complex<short>* inData_d, thrust::complex<float>* fftData_d);
    __host__ CppExample(gm::buffer::BufferPosition<std::complex<short>>* pos);
    __device__ __host__ ~CppExample();
    __device__ __host__ void copyKernel();
    __host__ void processSamples();
private:
    thrust::complex<float>* fftData_d;
    thrust::complex<short>* inData_d;
    int dmaSize;
    bool running;
    thrust::complex<short>* inData;
#ifdef __CUDA_ARCH__
#else //__CUDA_ARCH__
    long epoch;
    int streamNum;
    int bufferNum;
    CppExample** myref[NSBUFFERS];
    gm::buffer::BufferPosition<std::complex<short>>* bPos;
    cufftHandle plan[NSTREAMS];
    cudaStream_t stream[NSTREAMS];
#endif //__CUDA_ARCH__
};
}
}


#endif //_GM_CUDA_CPPEXAMPLE_