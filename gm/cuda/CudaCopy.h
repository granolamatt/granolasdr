#ifndef _GM_CUDA_CUDACOPY_
#define _GM_CUDA_CUDACOPY_

#include <cuda.h>
#include <thrust/complex.h>
#include <complex>
#include "gm/cuda/HostCuda.h"

namespace gm {
namespace cuda {

class CudaCopy {
public:
    CudaCopy();
    CudaCopy(cudaStream_t strm);
    ~CudaCopy();
    void copyKernel();
    void copyKernel(thrust::complex<float>* oData_d, thrust::complex<short>* iData_d, int data_size);
    void copyKernel(float* oData_d, short* iData_d, int data_size);
    void setInput(std::complex<short>* in);
    void setOutput(std::complex<float>* out);
    void setAve(float* ave);
    void setSize(int sz);
    void averageKernel();
    void averageKernel(thrust::complex<float>* oData_d, float* aData_d);
private:
    cudaStream_t stream;
    bool stream_set;
    thrust::complex<short>* inData_d;
    thrust::complex<float>* outData_d;
    float* aveData_d;
    int size;
};
}
}


#endif //_GM_CUDA_CUDACOPY_