#ifndef _GM_CUDA_HOSTCUDA_
#define _GM_CUDA_HOSTCUDA_

#include <cuda.h>
#include <complex>
#include <thrust/complex.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

#define cuda_check_error( err ) gm::cuda::throw_on_cuda_error(err, __FILE__, __LINE__);

namespace gm {
namespace cuda {

void throw_on_cuda_error(cudaError_t code, const char *file, int line);

namespace device {

class HostCuda {
public:
    HostCuda();
    HostCuda(cudaStream_t strm);
    ~HostCuda();
    void copyKernel(thrust::complex<float>* oData_d, thrust::complex<short>* iData_d, int data_size);
    void copyKernel(float* oData_d, short* iData_d, int data_size);
    void magKernel(std::complex<float>* data_d, uint8_t* mag_d, size_t data_size);
    void freqShift(std::complex<float>* input, std::complex<float>* output, int N, float shift_hz);
    void averageKernel(thrust::complex<float>* oData_d, float* aData_d);
    void makePixesKernel(thrust::complex<float>* oData_d, char* pixel_d);
private:
    cudaStream_t stream;
    bool stream_set;
};
}

}
}


#endif //_GM_CUDA_HOSTCUDA_
