#ifndef _GM_CUDA_FMDEMODLONG_H_
#define _GM_CUDA_FMDEMODLONG_H_

#include <cuda.h>
#include <cufft.h>


namespace gm {
namespace cuda {

class FMDemodLong {
public:
    FMDemodLong();
    FMDemodLong(cudaStream_t strm);
    ~FMDemodLong();
    void polarDescriminator();
    void deEmphasis();
    void doFilter(std::complex<float>* in_data, std::complex<float>* out_data);
    void doSquelch();
    void setOutput(std::complex<float>* out);
    void setAnswer(float* out);
    void setSquelch(float* out);
    void setSize(int sz, int tunerSize);
    void makeFilter();
    void makeDeEmphasis();
    int doDemod(cufftHandle plan, cufftComplex *src, cufftComplex *dst);
private:
    cudaStream_t stream;
    bool stream_set;
    thrust::complex<float>* outData_d;
    thrust::complex<float>* filterData_d;
    thrust::complex<float>* deEmphasisData_d;
    float* answer_d;
    float* squelch_d;
    int size;
    int tunerSize;
};
}
}


#endif //_GM_CUDA_FMDEMODLONG_H_