#include <cuda.h>
#include <thrust/complex.h>
#include "gm/cuda/CudaCopy.h"

namespace gm {
namespace cuda {


#define NLARGE (1048576)
#define NSMALL 4096

/**
*  We copy two samples at a time for coelescence and so that we can invert the
*  rx spectrum.  This puts freq 0 from the tuner at FFTSIZE / 2
*  bin allowing us to never have to deal with the ends of the fft data.
*/
__global__ void copyKernelShort(thrust::complex<short>* inData_d, thrust::complex<float>* outData_d, int size)
{
    const int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // this thread
    const int numThreads = 2 * (blockDim.x * gridDim.x);
    const int end = size;

    for (int cnt = idx; cnt < end; cnt += numThreads)
    {
        thrust::complex<short> sig1 = inData_d[cnt];
        thrust::complex<short> sig2 = inData_d[cnt + 1];

        thrust::complex<float> c1(((float) sig1.real()) / 4096.0f,
                                  ((float) sig1.imag()) / 4096.0f);
        thrust::complex<float> c2((-(float) sig2.real()) / 4096.0f,
                                  (-(float) sig2.imag()) / 4096.0f);
        outData_d[cnt] = c1;
        outData_d[cnt + 1] = c2;
    }
    __syncthreads();
}
__global__ void averageKernelWork(thrust::complex<float>* outData_d, float* aveData_d) {
// Reduction (min/max/avr/sum), valid only when blockDim.x is a power of two:
    int  thread2;
    //const int BLOCK_SIZE = blockDim.x * gridDim.x;
    __shared__ float sum[NLARGE/NSMALL];
    //thrust::complex<float>* sum = (thrust::complex<float>*)&backing[0];
    int cnt = gridDim.x - blockIdx.x - 1;

//    for (int cnt = 0; cnt < NSMALL; cnt++) {
        int index = cnt;
        if (index == 2048) {
            index = 2047; //DC Middle
        }
        sum[threadIdx.x] = abs(outData_d[threadIdx.x + NLARGE/NSMALL*index]);

        __syncthreads();
        int nTotalThreads = blockDim.x; // Total number of active threads

        while (nTotalThreads > 1)
        {
            int halfPoint = (nTotalThreads >> 1); // divide by two
            // only the first half of the threads will be active.

            if (threadIdx.x < halfPoint)
            {
                thread2 = threadIdx.x + halfPoint;

                sum[threadIdx.x] += sum[thread2];

            }
            __syncthreads();

            // Reducing the binary tree size by two:
            nTotalThreads = halfPoint;
        }
        
        if (threadIdx.x == 0) {
            aveData_d[blockIdx.x] = sum[0] / (float)(NLARGE/NSMALL);
        }
        __syncthreads();
//    }

}

CudaCopy::CudaCopy(cudaStream_t strm) : stream(strm), stream_set(true) {}
CudaCopy::~CudaCopy() {}
CudaCopy::CudaCopy() : stream_set(false) {}
void CudaCopy::setInput(std::complex<short>* in) {
    inData_d = (thrust::complex<short>*)in;
}
void CudaCopy::setOutput(std::complex<float>* out) {
    outData_d = (thrust::complex<float>*)out;
}
void CudaCopy::setAve(float* ave) {
    aveData_d = ave;
}
void CudaCopy::setSize(int sz) {
    size = sz;
}

void CudaCopy::copyKernel() {
    if (stream_set) {
        copyKernelShort <<< 32, 256, 0, stream >>>(inData_d, outData_d, size);
    } else {
        copyKernelShort <<< 32, 256 >>>(inData_d, outData_d, size);
    }
}

void CudaCopy::averageKernel() {
    if (stream_set) {
        averageKernelWork <<< NSMALL, NLARGE/NSMALL, 0, stream >>>(outData_d, aveData_d);
    } else {
        averageKernelWork <<< NSMALL, NLARGE/NSMALL >>>(outData_d, aveData_d);
    }
}

}
}


