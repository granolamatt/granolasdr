#include <thrust/complex.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

#include "gm/cuda/HostCuda.h"

namespace gm {
namespace cuda {

// Everything now is final HF 1048576 samples at 4.172325 Hz per bin
// So 600 bins for 2.5 khz 1200 if we want to filter it by averaging
// Do we do 2048 then so we can make it easy?  Also if we change the 
// sample rate to be slower we have to change these definitions
// they are only hard coded for the shared memory in the kernels
// in the average kernel so let's try and eliminate that kernel

#define NLARGE (1048576)
#define NSMALL 1024

void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::stringstream ss;
        ss << file << "(" << line << ")";
        std::string file_and_line;
        ss >> file_and_line;
        throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
    }
}

namespace device {
/**
*  We copy two samples at a time for coelescence and so that we can invert the
*  rx spectrum.  This puts freq 0 from the tuner at FFTSIZE / 2
*  bin allowing us to never have to deal with the ends of the fft data.
*/
__global__ void copyKernelComplexShort(thrust::complex<short>* inData_d, thrust::complex<float>* outData_d, int size)
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

__global__ void copyKernelShort(short* inData_d, float* outData_d, int size)
{
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x); // this thread
    const int numThreads = (blockDim.x * gridDim.x);
    const int end = size;

    for (int cnt = idx; cnt < end; cnt += numThreads)
    {
        short sig = inData_d[cnt];

        float c = (((float) sig) / 8192.0f);
        outData_d[cnt] = c;
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

__global__ void polarDescKernel(thrust::complex<float>* inData_d, float* outData_d, int size) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x); // this thread
    const int numThreads = (blockDim.x * gridDim.x);
    const int end = size;

    for (int cnt = idx; cnt < end; cnt += numThreads)
    {
        thrust::complex<float> sig1 = inData_d[cnt];
        thrust::complex<float> sig2 = -conj(inData_d[cnt + 1]);
        
        thrust::complex<float> mult = sig1*sig2;
        float ang = atan2(mult.imag(), mult.real());
        __syncthreads();
        outData_d[cnt] = ang;
    }
    __syncthreads();
}

__global__ void deEmphasisKernel(thrust::complex<float>* inData_d, thrust::complex<float>* deEmphasisData_d, int size) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x); // this thread
    const int numThreads = (blockDim.x * gridDim.x);
    const int end = size;

    for (int cnt = idx; cnt < end; cnt += numThreads)
    {
        thrust::complex<float> in = inData_d[cnt] * deEmphasisData_d[cnt];
        inData_d[cnt] = in;
    }
}

__global__ void doFilterKernel(thrust::complex<float>* inData_d, thrust::complex<float>* outData_d, thrust::complex<float>* filterData_d, int size) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x); // this thread
    const int numThreads = (blockDim.x * gridDim.x);
    const int end = size;

    for (int cnt = idx; cnt < end; cnt += numThreads)
    {
        thrust::complex<float> in = inData_d[cnt] * filterData_d[cnt];
        outData_d[cnt] = in;
    }
}

// The size of the frequency is 256+1, ignore last sample and give offset with 256 threads
__global__ void stddevKernel(float* inData_d, float* squelch) {
    const int start = blockIdx.x * (512) + 128;
// Reduction (min/max/avr/sum), valid only when blockDim.x is a power of two:
    int  thread2;
    __shared__ float sum[256];
    __shared__ float stddev[256];
    
    sum[threadIdx.x] = inData_d[threadIdx.x + start];
    stddev[threadIdx.x] = sum[threadIdx.x];

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
        
    // if (threadIdx.x == 0) {
    //     squelch[blockIdx.x] = sum[0] / 256;
    // }
    __syncthreads();
    float ave = sum[0] / 256.0f;
    stddev[threadIdx.x] = stddev[threadIdx.x] - ave;
    stddev[threadIdx.x] = stddev[threadIdx.x] * stddev[threadIdx.x];
    nTotalThreads = blockDim.x; // Total number of active threads
    __syncthreads();
    while (nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1); // divide by two
        // only the first half of the threads will be active.
        if (threadIdx.x < halfPoint)
        {
            thread2 = threadIdx.x + halfPoint;

            stddev[threadIdx.x] += stddev[thread2];

        }
        __syncthreads();

        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }
        
    if (threadIdx.x == 0) {
        squelch[blockIdx.x] = stddev[0];
    }
    __syncthreads();
    
}


// The size of the frequency is 256+1, ignore last sample and give offset with 256 threads
__global__ void squelchKernel(thrust::complex<float>* inData_d, float* squelch) {
    const int start = blockIdx.x * (512);
// Reduction (min/max/avr/sum), valid only when blockDim.x is a power of two:
    int  thread2;
    __shared__ float sum[512];
    
    // sum[threadIdx.x] = inData_d[threadIdx.x + start].real() * inData_d[threadIdx.x + start].real() 
    //     + inData_d[threadIdx.x + start].imag() * inData_d[threadIdx.x + start].imag();
    
    //thrust::complex<float> val = threadIdx.x & 1 ? inData_d[threadIdx.x + start] : -inData_d[threadIdx.x + start];
    float val = threadIdx.x & 1 ? abs(inData_d[threadIdx.x + start]) : 0;
    
    sum[threadIdx.x] = val;

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
        squelch[blockIdx.x] = sum[0] / 256;
    }
    __syncthreads();
}

HostCuda::HostCuda(cudaStream_t strm) : stream(strm), stream_set(true) {}
HostCuda::~HostCuda() {}
HostCuda::HostCuda() : stream_set(false) {}

void HostCuda::copyKernel(thrust::complex<float>* oData_d, thrust::complex<short>* iData_d, int data_size) {
    if (stream_set) {
        copyKernelComplexShort <<< 32, 256, 0, stream >>>(iData_d, oData_d, data_size);
    } else {
        copyKernelComplexShort <<< 32, 256 >>>(iData_d, oData_d, data_size);
    }
}

void HostCuda::copyKernel(float* oData_d, short* iData_d, int data_size) {
    if (stream_set) {
        copyKernelShort <<< 32, 256, 0, stream >>>(iData_d, oData_d, data_size);
    } else {
        copyKernelShort <<< 32, 256 >>>(iData_d, oData_d, data_size);
    }
}

void HostCuda::averageKernel(thrust::complex<float>* oData_d, float* aData_d) {
    if (stream_set) {
        averageKernelWork <<< NSMALL, NLARGE/NSMALL, 0, stream >>>(oData_d, aData_d);
    } else {
        averageKernelWork <<< NSMALL, NLARGE/NSMALL >>>(oData_d, aData_d);
    }
}

void HostCuda::makePixesKernel(thrust::complex<float>* oData_d, char* pixel_d) {
    // if (stream_set) {
    //     averageKernelWork <<< NSMALL, NLARGE/NSMALL, 0, stream >>>(oData_d, aData_d);
    // } else {
    //     averageKernelWork <<< NSMALL, NLARGE/NSMALL >>>(oData_d, aData_d);
    // }
}

}

}
}


