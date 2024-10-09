#include <iostream>
#include <cuda.h>
#include <thrust/complex.h>
#include "gm/cuda/FMDemodLong.h"

namespace gm {
namespace cuda {


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

FMDemodLong::FMDemodLong(cudaStream_t strm) : stream(strm), stream_set(true) { }
FMDemodLong::~FMDemodLong() {}
FMDemodLong::FMDemodLong() : stream_set(false) { }

void FMDemodLong::setOutput(std::complex<float>* out) {
    outData_d = (thrust::complex<float>*)out;
}

void FMDemodLong::setAnswer(float* out) {
    answer_d = out;
}

void FMDemodLong::setSquelch(float* out) {
    squelch_d = out;
}

void FMDemodLong::setSize(int sz, int tSize) {
    size = sz;
    tunerSize = tSize;
}

void FMDemodLong::makeDeEmphasis() {
    int numberTuners = size/tunerSize;
    cudaMalloc((void**)&deEmphasisData_d, sizeof(std::complex<float>) * (tunerSize / 2 + 1) * numberTuners);
    std::complex<float> *filter = (std::complex<float> *)calloc(sizeof(std::complex<float>), tunerSize / 2 + 1);
    std::complex<float> *filter_time = (std::complex<float> *)calloc(sizeof(std::complex<float>), tunerSize);
    std::complex<float> *filter_freq = (std::complex<float> *)calloc(sizeof(std::complex<float>), tunerSize);
    std::cout << "Tuner size is " << tunerSize << " Number " << numberTuners << std::endl;
    
    // filter goes until 2122 Hz then drops off, this is bin 89
    
    for (int cnt = 0; cnt < 170; cnt++) {
        filter_time[cnt] = std::complex<float>(1,0);
    }
    float drop = 1.0f;
    for (int cnt = 170; cnt < 180; cnt++) {
        filter_time[cnt] = std::complex<float>(1.0f/drop,0);
        drop += 0.25f;
    }
    // for (int cnt = 180; cnt < 257; cnt++) {
    //     filter_time[cnt] = std::complex<float>(1.0f/drop,0);
    //     drop += 10.0f;
    // }
    
    for (int cnt = 0; cnt < tunerSize/2; cnt++) {
        filter_time[tunerSize - cnt - 1] = filter_time[cnt];
    }
    
    int N = tunerSize;
    
    for (int k = 0; k < N; k++) {
        std::complex<float> sum(0,0);
        for (int n = 0; n < N; n++) {
            std::complex<float> expj(cos(-2.0f*M_PI*n*k/(float)N), sin(-2.0f*M_PI*n*k/(float)N));
            sum = sum + expj * filter_time[n];
        }
        filter_freq[k] = sum;
    }
    
    for(int cnt = tunerSize/4; cnt < 3*tunerSize/4; cnt++) {
        filter_freq[cnt] = std::complex<float>(0,0);
    }

    for (int k = 0; k < N/2; k++) {
        std::complex<float> sum(0,0);
        for (int n = 0; n < N; n++) {
            std::complex<float> expj(cos(2.0f*M_PI*n*k/(float)N), sin(2.0f*M_PI*n*k/(float)N));
            sum = sum + expj * filter_freq[n];
        }
        filter[k] = sum / 512.0f;
        //filter[k] = filter_time[k];
        std::cout << "Eq pt " << k << " : " << filter[k] << std::endl;
    }
    

    for (int cnt = 0; cnt < numberTuners; cnt++) {
        cudaMemcpy(&deEmphasisData_d[cnt * (tunerSize/2 + 1)], filter, sizeof(std::complex<float>) * (tunerSize/2 + 1), cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    free(filter);
    free(filter_freq);
    free(filter_time);
}

void FMDemodLong::makeFilter() {
    int numberTuners = (size / 4) / tunerSize; //Overlap has not been aligned yet
    cudaMalloc((void**)&filterData_d, sizeof(std::complex<float>) * tunerSize * numberTuners);
    std::complex<float> *filter = (std::complex<float> *)calloc(sizeof(std::complex<float>), tunerSize);
    std::complex<float> *filter_freq = (std::complex<float> *)calloc(sizeof(std::complex<float>), tunerSize);
    std::cout << "Tuner size is " << tunerSize << " Number " << numberTuners << std::endl;
    for (int cnt = 64; cnt < tunerSize-64; cnt++) {
        filter[cnt] = std::complex<float>(1,0);
    }
    
    int N = tunerSize;
    for (int k = 0; k < N; k++) {
        std::complex<float> sum(0,0);
        for (int n = 0; n < N; n++) {
            std::complex<float> expj(cos(-2.0f*M_PI*n*k/(float)N), sin(-2.0f*M_PI*n*k/(float)N));
            sum = sum + expj * filter[n];
        }
        filter_freq[k] = sum;
    }
    
    for(int cnt = tunerSize/4; cnt < 3*tunerSize/4; cnt++) {
        filter_freq[cnt] = std::complex<float>(0,0);
    }

    for (int k = 0; k < N; k++) {
        std::complex<float> sum(0,0);
        for (int n = 0; n < N; n++) {
            std::complex<float> expj(cos(2.0f*M_PI*n*k/(float)N), sin(2.0f*M_PI*n*k/(float)N));
            sum = sum + expj * filter_freq[n];
        }
        filter[k] = sum / 512.0f;
        //filter[k] = filter_time[k];
        std::cout << "Filter pt " << k << " : " << filter[k] << std::endl;
    }
    
    for (int cnt = 0; cnt < numberTuners; cnt++) {
        cudaMemcpy(&filterData_d[cnt * tunerSize], filter, sizeof(std::complex<float>) * tunerSize, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    free(filter);
    free(filter_freq);
}

void FMDemodLong::polarDescriminator() {
    if (stream_set) {
        polarDescKernel <<< 32, 256, 0, stream >>>(outData_d, answer_d, size);
    } else {
        polarDescKernel <<< 32, 256 >>>(outData_d, answer_d, size);
    }
}

int FMDemodLong::doDemod(cufftHandle plan, cufftComplex *src, cufftComplex *dst) {
    doFilter((std::complex<float>*) src, (std::complex<float>*) dst);
    cufftResult_t rval = cufftExecC2C(plan, dst, dst, CUFFT_INVERSE);
    if (rval) {
        printf("Error in fft\n");
    }
    return (int)rval;
}

// beware that the numbers are hard coded here
void FMDemodLong::doSquelch() {
    if (stream_set) {
        //stddevKernel <<< 8192, 256, 0, stream >>>(answer_d, squelch_d);
        squelchKernel <<< 8192, 512, 0, stream >>>(outData_d, squelch_d);
    } else {
        //stddevKernel <<< 8192, 256 >>>(answer_d, squelch_d);
        squelchKernel <<< 8192, 512 >>>(outData_d, squelch_d);
    }
}


void FMDemodLong::deEmphasis() {
    int numberTuners = size / tunerSize;
    if (stream_set) {
        deEmphasisKernel <<< 32, 256, 0, stream >>>((thrust::complex<float>*)answer_d, deEmphasisData_d, (size + numberTuners)/2);
    } else {
        deEmphasisKernel <<< 32, 256 >>>((thrust::complex<float>*)answer_d, deEmphasisData_d, (size + numberTuners)/2);
    }
}

void FMDemodLong::doFilter(std::complex<float>* in_data, std::complex<float>* out_data) {
    if (stream_set) {
        doFilterKernel <<< 32, 256, 0, stream >>>((thrust::complex<float>*)in_data, (thrust::complex<float>*)out_data, filterData_d, size/4);
    } else {
        doFilterKernel <<< 32, 256 >>>((thrust::complex<float>*)in_data, (thrust::complex<float>*)out_data, filterData_d, size/4);
    }
}

}
}

