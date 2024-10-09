#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <complex>
#include <cmath>
#include <thrust/complex.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/HostCuda.h"
#include "gm/cuda/CppExample.h"

namespace gm {
namespace cuda {

__global__ void createCppExample(CppExample** mycuda, thrust::complex<short>* dma_d, thrust::complex<float>* fft_d) {
    *mycuda = new gm::cuda::CppExample(dma_d, fft_d);
}

__global__ void deleteCppExample(gm::cuda::CppExample** mycuda) {
    delete *mycuda;
}

__global__ void copyKernel_d(gm::cuda::CppExample** mycuda) {
    (*mycuda)->copyKernel();
}

__host__ CppExample::CppExample(gm::buffer::BufferPosition<std::complex<short>>* pos)
    : inData((thrust::complex<short>*)pos->getBuffer()),
      dmaSize(pos->getBufferSize()) {
#ifdef  __CUDA_ARCH__
#else //__CUDA_ARCH__
    bPos = pos;
    try {
        cuda_check_error(cudaHostRegister(inData, dmaSize, 0));
        cuda_check_error(cudaMalloc(&inData_d, sizeof(thrust::complex<short>) * NLARGE * NSTREAMS));
        cuda_check_error(cudaMalloc(&fftData_d, sizeof(thrust::complex<float>) * NLARGE * NSBUFFERS));
        for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
            int in_pos = (cnt % NSTREAMS) * NLARGE;
            int out_pos = (cnt % NSBUFFERS) * NLARGE;
            cuda_check_error(cudaMalloc(&myref[cnt], sizeof(gm::cuda::CppExample**)));
            gm::cuda::createCppExample <<< 1, 1>>>(myref[cnt], &inData_d[in_pos], &fftData_d[out_pos]);
            cuda_check_error(cudaPeekAtLastError());
            cuda_check_error(cudaDeviceSynchronize());
        }

        for (int cnt = 0; cnt < NSTREAMS; cnt++) {
            cuda_check_error(cudaStreamCreate(&stream[cnt]));
            cufftResult fftRes = cufftPlan1d(&plan[cnt], NLARGE, CUFFT_C2C, 1);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftSetStream(plan[cnt], stream[cnt]);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
        }

    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
#endif //__CUDA_ARCH__
}

__device__ CppExample::CppExample(thrust::complex<short>* dma_d, thrust::complex<float>* fft_d) :
    inData_d(dma_d), fftData_d(fft_d)
{
}

__device__ __host__ CppExample::~CppExample() {
#ifdef  __CUDA_ARCH__
#else //__CUDA_ARCH__
    for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
        gm::cuda::deleteCppExample <<< 1, 1>>>(myref[cnt]);
        cuda_check_error(cudaPeekAtLastError());
        cuda_check_error(cudaFree(myref[cnt]));
        myref[cnt] = NULL;
    }
    cuda_check_error(cudaFree(inData_d));
    cuda_check_error(cudaFree(fftData_d));
#endif //__CUDA_ARCH__
}

/**
 *  This function copies data from the icepic into the device memory used for fft
 *  We copy two samples at a time for coelescence and so that we can invert the
 *  rx spectrum from the icepic.  This puts freq 0 from the tuner at FFTSIZE / 2
 *  bin allowing us to never have to deal with the ends of the fft data.
 */
__device__ __host__ void CppExample::copyKernel()
{

#ifdef  __CUDA_ARCH__
    const int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x); // this thread
    const int numThreads = 2 * (blockDim.x * gridDim.x);
    const int end = CppExample::NLARGE / 2;

    for (int cnt = idx; cnt < end; cnt += numThreads)
    {
        thrust::complex<short> sig1 = inData_d[cnt];
        thrust::complex<short> sig2 = inData_d[cnt + 1];

        thrust::complex<float> c1(((float) sig1.real()) / 16384.0f,
                                  ((float) sig1.imag()) / 16384.0f);
        thrust::complex<float> c2(((float) - sig2.real()) / 16384.0f,
                                  ((float) - sig2.imag()) / 16384.0f);
        fftData_d[cnt] = c1;
        fftData_d[cnt + 1] = c2;
    }
    __syncthreads();
#else //__CUDA_ARCH__
    gm::cuda::copyKernel_d<<< 32, 256, 0, stream[streamNum]>>>(myref[bufferNum]);
#endif //__CUDA_ARCH__
}

__host__ void CppExample::processSamples() {

}

__host__ void CppExample::run() {
#ifndef __CUDA_ARCH__
    try
    {
        running = true;
        printf("Starting host cuda runnable\n");
        cuda_check_error(cudaSetDevice(0));
        //		cuda_check_error(cudaDeviceSetLimit(cudaLimitMallocHeapSize,20000000*sizeof(thrust::complex<float>)));

        long position = bPos->getNow();
        epoch = position / NEPOCH + 1;
        while(running) {
            printf("On epoch %ld\n", epoch);
            position = epoch * NEPOCH;
            const int length = bPos->getElementSize() * NEPOCH;
            bPos->getPosition(position);
            int dma_position = (int)(position % dmaSize);
            streamNum = epoch % NSTREAMS;
            bufferNum = epoch % NSBUFFERS;
            int lastStreamNum = streamNum - 1;
            if (lastStreamNum < 0) {
                lastStreamNum += NSTREAMS;
            }
            cudaStream_t myStream = stream[streamNum];
            cufftHandle myPlan = plan[streamNum];
            thrust::complex<short>* data_h = &inData[dma_position];
            thrust::complex<short>* data_d = &inData_d[streamNum*NLARGE];
            thrust::complex<short>* last_data_d = &inData_d[lastStreamNum*NLARGE];
            thrust::complex<float>* fft_d = &fftData_d[bufferNum*NLARGE];

            const int first = dmaSize - dma_position;
            if (first >= NEPOCH) {
                cudaMemcpyAsync(&data_d[NEPOCH], data_h, length, cudaMemcpyHostToDevice, myStream);
            } else {
                cudaMemcpyAsync(&data_d[NEPOCH], data_h, first * bPos->getElementSize(), cudaMemcpyHostToDevice, myStream);
                cudaMemcpyAsync(&data_d[NEPOCH + first], inData, length - first * bPos->getElementSize(), cudaMemcpyHostToDevice, myStream);
            }
            cudaMemcpyAsync(data_d, &last_data_d[NEPOCH], length, cudaMemcpyDeviceToDevice, myStream);
            copyKernel();
            cuda_check_error(cudaPeekAtLastError());
            cufftResult_t rval = cufftExecC2C(myPlan, (cufftComplex *)fft_d, (cufftComplex *) fft_d, CUFFT_FORWARD);
            if (rval) {
                printf("Error in fft\n");
            }
            epoch++;
            //cudaStreamAddCallback(*myStream, MyCallback, (void *)storage->ice_result[streamNum], 0);
        }
    }
    catch (thrust::system_error &e)
    {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
#endif //__CUDA_ARCH__
}

}
}


