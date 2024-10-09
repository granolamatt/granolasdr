#include <cuda.h>
#include <cufft.h>
#include <thrust/complex.h>
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"
#include "gm/cuda/FreqScan.h"

namespace gm {
namespace cuda {
FreqScan::FreqScan(gm::buffer::BufferPosition<std::complex<short>>* inP) :
    inPos(inP),
    inData((thrust::complex<short>*)inP->getBuffer()),
    dmaSize(inP->getBufferSize()),
    outPos()
{

    long now = inPos->getNow();
    //epoch = now / (NLARGE / 2);

    float* outData = (float*)calloc(NSMALL * NSCANS, sizeof(float));
    outPos.setBuffer(outData, NLARGE * NSBUFFERS);
    std::cout << "Memory Made" << std::endl;
    try {
        cuda_check_error(cudaSetDevice(0));
        //cuda_check_error(cudaHostRegister(inData, inP->getElementSize()*dmaSize, 0));
        cuda_check_error(cudaMalloc((void**)&inData_d, sizeof(thrust::complex<short>) * NLARGE * NSTREAMS));
        cuda_check_error(cudaMalloc((void**)&fftData_d, sizeof(thrust::complex<float>) * NLARGE * NSBUFFERS));
        cuda_check_error(cudaMalloc((void**)&aveData_d, sizeof(float) * NSMALL * NSBUFFERS));


        //cuda_check_error(cudaHostRegister(outData, outPos.getByteSize(), 0));

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
        for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
            int in_pos = (cnt % NSTREAMS) * NLARGE;
            int out_pos = cnt * NLARGE;
            cudaCopy[cnt] = new CudaCopy(stream[(cnt % NSTREAMS)]);
            cudaCopy[cnt]->setInput((std::complex<short>*)&inData_d[in_pos]);
            cudaCopy[cnt]->setOutput((std::complex<float>*)&fftData_d[out_pos]);
            cudaCopy[cnt]->setAve(&aveData_d[cnt * NSMALL]);
            cudaCopy[cnt]->setSize(NLARGE);
        }
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
}


FreqScan::~FreqScan() {
    for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
        delete(cudaCopy[cnt]);
    }
    cudaFree((void*)inData_d);
    cudaFree((void*)fftData_d);
    cudaFree((void*)aveData_d);
    //XXX Need to clean up streams and cufft
}

long FreqScan::capture(long epoch) {
    try
    {
        long position = inPos->getNow();
        // Might be in the middle of a transfer, get the next block
        inPos->getPosition(position + 100L);
        position = inPos->getNow();
        //long position = epoch * NLARGE / 2;
        inPos->getPosition(position + NLARGE);
        // What if there are two??
        static gm::buffer::BufferPosition<float>* bpos = &outPos;
        //printf("On epoch %ld\n", epoch);
        const int length = inPos->getElementSize() * NLARGE;
        int dma_position = (int)(position % dmaSize);
        streamNum = epoch % NSTREAMS;
        bufferNum = epoch % NSBUFFERS;
        captureNum = epoch % NSCANS;
        cudaStream_t myStream = stream[streamNum];
        cufftHandle myPlan = plan[streamNum];
        thrust::complex<short>* data_h = &inData[dma_position];
        thrust::complex<short>* data_d = &inData_d[streamNum * NLARGE];
        thrust::complex<float>* fft_d = &fftData_d[bufferNum * NLARGE];
        float* ave_d = &aveData_d[bufferNum * NSMALL];

        const int first = dmaSize - dma_position;
        if (first >= NLARGE) {
            //printf("We are good %ld\n", position);
            cudaMemcpy(data_d, data_h, length, cudaMemcpyHostToDevice);
        } else {
            //printf("Check it now\n");
            cudaMemcpy(data_d, data_h, first * inPos->getElementSize(), cudaMemcpyHostToDevice);
            cudaMemcpy(&data_d[first], inData, length - first * inPos->getElementSize(), cudaMemcpyHostToDevice);
        }
        cudaCopy[bufferNum]->copyKernel();
        cuda_check_error(cudaPeekAtLastError());
        cufftResult_t rval = cufftExecC2C(myPlan, (cufftComplex *)fft_d, (cufftComplex *) fft_d, CUFFT_FORWARD);
        if (rval) {
            printf("Error in fft\n");
        }
        cudaCopy[bufferNum]->averageKernel();
        //if (!bufferNum) printf("length is %d\n",length);
        cudaMemcpy(&outPos.getBuffer()[captureNum * NSMALL], ave_d, NSMALL*sizeof(float), cudaMemcpyDeviceToHost);
        //cudaStreamAddCallback(myStream, MyCallback, (void *)&outPos, 0);
        cudaStreamAddCallback(myStream,
        [](cudaStream_t stream, cudaError_t status, void *data) {
            bpos->setPosition((long)data);
            //long position = bpos->getNow();
            //printf("callback now %ld\n", position);
        }
        , (void *)epoch, 0);
        //epoch++;
    }
    catch (thrust::system_error &e)
    {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
    return 0;
}

}
}


