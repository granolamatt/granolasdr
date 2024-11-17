#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include <chrono>

#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"

#include "ft8_lib/common/monitor.h"

namespace gm {
namespace cuda {

HFChannelizer::HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP) :
inPos(inP),
inData_d(NULL),
fftInData_d(NULL), 
fftData_d(NULL),
demodData_d(NULL),
channelData_d(NULL),
num_blocks(0),
buffer_number(0) {
    inShape = inPos->getShape();
    inData = (int16_t*)inPos->getBuffer();

    try {
        //cuda_check_error(cudaHostAlloc((void**)&inData_d, inPos->getByteSize(), cudaHostAllocPortable));
        cuda_check_error(cudaMalloc((void**)&inData_d, inPos->getByteSize()));
        cuda_check_error(cudaMalloc((void**)&fftInData_d,
            sizeof(float)*(gm::rx888::rx888::NLARGE*gm::rx888::rx888::BUFFERS + gm::rx888::rx888::NLARGE)));
        cuda_check_error(cudaMalloc((void**)&fftData_d, 
            sizeof(std::complex<float>)*(2*gm::rx888::rx888::NLARGE + 1024))); // 1024 is pad for the extra data from realfft

        cuda_check_error(cudaSetDevice(0));
        cuda_check_error(cudaStreamCreate(&stream));
        cufftResult fftRes = cufftPlan1d(&plan, gm::rx888::rx888::NLARGE*2, CUFFT_R2C, 1);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        fftRes = cufftSetStream(plan, stream);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        cuda_h = gm::cuda::device::HostCuda(stream);
        // From calc_rf.py
        bins = {{13480,14980,1500},
                {26212,29960,3748},
                {39920,40712,792},
                {52424,54676,2252},
                {75644,76024,380},
                {104856,107480,2624},
                {135324,136076,752},
                {157284,160660,3376},
                {186420,187172,752},
                {209712,222448,12736},
            };
        fft_length = 32768;

        cuda_check_error(cudaMalloc((void**)&demodData_d, BUFFERS * fft_length / 2 * sizeof(std::complex<float>) + 1024));
        
        hfBufferPosition.setBuffer(demodData_d, {BUFFERS,fft_length / 2});

        cuda_check_error(cudaMalloc((void**)&channelData_d, fft_length*sizeof(std::complex<float>) + 1024));
        printf("Total fft length is %u\n", fft_length);

        // Now for the sub channels
        fftRes = cufftPlan1d(&iplan, fft_length, CUFFT_C2C, 1);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        fftRes = cufftSetStream(iplan, stream);
        if (fftRes) {
            printf("Error: exit for now\n");
        }

    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
}

HFChannelizer::~HFChannelizer() {
    if (inData_d) cudaFree(inData_d);
    if (fftInData_d) cudaFree(fftInData_d);
    if (fftData_d) cudaFree(fftData_d);
    if (channelData_d) cudaFree(channelData_d);
    if (demodData_d) cudaFree(demodData_d);
    cufftDestroy(plan);
    cufftDestroy(iplan);
    cudaStreamDestroy(stream);
}

int HFChannelizer::doCopy(uint64_t now) {
    try {
        static gm::buffer::BufferPosition<std::complex<float>>* bpos = &hfBufferPosition;
        size_t length = inShape[1];
        int in_position = (now % inShape[0]) * length;
        // first copy the data into the device
        cuda_check_error(cudaMemcpyAsync(&inData_d[in_position], &inData[in_position], 
            length*inPos->getElementSize(), cudaMemcpyHostToDevice, stream));
        // now cast it to floats and scale it
        cuda_h.copyKernel(&fftInData_d[in_position + length], &inData_d[in_position], length);
        // if we are on first buffer then do the wrap

        if (!(now % gm::rx888::rx888::BUFFERS)) {
            cudaMemcpyAsync(&fftInData_d, 
                            &fftInData_d[gm::rx888::rx888::BUFFERS*length], 
                            length*sizeof(float),cudaMemcpyDeviceToDevice, stream);
        }
        
        cufftResult_t rval = cufftExecR2C(plan, &fftInData_d[in_position], (cufftComplex *) fftData_d);
        if (rval) {
            printf("Error in fft\n");
            return 0;
        }
        
        uint32_t offset = 0;
        cuda_check_error(cudaMemsetAsync(channelData_d, 0, fft_length*sizeof(std::complex<float>), stream));
        // Does the USB really belong here??  It is really
        // the same just inverted spectrum
        // copy it in backwards maybe
        for(const std::vector<uint32_t>& b : bins) {
            cuda_check_error(cudaMemcpyAsync(&channelData_d[offset], 
                &fftData_d[b[0]],
                b[2]*sizeof(std::complex<float>),cudaMemcpyDeviceToDevice, stream));
            offset += b[2];
        }

        // now time data of all our freqs
        rval = cufftExecC2C(iplan, (cufftComplex *)&channelData_d[0],
            (cufftComplex *)&channelData_d[0], CUFFT_INVERSE);        
        if (rval) {
            printf("Error in fft\n");
            return 0;
        }
        
        // we are done so put it in a buffer so others can use it??
        // but it stays in cuda so reuse the stream and have events??
        // I guess for now lets do ft8 here to make sure we have a concept
        

        cuda_check_error(cudaMemcpyAsync(&demodData_d[(buffer_number % BUFFERS) * fft_length / 2],
            &channelData_d[fft_length/4],
            fft_length / 2 * sizeof(std::complex<float>),cudaMemcpyDeviceToDevice, stream));

        cudaStreamAddCallback(stream,
        [](cudaStream_t mstream, cudaError_t status, void *data) {
            uint64_t* pos = (uint64_t*)data;
            bpos->setPosition(*pos, 1);
            *pos += 1;
        }
        , (void *)&buffer_number, 0);
        
        return 1;
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaCopy: " << e.what() << std::endl;
    }
    return 0;
}

void HFChannelizer::run() {
    uint64_t now = inPos->getNow(1) + 1;
    
    while(isRunning()) {
        uint64_t next = inPos->getPosition(now+1, 1);
        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cout << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
                now = next;
                break;
            }
            int numCopied = doCopy(now);
            if (!numCopied) exit(-200);
            //outPos.setPosition(now, 1);
	        now += numCopied;
        }
    }
}

}
}
