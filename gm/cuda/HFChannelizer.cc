#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"

namespace gm {
namespace cuda {

HFChannelizer::HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP) :
inPos(inP), 
inData_d(NULL), 
fftInData_d(NULL), 
fftData_d(NULL),
channelData_d(NULL),
demodData_d(NULL),
pixel_d(NULL) {
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
        bins = getBins();
        fft_length = 0;
        for(const std::vector<uint32_t>& b : bins) {
            fft_length += b[2];
        }
        // now make length power of 2
        uint32_t binsize = 1024;
        while(binsize < fft_length) {
            binsize *= 2;
        }
        fft_length = binsize;
        cuda_check_error(cudaMalloc((void**)&channelData_d, fft_length*sizeof(std::complex<float>) + 1024));
        printf("Total fft length is %u\n", fft_length);

        /**
         *       | band 1 | band 2 | band 3 | ... band 10|
         * 1st   |2100*2|2100*2| .....              end|
         * 2nd   |  2100*2|2100*2| .....              end|
         * .
         * .
         * 8th   |  2100*2|2100*2| .....              end|
         **/

        uint32_t baseband_bins = (uint32_t)(4200.0/(1e6/freqsperbin));
        binsize = 256;
        while (binsize < baseband_bins) {
            binsize *= 2;
        }
        nTune = binsize;
        nChannels = fft_length / nTune; // -1 because the last channel will be sub divided
        printf("Making batch fft with %u bins and %u channels\n", nTune, nChannels);

        // cuda_check_error(cudaMalloc((void**)&demodData_d, 8*(fft_length*sizeof(std::complex<float>) + nTune)));
        // cuda_check_error(cudaMalloc((void**)&pixel_d, 1024*128));

        // Now for the sub channels
        fftRes = cufftPlan1d(&iplan, nTune, CUFFT_C2C, nChannels);
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
    // if (demodData_d) cudaFree(demodData_d);
    // if (pixel_d) cudaFree(pixel_d);
    cufftDestroy(plan);
    cufftDestroy(iplan);
    cudaStreamDestroy(stream);
}

std::vector<std::vector<uint32_t>> HFChannelizer::getBins() {
        double srate = (double)gm::rx888::rx888::rx_samplerate / 2e6;
        freqsperbin = (double)gm::rx888::rx888::NLARGE * 2.0 / srate;
        // Now we need the HF bands
        // 160 Meters 1.8 - 2.0 MHz
        // 80 Meters 3.5 - 4.0 MHz
        // 60 Meters USB 5.3305 - 5.4355 channels 5.332 5.348 5.3585 5.373 5.405
        // 40 Meters 7.0 - 7.3 MHz
        // 30 Meters 10.1 - 10.15 MHz
        // 20 Meters 14.0 - 14.35 MHz
        // 17 Meters 18.068 - 18.168 MHz
        // 15 Meters 21.0 - 21.45 MHz
        // 12 Meters 24.89 - 24.99 MHz
        // 10 Meters 28.0 - 29.7 MHz
        const std::vector<std::vector<double>> frequencies = {
            {1.8,2.0},
            {3.5,4.0},
            {5.3305, 5.4355},
            {7.0,7.3},
            {10.1,10.15},
            {14.0,14.35},
            {18.068, 18.168},
            {21.0, 21.45},
            {24.89, 24.99},
            {28,29.7}
        };
        std::vector<std::vector<uint32_t>> ret;
        for(const std::vector<double>& f : frequencies) {
            uint32_t start = (int)(f[0] * freqsperbin);
            uint32_t stop = (int)(f[1] * freqsperbin);
            if (start % 4) {
                start = (start + 4) - (start % 4);
            }
            if (stop % 4) {
                stop = (stop + 4) - (stop % 4);
            }
            uint32_t length = stop - start;
            ret.push_back({start, stop, length});
        }
        return ret;
}

int HFChannelizer::doCopy(uint64_t now) {
    try {
        size_t length = inShape[1];
        int in_position = (now % inShape[0]) * length;
        // first copy the data into the device
        cuda_check_error(cudaMemcpyAsync(&inData_d[in_position], &inData[in_position], 
            length*inPos->getElementSize(), cudaMemcpyHostToDevice, stream));
        // now cast it to floats and scale it
        cuda_h.copyKernel(&fftInData_d[in_position + gm::rx888::rx888::NLARGE], &inData_d[in_position], length);
        // if we are on first buffer then do the wrap
        if (!(now % gm::rx888::rx888::BUFFERS)) {
            cudaMemcpyAsync(&fftInData_d, 
                            &fftInData_d[in_position + gm::rx888::rx888::NLARGE], 
                            inShape[1]*sizeof(float),cudaMemcpyDeviceToDevice, stream);
        }
        cufftResult_t rval = cufftExecR2C(plan, &fftInData_d[in_position], (cufftComplex *) fftData_d);
        if (rval) {
            printf("Error in fft\n");
            return 0;
        }
        uint32_t offset = 0;
        cuda_check_error(cudaMemsetAsync(fftData_d, 0, fft_length*sizeof(std::complex<float>), stream));
        // Does the USB really belong here??  It is really
        // the same just inverted spectrum
        // copy it in backwards maybe
        for(const std::vector<uint32_t>& b : bins) {
            cuda_check_error(cudaMemcpyAsync(&channelData_d[offset], 
                &fftData_d[b[0]], 
                b[2]*sizeof(float),cudaMemcpyDeviceToDevice, stream));
            offset += b[2];
        }
        // printf("Copied out %u total size %u freqsperbin %f\n", offset, fft_length, 1e6/freqsperbin);
        // for (int cnt = 0; cnt < 8; cnt++) {
        //     cufftResult_t rval = cufftExecC2C(iplan, (cufftComplex *)&channelData_d[cnt*nTune/16],
        //          (cufftComplex *)&demodData_d[fft_length*cnt], CUFFT_INVERSE);
        //     if (rval) {
        //         printf("Error in fft\n");
        //         return 0;
        //     }
        // }
        // printf("Finished processing %d x8 samples for %d tuners\n", fft_length, nTune);
        // at 1048576 for all hf with 1024 samples oversampled by 8
        // cuda_h.averageKernel((cufftComplex *)demodData_d, char* pixel_d);

        // Now need to conjugate the USB then filter
        // Then we are ready to look for channels

        // Can probably make a video of the channels too

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
        printf("Got epoch %d\n", next);
        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cout << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
                now = next;
                break;
            }
            printf("Going copy on epoch %d\n", now);
            int numCopied = doCopy(now);
            if (!numCopied) exit(-200);
            //outPos.setPosition(now, 1);
	    now += numCopied;
        }
    }
}

}
}
