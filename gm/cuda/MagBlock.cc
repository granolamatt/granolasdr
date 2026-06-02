#include <cstdio>
#include <cstring>
#include <complex>
#include <cuda.h>
#include <cufft.h>
#include "gm/cuda/MagBlock.h"
#include "gm/cuda/HostCuda.h"
#include "gm/hf/ft8_capture.h"

namespace gm {
namespace cuda {

MagBlock::MagBlock(gm::buffer::BufferPosition<std::complex<float>>* inP, int zmq_port)
    : inPos_(inP), zmq_ctx_(1), zmq_pub_(zmq_ctx_, ZMQ_PUB)
{
    if (zmq_port > 0)
        zmq_pub_.connect("tcp://localhost:" + std::to_string(zmq_port));

    inShape_  = inPos_->getShape();
    inData_d_ = (std::complex<float>*)inPos_->getBuffer();

    cuda_check_error(cudaSetDevice(0));
    cuda_check_error(cudaStreamCreate(&stream_));
    cuda_check_error(cudaEventCreateWithFlags(&ring_.ready, cudaEventDisableTiming));

    cuda_h_      = gm::cuda::device::HostCuda(stream_);
    rfft_length_ = 1048576;  // 2^20; 6553600 Hz / 6.25 Hz/bin

    // demodData_d: double-buffer for overlap-save RFFT
    cuda_check_error(cudaMalloc((void**)&demodData_d_,
        4 * rfft_length_ * sizeof(std::complex<float>) + 1024));

    // demodFT8_d: FT8_TIME_OSR × FT8_FREQ_OSR sub-band FFT outputs
    cuda_check_error(cudaMalloc((void**)&demodFT8_d_,
        (size_t)FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length_ * sizeof(std::complex<float>) + 1024));

    // magFT8_d: magnitude (uint8) output before ring write
    cuda_check_error(cudaMalloc((void**)&magFT8_d_,
        (size_t)FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length_ * sizeof(uint8_t) + 1024));

    // demodShift_d: frequency-shift staging buffer
    cuda_check_error(cudaMalloc((void**)&demodShift_d_,
        rfft_length_ * sizeof(std::complex<float>) + 1024));

    // Device-side mag ring: RING_BLOCKS slots × slot_bytes each
    const size_t slot_bytes =
        (size_t)FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length_ * sizeof(uint8_t);
    const size_t dev_ring_bytes = (size_t)RING_BLOCKS * slot_bytes;
    cuda_check_error(cudaMalloc((void**)&ring_.base_d, dev_ring_bytes));
    cuda_check_error(cudaMemset(ring_.base_d, 0, dev_ring_bytes));
    ring_.slot_bytes = slot_bytes;
    ring_.num_bins   = rfft_length_;
    printf("GPU mag ring: %.1f MB\n", (double)dev_ring_bytes / 1e6);

    // cuFFT plan: single 1D C2C of rfft_length
    cufftResult fftRes = cufftPlan1d(&rplan_, (int)rfft_length_, CUFFT_C2C, 1);
    if (fftRes) { fprintf(stderr, "MagBlock: cufftPlan1d failed\n"); exit(1); }
    fftRes = cufftSetStream(rplan_, stream_);
    if (fftRes) { fprintf(stderr, "MagBlock: cufftSetStream failed\n"); exit(1); }
}

MagBlock::~MagBlock()
{
    if (demodData_d_)  cudaFree(demodData_d_);
    if (demodFT8_d_)   cudaFree(demodFT8_d_);
    if (demodShift_d_) cudaFree(demodShift_d_);
    if (magFT8_d_)     cudaFree(magFT8_d_);
    if (ring_.base_d)  cudaFree(ring_.base_d);
    cufftDestroy(rplan_);
    cudaEventDestroy(ring_.ready);
    cudaStreamDestroy(stream_);
}

void MagBlock::pubBin(const char* topic, const void* data, size_t len)
{
    std::lock_guard<std::mutex> lk(zmq_mu_);
    zmq::message_t t(topic, strlen(topic));
    zmq::message_t d(data, len);
    zmq_pub_.send(t, zmq::send_flags::sndmore);
    zmq_pub_.send(d, zmq::send_flags::none);
}

int MagBlock::doCopy(uint64_t now)
{
    const size_t length = inShape_[1];
    cudaMemcpyAsync(&demodData_d_[buff_pos_],
                    &inData_d_[length * (now % inShape_[0])],
                    length * sizeof(std::complex<float>),
                    cudaMemcpyDeviceToDevice, stream_);
    buff_pos_ += (uint32_t)length;

    if (buff_pos_ > 2 * rfft_length_) {
        // Run FT8_TIME_OSR × FT8_FREQ_OSR RFFTs with fractional frequency shifts.
        for (int t = 0; t < FT8_TIME_OSR; ++t) {
            for (int f = 0; f < FT8_FREQ_OSR; ++f) {
                std::complex<float>* input = &demodData_d_[t * rfft_length_ / FT8_TIME_OSR];
                if (f > 0) {
                    cuda_h_.freqShift(input, demodShift_d_, rfft_length_,
                                      f * 6.25f / FT8_FREQ_OSR);
                    input = demodShift_d_;
                }
                cufftResult rv = cufftExecC2C(rplan_,
                    (cufftComplex*)input,
                    (cufftComplex*)&demodFT8_d_[(t * FT8_FREQ_OSR + f) * rfft_length_],
                    CUFFT_FORWARD);
                if (rv) {
                    fprintf(stderr, "MagBlock: cufftExecC2C error (t=%d f=%d)\n", t, f);
                    return 0;
                }
            }
        }

        buff_pos_ -= (uint32_t)rfft_length_;
        cudaMemcpyAsync(&demodData_d_[0], &demodData_d_[rfft_length_],
                        buff_pos_ * sizeof(std::complex<float>),
                        cudaMemcpyDeviceToDevice, stream_);

        // Compute magnitude → ring slot.
        cuda_h_.magKernel(&demodFT8_d_[0], &magFT8_d_[0],
                          FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length_);

        const size_t block_bytes = (size_t)FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length_;
        uint64_t wi = ring_.write_idx.load(std::memory_order_relaxed);
        cuda_check_error(cudaMemcpyAsync(
            &ring_.base_d[(wi % RING_BLOCKS) * block_bytes],
            &magFT8_d_[0],
            block_bytes * sizeof(uint8_t),
            cudaMemcpyDeviceToDevice, stream_));

        // Record ring_.ready BEFORE incrementing write_idx so that any reader
        // who sees the new write_idx is guaranteed ring_.ready has been
        // recorded on the stream (and will complete after the memcpy above).
        cuda_check_error(cudaEventRecord(ring_.ready, stream_));
        ring_.write_idx.fetch_add(1, std::memory_order_release);
    }
    return 1;
}

void MagBlock::run()
{
    uint64_t now = inPos_->getNow(1) + 1;
    while (isRunning()) {
        uint64_t next = inPos_->getPosition(now + 1, 1);
        while (now < next) {
            uint64_t lag = next - now;
            if (lag > 4) {
                fprintf(stderr, "Error Falling Behind in MagBlock::doCopy, Dropping Data\n");
                now = next;
                break;
            }
            int copied = doCopy(now);
            if (!copied) exit(-200);
            now += copied;
        }
    }
}

} // namespace cuda
} // namespace gm
