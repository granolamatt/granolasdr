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

template<int N>
MagBlock<N>::MagBlock(gm::buffer::BufferPosition<std::complex<float>>* inP,
                      int rfft_len, int time_osr, int freq_osr, int zmq_port)
    : inPos_(inP), zmq_ctx_(1), zmq_pub_(zmq_ctx_, ZMQ_PUB)
{
    if (zmq_port > 0)
        zmq_pub_.connect("tcp://localhost:" + std::to_string(zmq_port));

    inShape_  = inPos_->getShape();
    inData_d_ = (std::complex<float>*)inPos_->getBuffer();

    rfft_length_ = (size_t)rfft_len;
    time_osr_    = time_osr;
    freq_osr_    = freq_osr;
    bin_hz_      = 204800.0f / (float)rfft_length_;

    cuda_check_error(cudaSetDevice(0));
    cuda_check_error(cudaStreamCreate(&stream_));
    cuda_check_error(cudaEventCreateWithFlags(&ring_.ready, cudaEventDisableTiming));

    cuda_h_ = gm::cuda::device::HostCuda(stream_);

    cuda_check_error(cudaMalloc((void**)&demodData_d_,
        4 * rfft_length_ * sizeof(std::complex<float>) + 1024));

    cuda_check_error(cudaMalloc((void**)&demodFT8_d_,
        (size_t)time_osr_ * freq_osr_ * rfft_length_ * sizeof(std::complex<float>) + 1024));

    cuda_check_error(cudaMalloc((void**)&magFT8_d_,
        (size_t)time_osr_ * freq_osr_ * rfft_length_ * sizeof(uint8_t) + 1024));

    cuda_check_error(cudaMalloc((void**)&demodShift_d_,
        rfft_length_ * sizeof(std::complex<float>) + 1024));

    const size_t slot_bytes = (size_t)time_osr_ * freq_osr_ * rfft_length_ * sizeof(uint8_t);
    const size_t dev_ring_bytes = (size_t)N * slot_bytes;
    cuda_check_error(cudaMalloc((void**)&ring_.base_d, dev_ring_bytes));
    cuda_check_error(cudaMemset(ring_.base_d, 0, dev_ring_bytes));
    ring_.slot_bytes = slot_bytes;
    ring_.num_bins   = rfft_length_;
    printf("GPU mag ring: %.1f MB  (N=%d  rfft=%zu  osr=%dx%d  bin=%.2f Hz)\n",
           (double)dev_ring_bytes / 1e6, N, rfft_length_, time_osr_, freq_osr_, (double)bin_hz_);

    cufftResult fftRes = cufftPlan1d(&rplan_, (int)rfft_length_, CUFFT_C2C, 1);
    if (fftRes) { fprintf(stderr, "MagBlock: cufftPlan1d failed\n"); exit(1); }
    fftRes = cufftSetStream(rplan_, stream_);
    if (fftRes) { fprintf(stderr, "MagBlock: cufftSetStream failed\n"); exit(1); }
}

template<int N>
MagBlock<N>::~MagBlock()
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

template<int N>
void MagBlock<N>::pubBin(const char* topic, const void* data, size_t len)
{
    std::lock_guard<std::mutex> lk(zmq_mu_);
    zmq::message_t t(topic, strlen(topic));
    zmq::message_t d(data, len);
    zmq_pub_.send(t, zmq::send_flags::sndmore);
    zmq_pub_.send(d, zmq::send_flags::none);
}

template<int N>
int MagBlock<N>::doCopy(uint64_t now)
{
    const size_t length = inShape_[1];
    cudaMemcpyAsync(&demodData_d_[buff_pos_],
                    &inData_d_[length * (now % inShape_[0])],
                    length * sizeof(std::complex<float>),
                    cudaMemcpyDeviceToDevice, stream_);
    buff_pos_ += (uint32_t)length;

    if (buff_pos_ > 2 * rfft_length_) {
        for (int t = 0; t < time_osr_; ++t) {
            for (int f = 0; f < freq_osr_; ++f) {
                std::complex<float>* input = &demodData_d_[t * rfft_length_ / time_osr_];
                if (f > 0) {
                    cuda_h_.freqShift(input, demodShift_d_, (int)rfft_length_,
                                      f * bin_hz_ / freq_osr_,
                                      204800.0f);
                    input = demodShift_d_;
                }
                cufftResult rv = cufftExecC2C(rplan_,
                    (cufftComplex*)input,
                    (cufftComplex*)&demodFT8_d_[(t * freq_osr_ + f) * rfft_length_],
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

        const size_t slot_elems = (size_t)time_osr_ * freq_osr_ * rfft_length_;
        cuda_h_.magKernel(&demodFT8_d_[0], &magFT8_d_[0], slot_elems);

        uint64_t wi = ring_.write_idx.load(std::memory_order_relaxed);
        cuda_check_error(cudaMemcpyAsync(
            ring_.slot(wi),
            &magFT8_d_[0],
            ring_.slot_bytes,
            cudaMemcpyDeviceToDevice, stream_));

        cuda_check_error(cudaEventRecord(ring_.ready, stream_));
        ring_.write_idx.fetch_add(1, std::memory_order_release);
    }
    return 1;
}

template<int N>
void MagBlock<N>::run()
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

template class MagBlock<200>;
template class MagBlock<128>;

} // namespace cuda
} // namespace gm
