#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <complex>
#include <thread>
#include <cuda_runtime.h>
#include "gm/buffer/BufferFile.h"

namespace gm {
namespace buffer {

// ---- Playback constructor ---------------------------------------------------

template<typename T>
BufferFile<T>::BufferFile(const std::string& path, bool loop)
    : mode_(Mode::Playback), path_(path), loop_(loop)
{
    fp_ = fopen(path_.c_str(), "rb");
    if (!fp_) { perror(("BufferFile: cannot open " + path_).c_str()); exit(1); }

    BufferFileHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, fp_) != 1) {
        fprintf(stderr, "BufferFile: failed to read header from %s\n", path_.c_str());
        exit(1);
    }
    if (hdr.magic != kBufferFileMagic) {
        // Accept old FileChannelizer magic (same value) for backward compat.
        fprintf(stderr, "BufferFile: bad magic 0x%08x in %s\n", hdr.magic, path_.c_str());
        exit(1);
    }
    if (hdr.version != 1) {
        fprintf(stderr, "BufferFile: unsupported version %u\n", hdr.version);
        exit(1);
    }

    block_samples_     = hdr.block_samples;
    block_interval_ns_ = hdr.block_interval_ns;

    long data_start = (long)sizeof(BufferFileHeader);
    fseek(fp_, 0, SEEK_END);
    long file_end = ftell(fp_);
    fseek(fp_, data_start, SEEK_SET);
    uint64_t total_blocks = (uint64_t)(file_end - data_start) /
                            ((uint64_t)block_samples_ * sizeof(T));
    double total_sec = (double)total_blocks * block_interval_ns_ / 1e9;
    printf("BufferFile: %s  blocks=%u  interval=%.3f ms  out_rate=%u Hz  rx_rate=%u Hz\n",
           path_.c_str(), block_samples_,
           (double)block_interval_ns_ / 1e6,
           hdr.sample_rate_hz, hdr.rx_sample_rate);
    printf("BufferFile: file contains %llu blocks (%.1f s); scans begin after ~20 s warmup\n",
           (unsigned long long)total_blocks, total_sec);

    cudaSetDevice(0);
    cudaStreamCreate(&stream_);

    size_t ring_bytes = (size_t)ring_slots_ * block_samples_ * sizeof(T);
    cudaMalloc((void**)&ring_d_, ring_bytes);
    cudaHostAlloc((void**)&host_block_, block_samples_ * sizeof(T), cudaHostAllocDefault);

    buf_pos_.setBuffer(ring_d_, {(size_t)ring_slots_, (size_t)block_samples_});
}

// ---- Record constructor ----------------------------------------------------

template<typename T>
BufferFile<T>::BufferFile(gm::buffer::BufferPosition<T>* src,
                          const std::string& path,
                          const BufferFileParams& params)
    : mode_(Mode::Record), path_(path), src_(src), params_(params)
{
    fp_ = fopen(path_.c_str(), "wb");
    if (!fp_) { perror(("BufferFile: cannot open " + path_).c_str()); exit(1); }

    BufferFileHeader hdr{};
    hdr.magic            = kBufferFileMagic;
    hdr.version          = 1;
    hdr.block_samples    = params_.block_samples;
    hdr.element_bytes    = sizeof(T);
    hdr.block_interval_ns = params_.block_interval_ns;
    hdr.sample_rate_hz   = params_.sample_rate_hz;
    hdr.rx_sample_rate   = params_.rx_sample_rate;
    fwrite(&hdr, sizeof(hdr), 1, fp_);

    cudaHostAlloc((void**)&record_host_, params_.block_samples * sizeof(T), cudaHostAllocDefault);
    printf("BufferFile: recording to %s  (block=%u samples, %.3f ms each)\n",
           path_.c_str(), params_.block_samples,
           (double)params_.block_interval_ns / 1e6);
}

// ---- Destructor ------------------------------------------------------------

template<typename T>
BufferFile<T>::~BufferFile()
{
    if (fp_)           fclose(fp_);
    if (ring_d_)       cudaFree(ring_d_);
    if (host_block_)   cudaFreeHost(host_block_);
    if (record_host_)  cudaFreeHost(record_host_);
    if (stream_)       cudaStreamDestroy(stream_);
}

// ---- run() dispatch --------------------------------------------------------

template<typename T>
void BufferFile<T>::run()
{
    if (mode_ == Mode::Playback) runPlayback();
    else                         runRecord();
}

// ---- Playback loop ---------------------------------------------------------

template<typename T>
void BufferFile<T>::runPlayback()
{
    const size_t block_bytes = block_samples_ * sizeof(T);
    const long   data_start  = (long)sizeof(BufferFileHeader);

    using Clock = std::chrono::steady_clock;
    using NS    = std::chrono::nanoseconds;
    auto next_tick = Clock::now();
    const auto interval = NS((long long)block_interval_ns_);

    while (isRunning()) {
        size_t n = fread(host_block_, sizeof(T), block_samples_, fp_);
        if (n < block_samples_) {
            if (feof(fp_) && loop_) {
                fseek(fp_, data_start, SEEK_SET);
                continue;
            }
            printf("BufferFile: end of file\n");
            break;
        }

        size_t slot = block_num_ % (size_t)ring_slots_;
        cudaMemcpyAsync(&ring_d_[slot * block_samples_], host_block_, block_bytes,
                        cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);
        buf_pos_.setPosition(block_num_, 1);
        block_num_++;

        if (block_num_ % 1000 == 0)
            printf("BufferFile: fed %llu blocks (%.1f s)\n",
                   (unsigned long long)block_num_,
                   (double)block_num_ * block_interval_ns_ / 1e9);

        next_tick += interval;
        std::this_thread::sleep_until(next_tick);
    }
}

// ---- Record loop -----------------------------------------------------------

template<typename T>
void BufferFile<T>::runRecord()
{
    const size_t block_bytes   = params_.block_samples * sizeof(T);
    const auto   shape         = src_->getShape();  // {num_slots, block_samples}
    const size_t num_slots     = shape[0];
    uint64_t     pos           = src_->getNow(1);

    while (isRunning()) {
        uint64_t next = src_->getPosition(pos + 1, 1);
        size_t   slot = (size_t)(next % num_slots);
        cudaMemcpy(record_host_, src_->getBuffer() + slot * params_.block_samples,
                   block_bytes, cudaMemcpyDeviceToHost);
        fwrite(record_host_, block_bytes, 1, fp_);
        pos = next;
    }
    fflush(fp_);
}

// ---- Explicit instantiation ------------------------------------------------

template class BufferFile<std::complex<float>>;

} // namespace buffer
} // namespace gm
