#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#include "gm/cuda/FileChannelizer.h"

namespace gm {
namespace cuda {

FileChannelizer::FileChannelizer(const std::string& filename, bool loop)
    : filename_(filename), loop_(loop) {

    fp_ = fopen(filename_.c_str(), "rb");
    if (!fp_) {
        perror(("FileChannelizer: cannot open " + filename_).c_str());
        exit(1);
    }

    ChannelizerFileHeader hdr{};
    if (fread(&hdr, sizeof(hdr), 1, fp_) != 1) {
        fprintf(stderr, "FileChannelizer: failed to read header from %s\n", filename_.c_str());
        exit(1);
    }
    if (hdr.magic != kChannelizerFileMagic) {
        fprintf(stderr, "FileChannelizer: bad magic 0x%08x in %s\n", hdr.magic, filename_.c_str());
        exit(1);
    }
    if (hdr.version != 1) {
        fprintf(stderr, "FileChannelizer: unsupported version %u\n", hdr.version);
        exit(1);
    }

    block_samples_      = hdr.block_samples;
    block_interval_ns_  = hdr.block_interval_ns;

    // Compute total duration for the progress banner.
    long data_start = (long)sizeof(ChannelizerFileHeader);
    fseek(fp_, 0, SEEK_END);
    long file_end = ftell(fp_);
    fseek(fp_, data_start, SEEK_SET);
    long data_bytes = file_end - data_start;
    uint64_t total_blocks = (data_bytes > 0 && hdr.block_samples > 0)
        ? (uint64_t)data_bytes / ((uint64_t)hdr.block_samples * sizeof(std::complex<float>))
        : 0;
    double total_sec = (double)total_blocks * hdr.block_interval_ns / 1e9;

    printf("FileChannelizer: %s  blocks=%u  interval=%.3f ms  out_rate=%u Hz  rx_rate=%u Hz\n",
           filename_.c_str(), hdr.block_samples,
           (double)hdr.block_interval_ns / 1e6,
           hdr.sample_rate_hz, hdr.rx_sample_rate);
    printf("FileChannelizer: file contains %llu blocks (%.1f s); FT8/JS8 scans begin after ~20 s warmup\n",
           (unsigned long long)total_blocks, total_sec);

    cudaSetDevice(0);
    cudaStreamCreate(&stream_);

    const size_t ring_bytes = (size_t)BUFFERS * block_samples_ * sizeof(std::complex<float>);
    cudaMalloc((void**)&demodData_d, ring_bytes);

    cudaHostAlloc((void**)&host_block,
                  block_samples_ * sizeof(std::complex<float>),
                  cudaHostAllocDefault);

    // Shape must match HFChannelizer: {BUFFERS, block_samples}
    hfBufferPosition.setBuffer(demodData_d, {(size_t)BUFFERS, (size_t)block_samples_});
}

FileChannelizer::~FileChannelizer() {
    if (fp_)          fclose(fp_);
    if (host_block)   cudaFreeHost(host_block);
    if (demodData_d)  cudaFree(demodData_d);
    cudaStreamDestroy(stream_);
}

void FileChannelizer::run() {
    const size_t block_bytes = block_samples_ * sizeof(std::complex<float>);
    const long   data_start  = (long)sizeof(ChannelizerFileHeader);

    using Clock    = std::chrono::steady_clock;
    using NS       = std::chrono::nanoseconds;
    auto next_tick = Clock::now();
    const auto interval = NS((long long)block_interval_ns_);

    while (isRunning()) {
        size_t n = fread(host_block, sizeof(std::complex<float>), block_samples_, fp_);
        if (n < block_samples_) {
            if (feof(fp_) && loop_) {
                fseek(fp_, data_start, SEEK_SET);
                continue;
            }
            printf("FileChannelizer: end of file\n");
            break;
        }

        const size_t ring_slot = buffer_number % (size_t)BUFFERS;
        cudaMemcpyAsync(
            &demodData_d[ring_slot * block_samples_],
            host_block, block_bytes,
            cudaMemcpyHostToDevice, stream_);
        cudaStreamSynchronize(stream_);

        hfBufferPosition.setPosition(buffer_number, 1);
        buffer_number++;

        if (buffer_number % 1000 == 0)
            printf("FileChannelizer: fed %llu blocks (%.1f s)\n",
                   (unsigned long long)buffer_number,
                   (double)buffer_number * block_interval_ns_ / 1e9);

        // Pace to real time so FT8Cuda sees authentic epoch timing.
        next_tick += interval;
        std::this_thread::sleep_until(next_tick);
    }
}

} // namespace cuda
} // namespace gm
