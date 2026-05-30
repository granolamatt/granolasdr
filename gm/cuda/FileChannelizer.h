#ifndef _GM_CUDA_FILECHANNELIZER_H_
#define _GM_CUDA_FILECHANNELIZER_H_

#include <cstdint>
#include <complex>
#include <functional>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

// Binary file format written by HFChannelizer recording and read by FileChannelizer.
// Followed immediately by N blocks of block_samples complex<float> each.
#pragma pack(push, 1)
struct ChannelizerFileHeader {
    uint32_t magic;             // 0x474E4C48 ('G','N','L','H')
    uint32_t version;           // 1
    uint32_t block_samples;     // complex<float> samples per block (e.g. 32768)
    uint32_t reserved;
    uint64_t block_interval_ns; // nanoseconds between blocks (e.g. 5000000 = 5 ms)
    uint32_t sample_rate_hz;    // channelizer output sample rate
    uint32_t rx_sample_rate;    // original rx888 sample rate
};
#pragma pack(pop)

static const uint32_t kChannelizerFileMagic = 0x474E4C48u;

// Plays back a file recorded by HFChannelizer into the same BufferPosition
// that FT8Cuda/JS8Cuda expect.  Drop-in replacement for HFChannelizer in
// HFRx when --playback is given.  Broadcast methods are no-ops since there
// is no live web dashboard in playback mode.
class FileChannelizer : public Thread {
public:
    explicit FileChannelizer(const std::string& filename, bool loop = true);
    ~FileChannelizer();
    void run();
    void stop() { setRunning(false); }

    gm::buffer::BufferPosition<std::complex<float>>* getBuffer() {
        return &hfBufferPosition;
    }

    void setTimingCallback(std::function<void(float, float, uint32_t)>) {}
    void broadcastTiming(float, float, uint32_t) {}
    void broadcastJS8Timing(float, uint32_t) {}
    void broadcastDecode(const char*, float, float, double, const char* = "FT8") {}
    void broadcastWaterfall(const uint8_t*, int) {}

private:
    static const int BUFFERS = 16;

    std::string filename_;
    bool        loop_;
    FILE*       fp_{nullptr};

    uint32_t    block_samples_{0};
    uint64_t    block_interval_ns_{0};

    cudaStream_t                        stream_{};
    std::complex<float>*                demodData_d{nullptr};
    std::complex<float>*                host_block{nullptr};
    uint64_t                            buffer_number{0};

    gm::buffer::BufferPosition<std::complex<float>> hfBufferPosition;
};

} // namespace cuda
} // namespace gm

#endif // _GM_CUDA_FILECHANNELIZER_H_
