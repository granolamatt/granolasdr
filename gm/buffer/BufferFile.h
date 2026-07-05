#pragma once
#include <atomic>
#include <complex>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace buffer {

// Binary file format shared by BufferFile playback and record modes.
#pragma pack(push, 1)
struct BufferFileHeader {
    uint32_t magic;             // 0x474E4C48 ('G','N','L','H')
    uint32_t version;           // 1
    uint32_t block_samples;     // elements per block (e.g. 32768 complex<float>)
    uint32_t element_bytes;     // sizeof(T): sanity check on read
    uint64_t block_interval_ns; // nanoseconds between blocks (paces playback)
    uint32_t sample_rate_hz;    // channelizer output sample rate
    uint32_t rx_sample_rate;    // original RX sample rate (0 if N/A)
};
#pragma pack(pop)

static const uint32_t kBufferFileMagic = 0x474E4C48u;

// Parameters needed to write a BufferFile header (obtained from the producer).
struct BufferFileParams {
    uint32_t block_samples;
    uint64_t block_interval_ns;
    uint32_t sample_rate_hz;
    uint32_t rx_sample_rate;
};

// BufferFile<T>: save any BufferPosition<T> to a file, or replay a file as one.
//
// Playback constructor: reads file → allocates GPU ring → feeds BufferPosition<T>.
// Record constructor:   reads BufferPosition<T> → writes blocks to file.
template<typename T>
class BufferFile : public gm::Thread {
public:
    // Playback: allocates internal GPU ring, feeds it from file.
    explicit BufferFile(const std::string& path, bool loop = true);

    // Record: drains from src buffer, writes to file at path.
    BufferFile(gm::buffer::BufferPosition<T>* src,
               const std::string& path,
               const BufferFileParams& params);

    ~BufferFile();
    void run() override;
    void stop() { setRunning(false); }

    // Valid only in playback mode.
    gm::buffer::BufferPosition<T>* getBuffer() { return &buf_pos_; }

    // Playback, non-looping: true once the whole file has been fed and EOF hit.
    bool finished() const { return finished_.load(std::memory_order_acquire); }

private:
    std::atomic<bool> finished_{false};

    enum class Mode { Playback, Record };
    Mode        mode_;
    std::string path_;
    bool        loop_{false};
    FILE*       fp_{nullptr};

    // Playback state
    uint32_t     block_samples_{0};
    uint64_t     block_interval_ns_{0};
    T*           ring_d_{nullptr};      // GPU ring buffer
    T*           host_block_{nullptr};  // pinned staging block
    int          ring_slots_{16};
    uint64_t     block_num_{0};
    cudaStream_t stream_{};
    gm::buffer::BufferPosition<T> buf_pos_;

    // Record state
    gm::buffer::BufferPosition<T>* src_{nullptr};
    BufferFileParams                params_{};
    T*                              record_host_{nullptr};  // pinned D2H staging

    void runPlayback();
    void runRecord();
};

} // namespace buffer
} // namespace gm
