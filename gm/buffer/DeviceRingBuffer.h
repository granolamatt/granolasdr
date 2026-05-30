#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace gm {
namespace buffer {

// Non-copyable GPU ring buffer shared between one writer (MagBlock) and
// multiple readers (FT8Cuda, JS8Cuda). Readers track their own read cursor
// against write_idx; they wait on ready before accessing device memory.
//
// slot_bytes = FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length * sizeof(T)
// num_bins   = rfft_length  (passed as num_bins argument to scan kernels)
template<typename T, int N>
struct DeviceRingBuffer {
    T*                    base_d{nullptr};
    size_t                slot_bytes{0};
    size_t                num_bins{0};
    std::atomic<uint64_t> write_idx{0};
    cudaEvent_t           ready{};

    T* slot(uint64_t idx) const {
        return base_d + (idx % N) * (slot_bytes / sizeof(T));
    }

    DeviceRingBuffer()                               = default;
    DeviceRingBuffer(const DeviceRingBuffer&)        = delete;
    DeviceRingBuffer& operator=(const DeviceRingBuffer&) = delete;
};

} // namespace buffer
} // namespace gm
