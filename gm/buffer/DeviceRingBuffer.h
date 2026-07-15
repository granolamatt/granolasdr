#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <cuda_runtime.h>

namespace gm {
namespace buffer {

// Depth (in composite input blocks) of MagBlock's optional complex-composite
// retention ring, shared by MagBlock (writer) and FT8Cuda/JS8Cuda (refine
// readers). One FT8/JS8 frame is 79*32 = 2528 blocks (~12.6 s); 4096 ~= 20 s.
inline constexpr int kComplexCompositeBlocks = 4096;

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
    // Serializes the writer's cudaEventRecord(ready) against every reader's
    // cudaStreamWaitEvent(ready). One `ready` event is re-recorded each block by
    // MagBlock and waited on concurrently by FT8/JS8/Waterfall scan streams from
    // different threads; unsynchronized event ops from multiple threads race and,
    // under heavy dispatch load, can leave a scan stream permanently waiting on a
    // torn event state (GPU idle, consumer wedged). `mutable` so const readers can
    // lock it. Wrap EVERY record/wait on this event with lock_guard(ready_mu).
    mutable std::mutex    ready_mu;

    T* slot(uint64_t idx) const {
        return base_d + (idx % N) * (slot_bytes / sizeof(T));
    }

    DeviceRingBuffer()                               = default;
    DeviceRingBuffer(const DeviceRingBuffer&)        = delete;
    DeviceRingBuffer& operator=(const DeviceRingBuffer&) = delete;
};

} // namespace buffer
} // namespace gm
