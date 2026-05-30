#ifndef _GM_CUDA_HFCHANNELIZER_H_
#define _GM_CUDA_HFCHANNELIZER_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cufft.h>
#include <zmq.hpp>
#include "gm/cuda/HostCuda.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

// Per-SSE-connection event queue.  push() evicts oldest when full (depth 200).
// pop() blocks up to 15 s; returns a keep-alive SSE comment on timeout so that
// httplib's content provider can detect a broken connection via sink.write().
struct SseQueue {
    void push(std::string frame) {
        std::lock_guard<std::mutex> lk(mutex_);
        if (q_.size() >= 200) q_.pop();
        q_.push(std::move(frame));
        cv_.notify_one();
    }
    std::string pop() {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_.wait_for(lk, std::chrono::seconds(15), [this]{ return !q_.empty(); });
        if (!q_.empty()) {
            auto s = std::move(q_.front()); q_.pop(); return s;
        }
        return ": keep-alive\n\n";
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::string> q_;
};

class HFChannelizer : public Thread {
public:
    HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP,
                  const std::string& ctrl_host = "127.0.0.1",
                  int ctrl_port = 8080);
    ~HFChannelizer();
    void run();
    void stop() {
        setRunning(false);
    }
    gm::buffer::BufferPosition<std::complex<float>>* getBuffer() {
        return &hfBufferPosition;
    }

    // Register the timing callback; called by HFRx.cc after constructing FT8Cuda.
    // FT8Cuda's snapshot thread calls this with (scan_ms, ldpc_ms, n) after each epoch.
    void setTimingCallback(std::function<void(float, float, uint32_t)> cb) {
        timing_callback_ = std::move(cb);
    }

    // Push a decode event to all connected SSE clients.
    // mode defaults to "FT8"; pass "JS8" for JS8 decodes.
    void broadcastDecode(const char* call, float freq_hz, float snr, double unix_time,
                         const char* mode = "FT8");

    // Push FT8 timing to all SSE clients (type:"timing").
    void broadcastTiming(float scan_ms, float ldpc_ms, uint32_t n);

    // Push JS8 scan timing to all SSE clients (type:"js8timing").
    void broadcastJS8Timing(float scan_ms, uint32_t n);

private:
    cudaStream_t stream;
    cufftHandle plan;
    cufftHandle iplan;
    cufftHandle audio_plan;  // batched NUM_SINKS × AUDIO_BINS C2C IFFT
    double lastepoch;

    const static int BUFFERS = 16;

    int num_blocks;
    uint64_t buffer_number;

    gm::buffer::BufferPosition<std::complex<float>> hfBufferPosition;

    gm::cuda::device::HostCuda cuda_h;
    std::vector<size_t> inShape;
    int16_t* inData_d;
    int16_t* inData;
    float* fftInData_d;
    std::complex<float>* fftData_d;
    gm::buffer::BufferPosition<int16_t>* inPos;
    int doCopy(uint64_t now);
    std::complex<float>* channelData_d;
    std::complex<float>* demodData_d;
    std::complex<float>* audioBins_d;  // device staging: NUM_SINKS × AUDIO_BINS for batched IFFT
    std::vector<std::vector<uint32_t>> bins;
    uint32_t fft_length;
    uint32_t nTune;
    uint32_t nChannels;

    // Audio extraction: NUM_SINKS tunable 48 kHz channels from front FFT.
    // Each sink is independently tunable at runtime via REST API.
    // AUDIO_BINS bins per sink → AUDIO_VALID valid PCM samples/block at 48 kHz.
    static const int NUM_SINKS  = 4;
    static const int AUDIO_BINS = 480;  // bins from front FFT (100 Hz/bin, 48 kHz BW)
    static const int AUDIO_VALID = 240; // valid samples per block (overlap-save, 48 kHz)
    static const int AUDIO_RING  = 16;  // ring depth for D2H→worker handoff

    // Per-sink tuning state: start bin in wideband FFT (freq_hz / 100).
    // Written by controlWorker, read in doCopy — atomic for wait-free access.
    std::atomic<uint32_t> sink_bins[NUM_SINKS];
    std::string           sink_labels[NUM_SINKS];

    // Pinned host ring: AUDIO_RING × NUM_SINKS × AUDIO_BINS complex floats
    std::complex<float>* audio_pinned;

    std::atomic<uint64_t> audio_produce_idx{0};
    std::atomic<uint64_t> audio_consume_idx{0};
    std::thread audio_thread;
    void audioWorker();

    // REST control server + SSE event broadcast (cpp-httplib, runs in its own thread)
    std::string ctrl_host_;
    int         ctrl_port_;
    std::thread ctrl_thread;
    void controlWorker();

    // SSE broadcast state: one SseQueue per connected browser tab.
    std::mutex sse_mutex_;
    std::vector<std::shared_ptr<SseQueue>> sse_queues_;

    // Registered by HFRx.cc so FT8Cuda's snapshot thread can push timing data.
    std::function<void(float, float, uint32_t)> timing_callback_;

    // WebSocket /waterfall: one frame queue per connected browser tab.
    // Uses SseQueue (stores std::string) for binary waterfall frames.
    std::mutex ws_mutex_;
    std::vector<std::shared_ptr<SseQueue>> ws_queues_;

public:
    // Called from FT8Cuda's waterfall_callback_ with each 2048-byte frame.
    void broadcastWaterfall(const uint8_t* data, int len);
private:

    zmq::context_t audio_zmq_ctx;
    zmq::socket_t* audio_sockets[NUM_SINKS];

    // Recording state — writes channelizer output blocks to a binary file.
    static const int RECORD_RING = 4;
    std::atomic<bool>            recording_{false};
    FILE*                        record_fp_{nullptr};
    std::complex<float>*         record_pinned_{nullptr};
    std::atomic<uint64_t>        record_produce_{0};
    std::atomic<uint64_t>        record_consume_{0};
    std::thread                  record_thread_;
    void recordWorker();

public:
    // Open filename and start writing channelizer output blocks.  Safe to call
    // after start().  Stop with stopRecording() or object destruction.
    void startRecording(const std::string& filename);
    void stopRecording();
};
}
}

#endif // _GM_CUDA_HFCHANNELIZER_H_
