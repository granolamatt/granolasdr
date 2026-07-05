#ifndef _GM_CUDA_HFCHANNELIZER_H_
#define _GM_CUDA_HFCHANNELIZER_H_

#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cufft.h>
#include "gm/cuda/HostCuda.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/buffer/BufferFile.h"

namespace gm {
namespace cuda {


class HFChannelizer : public Thread {
public:
    HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP,
                  int wsdict_port = 8765, bool cw_enabled = false);
    ~HFChannelizer();
    void run();
    void stop() {
        setRunning(false);
    }
    void join() {
        if (tci_vfo_thread_.joinable()) tci_vfo_thread_.join();
        if (audio_thread.joinable())    audio_thread.join();
    }
    gm::buffer::BufferPosition<std::complex<float>>* getBuffer() {
        return &hfBufferPosition;
    }
    // Second composite output: CW sub-bands (kCWBands) packed into an 8192-pt
    // IFFT -> 819.2 kHz.  Only produced when cw_enabled (--cw); null/idle
    // otherwise.  Built in-place from the same wideband FFT as the FT8/JS8
    // composite, so the FT8/JS8 path is byte-identical regardless.
    gm::buffer::BufferPosition<std::complex<float>>* getCWBuffer() {
        return &cwBufferPosition;
    }


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

    // CW skimmer second composite (parallel to hfBufferPosition; --cw only).
    bool          cw_enabled_{false};
    gm::buffer::BufferPosition<std::complex<float>> cwBufferPosition;
    cufftHandle   cw_iplan{0};
    std::complex<float>* cwChannelData_d{nullptr};   // cw_fft_length, packed+IFFT scratch
    std::complex<float>* cwDemodData_d{nullptr};     // BUFFERS × cw_fft_length/2 ring
    std::vector<std::vector<uint32_t>> cw_bins;       // {wb_start, wb_end, bw} per CW band
    uint32_t      cw_fft_length{0};
    uint64_t      cw_buffer_number{0};

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

    // wsdict audio command worker: publishes granolasdr:audio:status:N keys and
    // subscribes to granolasdr:audio:cmd for tune requests from the browser.
    int         wsdict_port_;
    std::thread cmd_thread_;
    void cmdWorker();

    // TCI VFO retune worker: drains tci_poll_vfo() and applies freq changes
    // to sink_bins[]. Joinable; joined in ~HFChannelizer() (D2).
    std::thread tci_vfo_thread_;
    void tciVfoWorker();


    // Spectral noise-floor normalization: every NORM_INTERVAL frames, snapshot
    // per-band magnitudes from fftData_d, fit a degree-NORM_POLY_DEG Legendre
    // polynomial to each band's log-magnitude, and compute per-bin gains that
    // flatten the noise floor to the band's center-frequency level.
    static const int NORM_INTERVAL = 64;   // ~10 s at 6.25 Hz frame rate
    static const int NORM_POLY_DEG = 3;

    int   norm_frame_{0};
    int   norm_update_count_{0};           // number of EMA updates applied so far
    int   norm_total_bins_{0};             // Σ b[2] for all bands (2110)
    float* norm_gains_d_{nullptr};         // device: one gain per composite bin
    std::vector<float>               norm_gains_h_;   // host mirror
    std::vector<std::complex<float>> norm_snap_h_;    // D2H workspace
    std::vector<float>               norm_ema_h_;     // EMA of linear |mag| per bin
    std::vector<float>               norm_logmag_h_;  // log10(ema) per bin for poly fit

    // Wideband equalization (WIDEBAND_EQ=1): flatten the ENTIRE wideband FFT
    // before any bin-selection, so CW and audio (tunable anywhere) get equalized
    // too.  Mutually exclusive with the per-band norm above.  All GPU-side.
    bool   wideband_eq_{false};
    int    weq_nbins_{0};                   // R2C bins of the wideband FFT (NLARGE+1)
    int    weq_half_win_{128};              // box-filter half window (WEQ_WIN)
    float  weq_max_gain_{100.0f};           // gain clamp (WEQ_MAXGAIN)
    int    weq_frame_{0};
    int    weq_update_count_{0};
    float* weq_ema_d_{nullptr};             // per-bin noise-floor EMA
    float* weq_logema_d_{nullptr};          // log10(ema) scratch for the box filter
    float* weq_gain_targ_d_{nullptr};       // latest computed per-bin gain (ramp end)
    float* weq_gain_prev_d_{nullptr};       // gain at start of the current ramp
    float* weq_sum_d_{nullptr};             // 1-float global-mean-log accumulator

public:
    gm::buffer::BufferFileParams getBufferFileParams() const;
    // Same, for the CW composite (819.2 kHz, cw_fft_length/2 samples/block).
    // Valid only when cw_enabled (--cw); used to record getCWBuffer().
    gm::buffer::BufferFileParams getCWBufferFileParams() const;
};
}
}

#endif // _GM_CUDA_HFCHANNELIZER_H_
