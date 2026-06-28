#pragma once
#include <string>

namespace gm {
namespace hf {

// Pure-CPU CW (Morse) decoder core.  No CUDA, no I/O, no pipeline deps — so it
// is unit-testable host-only (links only -lm), and is the Phase 0 de-risk gate.
//
// Consumes a magnitude envelope sampled at a fixed hop (default 5 ms / 200 Hz):
//
//   env[t] ─► [noise-floor / peak EMA] ─► hysteresis key (T_hi, T_lo)
//          ─► run-length ─► mark: dit/dah vs ADAPTIVE dit ─► symbol buffer
//                        └► space: elem / char / word ─► Morse table ─► text
//
// The dit length is estimated continuously from the observed marks (no fixed
// WPM assumption); seed it via the constructor for faster lock on short bursts.
class CwMorse {
public:
    explicit CwMorse(float hop_ms = 5.0f, float init_wpm = 20.0f);
    void reset();

    // Streaming: feed one envelope magnitude sample.
    void push(float mag);
    // End of stream / signal loss: close any open mark + pending symbol.
    void flush();
    // Drain decoded text accumulated so far (clears the internal buffer).
    std::string take();

    // Convenience for offline/tests: reset, push the whole buffer, flush, return text.
    std::string decode(const float* env, int n);

    float wpm()   const;                 // current adaptive estimate
    float ditMs() const { return dit_ms_; }

private:
    void edge(bool key_now);          // common transition bookkeeping
    void onMark(float ms);
    void onSpace(float ms);
    char symbolToChar() const;
    void flushSymbol(bool add_word_space);

    float hop_ms_;
    float init_dit_ms_;
    float dit_ms_;
    // Adaptive threshold trackers.
    float nf_, pk_;        // noise floor, signal peak (EMA)
    bool  have_env_;
    bool  key_;            // current key state (true = key-down / mark)
    int   run_;            // samples in the current run
    bool  started_;        // have we seen the first key-down yet?
    std::string symbol_;   // current ".-" element buffer
    std::string text_;     // decoded output
};

} // namespace hf
} // namespace gm
