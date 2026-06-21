#pragma once
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace gm {
namespace hf {

// Labeled LLR capture for next-generation LDPC decoder training data.
//
// One JSONL file per mode; each line carries a "status" label:
//   "pass"  decoded directly by the GPU SP / ft8_lib BP decoder (CRC valid)
//   "osd"   recovered only by the OSD fallback (CRC valid)
//   "fail"  strong-Costas candidate the OSD gate tried but no decoder cracked
//
// Schema (one object per line):
//   mode, status, score (Costas sync), snr, fo, to, ts, freq, unix,
//   text (decoded message; null for "fail"), osd_dist (soft distance; null
//   unless OSD was attempted), llr (N channel LLRs).
//
// Thread-safe; multiple decoder worker threads may write concurrently.
class LlrCapture {
public:
    LlrCapture() = default;
    ~LlrCapture();

    // Open "<prefix>_llr_<YYYYMMDD>[_<mode_tag>].jsonl" (appended) iff getenv(env_var)
    // is set.  mode_tag may be empty/null (FT8 has a single mode); spaces become '_'.
    void openIfEnabled(const char* env_var, const char* file_prefix, const char* mode_tag);

    bool enabled() const { return fp_ != nullptr; }

    void write(const char* mode, const char* status,
               const float* llr, int n,
               int32_t fo, int to, int ts, int score,
               double freq_hz, double snr, double unix_time,
               const char* text,        // nullptr -> JSON null
               const float* osd_dist);  // nullptr -> JSON null

private:
    FILE*      fp_ = nullptr;
    std::mutex mu_;
};

} // namespace hf
} // namespace gm
