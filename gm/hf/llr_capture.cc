#include "gm/hf/llr_capture.h"

#include <cstdlib>
#include <ctime>

#include "third_party/nlohmann_json.hpp"

namespace gm {
namespace hf {

LlrCapture::~LlrCapture()
{
    if (fp_) fclose(fp_);
}

void LlrCapture::openIfEnabled(const char* env_var, const char* file_prefix,
                               const char* mode_tag)
{
    if (!std::getenv(env_var)) return;

    time_t now_t = time(nullptr);
    struct tm* tm_p = gmtime(&now_t);

    char fname[96];
    if (mode_tag && mode_tag[0])
        snprintf(fname, sizeof(fname), "%s_llr_%04d%02d%02d_%s.jsonl",
                 file_prefix, tm_p->tm_year + 1900, tm_p->tm_mon + 1, tm_p->tm_mday, mode_tag);
    else
        snprintf(fname, sizeof(fname), "%s_llr_%04d%02d%02d.jsonl",
                 file_prefix, tm_p->tm_year + 1900, tm_p->tm_mon + 1, tm_p->tm_mday);
    for (char* p = fname; *p; ++p) if (*p == ' ') *p = '_';

    fp_ = fopen(fname, "a");
    if (fp_)
        printf("[%s] LLR capture -> %s\n", mode_tag && mode_tag[0] ? mode_tag : file_prefix, fname);
    else
        fprintf(stderr, "[%s] LLR capture open failed: %s\n",
                mode_tag && mode_tag[0] ? mode_tag : file_prefix, fname);
}

void LlrCapture::write(const char* mode, const char* status,
                       const float* llr, int n,
                       int32_t fo, int to, int ts, int score,
                       double freq_hz, double snr, double unix_time,
                       const char* text, const float* osd_dist)
{
    if (!fp_) return;

    nlohmann::json j;
    j["mode"]   = mode;
    j["status"] = status;
    j["score"]  = score;
    j["snr"]    = snr;
    j["fo"]     = fo;
    j["to"]     = to;
    j["ts"]     = ts;
    j["freq"]   = freq_hz;
    j["unix"]   = unix_time;
    if (text) j["text"] = text;
    else      j["text"] = nullptr;
    if (osd_dist) j["osd_dist"] = (double)*osd_dist;
    else          j["osd_dist"] = nullptr;

    auto& la = j["llr"] = nlohmann::json::array();
    for (int i = 0; i < n; ++i) la.push_back((double)llr[i]);

    std::string line = j.dump(-1, ' ', false,
                              nlohmann::json::error_handler_t::replace) + "\n";

    std::lock_guard<std::mutex> lk(mu_);
    fputs(line.c_str(), fp_);
    fflush(fp_);
}

} // namespace hf
} // namespace gm
