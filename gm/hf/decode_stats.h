#pragma once
// Lightweight per-mode decode-rate counter (A/B comparisons, tuning, monitoring).
// Enabled with DECODE_STATS=1; prints a cumulative + per-minute summary every
// DECODE_STATS_SEC seconds (default 60), broken down by mode.  Header-only,
// thread-safe; call gm::hf::decode_stats_count("FT8") once per emitted decode.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace gm { namespace hf {

class DecodeStats {
public:
    static DecodeStats& instance() { static DecodeStats s; return s; }

    void count(const char* mode) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lk(mu_);
        counts_[mode]++;
        total_++;
    }

private:
    DecodeStats() {
        const char* e = std::getenv("DECODE_STATS");
        enabled_ = e && std::string(e) != "0";
        if (!enabled_) return;
        const char* s = std::getenv("DECODE_STATS_SEC");
        period_sec_ = s ? std::atoi(s) : 60;
        if (period_sec_ < 1) period_sec_ = 60;
        start_  = std::chrono::steady_clock::now();
        worker_ = std::thread([this] { run(); });
        worker_.detach();
        printf("Decode stats: ON (summary every %d s; set DECODE_STATS=0 to disable)\n",
               period_sec_);
    }

    void run() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(period_sec_));
            std::lock_guard<std::mutex> lk(mu_);
            double min = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start_).count() / 60.0;
            if (min <= 0.0) continue;
            std::string line;
            char buf[160];
            for (const auto& kv : counts_) {
                snprintf(buf, sizeof buf, "%s=%llu(%.1f/min) ", kv.first.c_str(),
                         (unsigned long long)kv.second, (double)kv.second / min);
                line += buf;
            }
            snprintf(buf, sizeof buf, "TOTAL=%llu (%.1f/min over %.1f min)",
                     (unsigned long long)total_, (double)total_ / min, min);
            printf("[DECODE STATS] %s%s\n", line.c_str(), buf);
            fflush(stdout);
        }
    }

    bool enabled_{false};
    int  period_sec_{60};
    std::mutex mu_;
    std::map<std::string, unsigned long long> counts_;
    unsigned long long total_{0};
    std::chrono::steady_clock::time_point start_;
    std::thread worker_;
};

inline void decode_stats_count(const char* mode) { DecodeStats::instance().count(mode); }

}} // namespace gm::hf
