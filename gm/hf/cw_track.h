#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gm {
namespace hf {

// Amateur callsign shape check: prefix (1-3 alnum, letter first) + area digit +
// letter suffix (1-4).  Rejects "E E EE" junk and partial fragments that lack a
// digit; a shape-valid misdecode is caught by the tracker's confirm gate.
bool looksLikeCall(const std::string& s);

// A confirmed spot: a validated callsign heard on a tracked carrier.
struct CwSpot {
    double      hz;    // carrier frequency
    float       wpm;   // decoder speed estimate at emit time
    float       snr;   // track's smoothed detection metric
    std::string call;  // the callsign
};

// CwTracker: the per-signal continuity layer over the CW skimmer's per-window
// detection.  A crowded band (cwtest2.dat: ~20 keyed 17m carriers) presents the
// same set of carriers window after window; decoding each window independently
// re-emits and cross-confirms unrelated bins.  The tracker instead keeps one
// Track per carrier: it associates each window's detection peaks to existing
// tracks (following small carrier drift), births tracks for genuinely new peaks,
// expires idle ones, and confirms a callsign only when the SAME carrier decodes
// it across >=confirm windows.  Pure CPU / host-only so it is unit-testable
// (test_cw_track) independent of the CUDA ring.
class CwTracker {
public:
    struct Config {
        float    snr_thresh = 12.0f;  // per-bin detection gate (absolute floor)
        float    adapt_k    = 0.0f;   // >0: gate = max(snr_thresh, median + k·MAD of the
                                      //     per-window metric) so it floats with band noise
        int      drift_bins = 2;      // ± bins a track may wander between windows (~100 Hz)
        int      merge_bins = 2;      // don't birth a track within this of an existing one
        int      confirm    = 2;      // windows a call must repeat on a track before emit
        uint64_t ttl_slots  = 60*50;  // a track dies after this many idle slots (~60 s)
    };

    CwTracker() = default;
    explicit CwTracker(Config cfg) : cfg_(cfg) {}

    // Advance one detection window.  `snr` holds the per-bin metric over [0,nbins);
    // `decode(bin)` returns the decoded text for a carrier bin, `wpm(bin)` its speed,
    // `binToHz(bin)` its RF frequency (<=0 = off-band, skip).  Returns the spots newly
    // confirmed this window.  `wi` is the ring write index (monotonic slot counter).
    std::vector<CwSpot> step(uint64_t wi, const float* snr, int nbins,
                             const std::function<std::string(int)>& decode,
                             const std::function<float(int)>& wpm,
                             const std::function<double(int)>& binToHz);

    int liveTracks() const { return (int)tracks_.size(); }

private:
    struct CallStat { int count = 0; uint64_t last_wi = 0; };  // times decoded + last window
    struct Track {
        int      bin;           // current center bin (follows drift)
        float    snr_ema;       // smoothed detection metric
        uint64_t born_wi;
        uint64_t last_wi;       // last window this track was matched
        std::unordered_map<std::string, CallStat> call_cnt;  // callsign -> {count, last_wi}
        std::unordered_set<std::string>           emitted;   // already spotted
    };

    Config cfg_{};
    std::vector<Track> tracks_;
};

} // namespace hf
} // namespace gm
