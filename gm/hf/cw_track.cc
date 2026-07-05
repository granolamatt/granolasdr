#include "gm/hf/cw_track.h"

#include <algorithm>

namespace gm {
namespace hf {

bool looksLikeCall(const std::string& s) {
    const int n = (int)s.size();
    if (n < 3 || n > 8) return false;
    int lastDigit = -1;
    for (int i = 0; i < n; ++i) {
        char c = s[i];
        bool up = (c >= 'A' && c <= 'Z'), dg = (c >= '0' && c <= '9');
        if (!up && !dg) return false;
        if (dg) lastDigit = i;
    }
    if (lastDigit < 1 || lastDigit > 3) return false;   // area digit after a 1-3 char prefix
    if (!(s[0] >= 'A' && s[0] <= 'Z')) return false;     // callsigns start with a letter
    const int suffix = n - 1 - lastDigit;                // letters after the last digit
    if (suffix < 1 || suffix > 4) return false;
    for (int i = lastDigit + 1; i < n; ++i)
        if (!(s[i] >= 'A' && s[i] <= 'Z')) return false;
    return true;
}

// True if a and b are within Levenshtein distance 1 (equal, one substitution,
// or one insertion/deletion).  Used to fold misdecode variants of one carrier's
// callsign together: a strong signal decodes WC6DX / TC6DX / WC6DD across windows.
static bool editDistLE1(const std::string& a, const std::string& b) {
    const int la = (int)a.size(), lb = (int)b.size();
    if (std::abs(la - lb) > 1) return false;
    if (la == lb) {                                   // count substitutions
        int diff = 0;
        for (int i = 0; i < la; ++i) if (a[i] != b[i] && ++diff > 1) return false;
        return true;
    }
    const std::string& s = (la < lb) ? a : b;         // shorter
    const std::string& t = (la < lb) ? b : a;         // longer (one extra char)
    int i = 0, j = 0; bool skipped = false;           // align with one deletion in t
    while (i < (int)s.size() && j < (int)t.size()) {
        if (s[i] == t[j]) { ++i; ++j; }
        else { if (skipped) return false; skipped = true; ++j; }
    }
    return true;
}

// Split on spaces into candidate tokens.
static std::vector<std::string> tokens(const std::string& txt) {
    std::vector<std::string> out;
    for (size_t p = 0; p < txt.size(); ) {
        size_t q = txt.find(' ', p);
        size_t end = (q == std::string::npos) ? txt.size() : q;
        if (end > p) out.emplace_back(txt.substr(p, end - p));
        p = (q == std::string::npos) ? txt.size() : q + 1;
    }
    return out;
}

std::vector<CwSpot> CwTracker::step(uint64_t wi, const float* snr, int nbins,
                                    const std::function<std::string(int)>& decode,
                                    const std::function<float(int)>& wpm,
                                    const std::function<double(int)>& binToHz) {
    // 1. Detection peaks: local maxima over ±2 bins above the gate.  A carrier's
    //    spectral shoulders don't out-peak its center, so one carrier => one peak.
    struct Peak { int bin; float s; };
    std::vector<Peak> peaks;
    for (int bin = 2; bin < nbins - 2; ++bin) {
        const float s = snr[bin];
        if (s < cfg_.snr_thresh) continue;
        if (s < snr[bin-1] || s < snr[bin+1] || s < snr[bin-2] || s < snr[bin+2]) continue;
        peaks.push_back({bin, s});
    }
    // Strongest first so a strong carrier claims its track before a weak neighbor.
    std::sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b){ return a.s > b.s; });

    // 2. Associate: greedy nearest-track within drift_bins.  A peak with no track
    //    in range but also not shadowing an existing track (within merge_bins) is
    //    a new carrier.  A peak within merge_bins of an already-claimed track is
    //    that carrier's sideband and is dropped.
    std::vector<char> claimed(tracks_.size(), 0);
    for (const Peak& p : peaks) {
        int best = -1, bestd = cfg_.drift_bins + 1;
        for (size_t t = 0; t < tracks_.size(); ++t) {
            if (claimed[t]) continue;
            int d = std::abs(tracks_[t].bin - p.bin);
            if (d <= cfg_.drift_bins && d < bestd) { best = (int)t; bestd = d; }
        }
        if (best >= 0) {
            Track& t = tracks_[best];
            t.bin = p.bin;                                   // follow carrier drift
            t.snr_ema = 0.7f * t.snr_ema + 0.3f * p.s;
            t.last_wi = wi;
            claimed[best] = 1;
            continue;
        }
        // No track to attach to: birth one unless a nearby track already exists
        // (claimed or not) that this peak is merely a shoulder of.
        bool shadowed = false;
        for (const Track& t : tracks_)
            if (std::abs(t.bin - p.bin) <= cfg_.merge_bins) { shadowed = true; break; }
        if (shadowed) continue;
        tracks_.push_back(Track{p.bin, p.s, wi, wi, {}, {}});
        claimed.push_back(1);
    }

    // 3. Expire idle tracks.
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                      [&](const Track& t){ return wi - t.last_wi > cfg_.ttl_slots; }),
                  tracks_.end());

    // 4. Decode each track active this window; confirm callsigns per-carrier.
    std::vector<CwSpot> spots;
    for (Track& t : tracks_) {
        if (t.last_wi != wi) continue;                       // not seen this window
        const double hz = binToHz(t.bin);
        if (hz <= 0.0) continue;
        const std::string txt = decode(t.bin);
        // Unique valid calls in THIS window's decode (a call repeated within one
        // window is one sighting, not proof — confirmation must span windows).
        std::unordered_set<std::string> seen_now;
        for (const std::string& tok : tokens(txt))
            if (looksLikeCall(tok)) seen_now.insert(tok);
        for (const std::string& call : seen_now) ++t.call_cnt[call];

        // Emit newly-confirmed calls, folding misdecode variants of one carrier's
        // callsign: suppress a candidate within edit-distance 1 of one already
        // emitted here, and defer to a not-yet-emitted variant with more sightings
        // so the most-decoded (usually correct) spelling wins.
        for (const std::string& call : seen_now) {
            if (t.call_cnt[call] < cfg_.confirm || t.emitted.count(call)) continue;

            bool is_variant = false;
            for (const std::string& e : t.emitted) {
                // Prefix fragment (WC6D<-WC6DX) or a near-miss respelling.
                if ((e.size() > call.size() && e.compare(0, call.size(), call) == 0) ||
                    editDistLE1(e, call)) { is_variant = true; break; }
            }
            if (is_variant) continue;

            bool defer = false;                          // a stronger sibling not yet out?
            for (const auto& kv : t.call_cnt)
                if (kv.first != call && kv.second > t.call_cnt[call] &&
                    !t.emitted.count(kv.first) && editDistLE1(kv.first, call)) { defer = true; break; }
            if (defer) continue;

            t.emitted.insert(call);
            spots.push_back(CwSpot{hz, wpm(t.bin), t.snr_ema, call});
        }
    }
    return spots;
}

} // namespace hf
} // namespace gm
