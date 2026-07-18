#include "gm/hf/cw_track.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

// Recover a callsign from a token that the decoder ran a procedural signal into,
// e.g. "CQK4RO" -> "K4RO" (a missed word gap after CQ). Only strips prefixes that
// are never a real callsign start, so a valid call is never mangled. Returns the
// call, or "" if the token isn't (or doesn't contain) a valid call.
static std::string extractCall(const std::string& tok) {
    // Strip a procedural prefix FIRST: "CQK4RO" spuriously passes looksLikeCall
    // as-is (fake 3-char prefix "CQK"), so we must prefer the stripped real call.
    // Only strips when the remainder is itself a valid call, so a genuine call is
    // never mangled (e.g. "DX1ABC" -> "1ABC" is invalid, so DX1ABC is kept).
    for (const std::string& p : {std::string("CQ"), std::string("QRZ"),
                                 std::string("TEST"), std::string("DX")}) {
        if (tok.size() > p.size() && tok.compare(0, p.size(), p) == 0) {
            std::string rest = tok.substr(p.size());
            if (looksLikeCall(rest)) return rest;
        }
    }
    if (looksLikeCall(tok)) return tok;
    return "";
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
    // Effective gate: the absolute floor (snr_thresh), optionally floated to a
    // noise-relative level (adapt_k × median of the per-bin metric) when adapt_k>0.
    // The median IS the noise floor (>99% of bins are noise; a handful of carriers
    // don't move it), so k·median tracks band conditions: it drops in a quiet band
    // (catching weak CW that a fixed 12 misses) and rises under QRN. A median-
    // MULTIPLE, not median+MAD — the p80-p50 metric is integer-valued so MAD is
    // often 0 and would collapse to the floor. On cwtest.dat (median≈3.7) k≈2.7
    // gives ≈10, the measured sweet spot before garbage climbs.
    float gate = cfg_.snr_thresh;
    if (cfg_.adapt_k > 0.0f && nbins > 8) {
        std::vector<float> m(snr, snr + nbins);
        const size_t mid = m.size() / 2;
        std::nth_element(m.begin(), m.begin() + mid, m.end());
        gate = std::max(gate, cfg_.adapt_k * m[mid]);
    }

    // 1. Detection peaks: local maxima over ±2 bins above the gate.  A carrier's
    //    spectral shoulders don't out-peak its center, so one carrier => one peak.
    struct Peak { int bin; float s; };
    std::vector<Peak> peaks;
    for (int bin = 2; bin < nbins - 2; ++bin) {
        const float s = snr[bin];
        if (s < gate) continue;
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
        for (const std::string& tok : tokens(txt)) {
            const std::string call = extractCall(tok);
            if (!call.empty()) seen_now.insert(call);
        }
        for (const std::string& call : seen_now) {
            auto& cs = t.call_cnt[call];
            ++cs.count;
            cs.last_wi = wi;
        }

        // Emit newly-confirmed calls with RELATIVE-DOMINANCE suppression: a QSB/
        // noise misread decodes to a DIFFERENT string on the same carrier only a
        // few times, far less than the true call (7036.8: KJ9C dominates, NF8M/
        // EM9C are sparse) — edit-distance folding can't catch those. So suppress
        // a call decoded < half as often as the carrier's dominant call.
        //
        // Dominance is over RECENT history only (calls last seen within ttl/4 ≈
        // 15 s), not lifetime counts: otherwise a station arriving on a long-lived
        // carrier must out-count the previous occupant's ENTIRE history to emit —
        // effectively never on a short QSO. A QRT station stops gatekeeping; a
        // QSB fade (still recent) rides through; two alternating stations both spot.
        const uint64_t recent = cfg_.ttl_slots / 4;
        for (const std::string& call : seen_now) {
            const int mine = t.call_cnt[call].count;
            if (mine < cfg_.confirm || t.emitted.count(call)) continue;

            int best = 0;
            for (const auto& kv : t.call_cnt)
                if (wi - kv.second.last_wi <= recent)
                    best = std::max(best, kv.second.count);
            if (mine * 2 < best) continue;               // sparse vs recently-dominant -> misread

            bool is_variant = false;
            for (const std::string& e : t.emitted) {
                // Prefix fragment (WC6D<-WC6DX) or a near-miss respelling.
                if ((e.size() > call.size() && e.compare(0, call.size(), call) == 0) ||
                    editDistLE1(e, call)) { is_variant = true; break; }
            }
            if (is_variant) continue;

            t.emitted.insert(call);
            spots.push_back(CwSpot{hz, wpm(t.bin), t.snr_ema, call});
        }
    }
    return spots;
}

} // namespace hf
} // namespace gm
