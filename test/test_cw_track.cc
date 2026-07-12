// test_cw_track.cc — Functional tests for the CW per-signal tracker.
//
// The tracker sits above per-window detection: it associates detection peaks to
// persistent per-carrier Tracks (following drift), suppresses sidebands, expires
// idle carriers, and confirms a callsign only when the SAME carrier decodes it
// across >=confirm windows.  These tests drive step() with synthetic snr[] arrays
// and decode/wpm/binToHz callbacks — no CUDA, no radio.
//
// Builds host-only: links gm/hf/cw_track.cc only.

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "gm/hf/cw_track.h"

using gm::hf::CwTracker;
using gm::hf::looksLikeCall;

static int g_fail = 0;
static void check(bool ok, const char* name, const std::string& got = "") {
    printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name,
           got.empty() ? "" : "  ", got.c_str());
    if (!ok) ++g_fail;
}

static const int NB = 512;
static double binHz(int bin) { return 18000000.0 + bin * 50.0; }  // 50 Hz/bin
static float  wpm20(int)     { return 20.0f; }

// Build a flat snr array with a peak (triangular shoulders) at each given bin.
static std::vector<float> snrWith(std::vector<std::pair<int,float>> peaks) {
    std::vector<float> s(NB, 0.0f);
    for (auto& p : peaks) {
        int b = p.first; float v = p.second;
        s[b]   = v;
        if (b-1 >= 0) s[b-1] = std::max(s[b-1], v*0.6f);
        if (b+1 <  NB) s[b+1] = std::max(s[b+1], v*0.6f);
        if (b-2 >= 0) s[b-2] = std::max(s[b-2], v*0.3f);
        if (b+2 <  NB) s[b+2] = std::max(s[b+2], v*0.3f);
    }
    return s;
}

// Collect every callsign emitted over `n` identical windows for a fixed peak set /
// decode map, advancing wi by STRIDE each window.
static std::vector<std::string> runWindows(
        CwTracker& tr, int n, const std::vector<float>& snr,
        const std::function<std::string(int)>& decode, int stride = 25) {
    std::vector<std::string> calls;
    for (int i = 0; i < n; ++i) {
        auto spots = tr.step((uint64_t)(i + 1) * stride, snr.data(), NB, decode, wpm20, binHz);
        for (auto& sp : spots) calls.push_back(sp.call);
    }
    return calls;
}

int main() {
    printf("test_cw_track — CW per-signal tracker\n\n");

    // ---- looksLikeCall shape gate --------------------------------------------
    check(looksLikeCall("WC6DX"),   "call WC6DX valid");
    check(looksLikeCall("KF0RRR"),  "call KF0RRR valid");
    check(looksLikeCall("W1AW"),    "call W1AW valid");
    check(!looksLikeCall("E"),      "reject E");
    check(!looksLikeCall("WC6"),    "reject WC6 (no suffix)");
    check(!looksLikeCall("ABCDE"),  "reject ABCDE (no digit)");
    check(!looksLikeCall("12AB"),   "reject 12AB (leading digit)");

    // ---- confirm gate: one call emits once after `confirm` windows ------------
    {
        CwTracker tr;
        auto snr = snrWith({{100, 30.0f}});
        auto dec = [](int){ return std::string("CQ WC6DX WC6DX"); };
        // window 1: seen once (not yet confirmed).  window 2: confirmed -> emit.
        auto w1 = tr.step(25, snr.data(), NB, dec, wpm20, binHz);
        check(w1.empty(), "no emit on first window (confirm=2)");
        auto w2 = tr.step(50, snr.data(), NB, dec, wpm20, binHz);
        check(w2.size() == 1 && w2[0].call == "WC6DX", "emit WC6DX on 2nd window",
              w2.empty() ? "(none)" : w2[0].call);
        auto w3 = tr.step(75, snr.data(), NB, dec, wpm20, binHz);
        check(w3.empty(), "no re-emit on 3rd window (dedup)");
        check(tr.liveTracks() == 1, "single track");
    }

    // ---- two well-separated carriers tracked independently -------------------
    {
        CwTracker tr;
        auto snr = snrWith({{100, 30.0f}, {200, 28.0f}});
        auto dec = [](int b){ return b < 150 ? std::string("W1AW") : std::string("K5XYZ"); };
        auto calls = runWindows(tr, 3, snr, dec);
        bool a = false, k = false;
        for (auto& c : calls) { if (c == "W1AW") a = true; if (c == "K5XYZ") k = true; }
        check(a && k, "both carriers spotted");
        check(calls.size() == 2, "each spotted exactly once",
              std::to_string(calls.size()) + " spots");
        check(tr.liveTracks() == 2, "two tracks");
    }

    // ---- sideband suppression: a shoulder peak births no second track --------
    {
        CwTracker tr;
        // One strong carrier at 100; its shoulder at 101 also clears the gate but
        // must not become its own track.
        std::vector<float> snr(NB, 0.0f);
        snr[100] = 30.0f; snr[101] = 25.0f; snr[99] = 18.0f; snr[102] = 14.0f;
        auto dec = [](int){ return std::string("N0CALL N0CALL"); };
        runWindows(tr, 2, snr, dec);
        check(tr.liveTracks() == 1, "shoulder did not birth a 2nd track",
              std::to_string(tr.liveTracks()) + " tracks");
    }

    // ---- drift: a carrier moving 1 bin/window stays one track ----------------
    {
        CwTracker tr;
        auto dec = [](int){ return std::string("VE3ABC VE3ABC"); };
        std::vector<std::string> calls;
        for (int i = 0; i < 4; ++i) {
            auto snr = snrWith({{100 + i, 30.0f}});   // drifts 100->103
            auto sp = tr.step((uint64_t)(i+1)*25, snr.data(), NB, dec, wpm20, binHz);
            for (auto& s : sp) calls.push_back(s.call);
        }
        check(tr.liveTracks() == 1, "drift stayed one track",
              std::to_string(tr.liveTracks()) + " tracks");
        check(calls.size() == 1 && calls[0] == "VE3ABC", "drifting carrier spotted once");
    }

    // ---- expiry then re-emit after TTL ---------------------------------------
    {
        CwTracker::Config c; c.ttl_slots = 100;      // short TTL for the test
        CwTracker tr(c);
        auto snr = snrWith({{100, 30.0f}});
        auto dec = [](int){ return std::string("AA1BB AA1BB"); };
        tr.step(25, snr.data(), NB, dec, wpm20, binHz);
        tr.step(50, snr.data(), NB, dec, wpm20, binHz);       // emit #1
        // Idle past TTL: advance wi with no peaks so the track dies.
        std::vector<float> quiet(NB, 0.0f);
        tr.step(200, quiet.data(), NB, dec, wpm20, binHz);
        check(tr.liveTracks() == 0, "track expired after idle TTL");
        // Carrier returns: new track, re-confirms, re-emits.
        tr.step(225, snr.data(), NB, dec, wpm20, binHz);
        auto again = tr.step(250, snr.data(), NB, dec, wpm20, binHz);
        check(again.size() == 1 && again[0].call == "AA1BB", "re-emit after TTL");
    }

    // ---- variant folding: one carrier's misdecodes collapse to one spot ------
    {
        CwTracker tr;
        auto snr = snrWith({{100, 30.0f}});
        // A strong carrier mostly decodes WC6DX but occasionally mis-reads it as
        // TC6DX / WC6DD (edit-distance 1).  Only the dominant spelling should emit.
        int w = 0;
        std::function<std::string(int)> dec = [&](int) -> std::string {
            static const char* seq[] = {"WC6DX","WC6DX","TC6DX","WC6DX","WC6DD","WC6DX"};
            return seq[(w) % 6];
        };
        std::vector<std::string> calls;
        for (w = 0; w < 6; ++w) {
            auto sp = tr.step((uint64_t)(w+1)*25, snr.data(), NB, dec, wpm20, binHz);
            for (auto& s : sp) calls.push_back(s.call);
        }
        check(calls.size() == 1 && calls[0] == "WC6DX",
              "variants folded to one spot (WC6DX)",
              calls.empty() ? "(none)" : (calls[0] + " x" + std::to_string(calls.size())));
    }

    // ---- two genuinely different calls on one carrier both emit --------------
    {
        CwTracker tr;
        auto snr = snrWith({{100, 30.0f}});
        // A QSO: two well-separated callsigns alternate on the same frequency.
        int w = 0;
        std::function<std::string(int)> dec = [&](int) -> std::string {
            return (w % 2) ? std::string("W1AW") : std::string("K5XYZ");
        };
        std::vector<std::string> calls;
        for (w = 0; w < 5; ++w) {
            auto sp = tr.step((uint64_t)(w+1)*25, snr.data(), NB, dec, wpm20, binHz);
            for (auto& s : sp) calls.push_back(s.call);
        }
        bool a = false, k = false;
        for (auto& c : calls) { if (c == "W1AW") a = true; if (c == "K5XYZ") k = true; }
        check(a && k && calls.size() == 2, "distinct calls on one carrier both spotted",
              std::to_string(calls.size()) + " spots");
    }

    // ---- CQ run into the call is stripped: "CQK4RO" -> spot "K4RO" -----------
    {
        CwTracker tr;
        auto snr = snrWith({{100, 30.0f}});
        auto calls = runWindows(tr, 3, snr, [](int){ return std::string("CQK4RO"); });
        check(calls.size() == 1 && calls[0] == "K4RO", "CQ prefix stripped to real call",
              calls.empty() ? "(none)" : calls[0]);
    }

    // ---- sparse misread on a carrier is suppressed; the dominant call spots ---
    {
        CwTracker tr;
        auto snr = snrWith({{100, 30.0f}});
        int w = 0;
        // KJ9C dominates; NF8M is an occasional QSB misread (reaches confirm but
        // stays < half KJ9C's count) -> relative-dominance must drop only NF8M.
        std::function<std::string(int)> dec = [&](int) -> std::string {
            return (w == 6 || w == 9) ? std::string("NF8M") : std::string("KJ9C");
        };
        std::vector<std::string> calls;
        for (w = 0; w < 10; ++w) {
            auto sp = tr.step((uint64_t)(w+1)*25, snr.data(), NB, dec, wpm20, binHz);
            for (auto& s : sp) calls.push_back(s.call);
        }
        bool kj = false, nf = false;
        for (auto& c : calls) { if (c == "KJ9C") kj = true; if (c == "NF8M") nf = true; }
        check(kj && !nf && calls.size() == 1, "sparse misread suppressed (KJ9C, not NF8M)",
              std::to_string(calls.size()) + " spots");
    }

    // ---- adaptive gate: k×median floats above the noise floor ----------------
    {
        CwTracker::Config cfg; cfg.snr_thresh = 0.0f; cfg.adapt_k = 2.5f;  // gate = 2.5×median
        CwTracker tr(cfg);
        std::vector<float> s(NB, 4.0f);                 // noise floor 4 -> median 4 -> gate 10
        auto put = [&](int b, float v){ s[b]=v; s[b-1]=s[b+1]=v*0.6f; s[b-2]=s[b+2]=v*0.3f; };
        put(100, 11.0f);                                // above gate 10 -> detected
        put(200,  9.0f);                                // below gate 10 -> rejected
        std::function<std::string(int)> dec = [&](int b){ return std::string(b < 150 ? "K1AAA" : "K2BBB"); };
        std::vector<std::string> calls;
        for (int w = 0; w < 3; ++w) {
            auto sp = tr.step((uint64_t)(w+1)*25, s.data(), NB, dec, wpm20, binHz);
            for (auto& x : sp) calls.push_back(x.call);
        }
        bool got1 = false, got2 = false;
        for (auto& c : calls) { if (c == "K1AAA") got1 = true; if (c == "K2BBB") got2 = true; }
        check(got1 && !got2, "adaptive gate passes >k*median, rejects below",
              std::to_string(calls.size()) + " spots");
    }

    // ---- crowded band: 20 carriers, all distinct, each once ------------------
    {
        CwTracker tr;
        std::vector<std::pair<int,float>> pk;
        for (int i = 0; i < 20; ++i) pk.push_back({20 + i*12, 25.0f + (i % 5)});  // 600 Hz apart
        auto snr = snrWith(pk);
        auto dec = [](int b){
            // each carrier decodes a unique valid call from its bin
            char c[8]; int idx = (b - 20) / 12;
            snprintf(c, sizeof c, "W%dAB%c", idx % 10, 'A' + (idx % 20));
            return std::string(c);
        };
        auto calls = runWindows(tr, 3, snr, dec);
        check(tr.liveTracks() == 20, "20 carriers -> 20 tracks",
              std::to_string(tr.liveTracks()) + " tracks");
        check((int)calls.size() == 20, "20 distinct spots",
              std::to_string(calls.size()) + " spots");
    }

    printf("\nPer-signal tracker: confirm gate, multi-carrier separation, sideband\n"
           "suppression, drift following, TTL expiry, 20-carrier crowd.\n");
    printf(g_fail ? "FAIL — %d failure(s)\n" : "OK — 0 failure(s)\n", g_fail);
    return g_fail ? 1 : 0;
}
