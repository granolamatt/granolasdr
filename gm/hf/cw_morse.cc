#include "gm/hf/cw_morse.h"
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gm {
namespace hf {

// Forward Morse table: element string (".-") -> character.
static const std::unordered_map<std::string, char>& morseTable() {
    static const std::unordered_map<std::string, char> t = {
        {".-",'A'},   {"-...",'B'}, {"-.-.",'C'}, {"-..",'D'},  {".",'E'},
        {"..-.",'F'}, {"--.",'G'},  {"....",'H'}, {"..",'I'},   {".---",'J'},
        {"-.-",'K'},  {".-..",'L'}, {"--",'M'},   {"-.",'N'},   {"---",'O'},
        {".--.",'P'}, {"--.-",'Q'}, {".-.",'R'},  {"...",'S'},  {"-",'T'},
        {"..-",'U'},  {"...-",'V'}, {".--",'W'},  {"-..-",'X'}, {"-.--",'Y'},
        {"--..",'Z'},
        {"-----",'0'},{".----",'1'},{"..---",'2'},{"...--",'3'},{"....-",'4'},
        {".....",'5'},{"-....",'6'},{"--...",'7'},{"---..",'8'},{"----.",'9'},
        {"-..-.",'/'},{"-...-",'='},{".-.-.",'+'},{"-....-",'-'},
        {".-.-.-",'.'},{"--..--",','},{"..--..",'?'},
    };
    return t;
}

CwMorse::CwMorse(float hop_ms, float init_wpm)
    : hop_ms_(hop_ms), init_dit_ms_(1200.0f / init_wpm) {
    reset();
}

void CwMorse::reset() {
    dit_ms_  = init_dit_ms_;
    nf_ = pk_ = 0.0f;
    have_env_ = false;
    key_ = false;
    run_ = 0;
    started_ = false;
    symbol_.clear();
    text_.clear();
}

void CwMorse::push(float mag) {
    if (!have_env_) { nf_ = pk_ = mag; have_env_ = true; }

    // Noise floor: fast attack down, very slow rise (so a long mark can't drag it up).
    if (mag < nf_) nf_ += (mag - nf_) * 0.25f;
    else           nf_ += (mag - nf_) * 0.0005f;
    // Peak: fast attack up, slow decay (holds through inter-element gaps).
    if (mag > pk_) pk_ += (mag - pk_) * 0.50f;
    else           pk_ += (mag - pk_) * 0.001f;

    const float span = pk_ - nf_;
    bool next = key_;
    // Only key when there is clear signal span (suppresses pure-noise chatter).
    if (span > 0.25f * pk_ && span > 1e-6f) {
        const float t_hi = nf_ + 0.60f * span;
        const float t_lo = nf_ + 0.40f * span;
        if      (!key_ && mag >= t_hi) next = true;
        else if ( key_ && mag <= t_lo) next = false;
    }

    edge(next);
}

void CwMorse::edge(bool key_now) {
    if (key_now != key_) {
        const float ms = run_ * hop_ms_;
        if (key_)          onMark(ms);    // a mark just ended
        else if (started_) onSpace(ms);   // a space just ended (ignore the leading one)
        key_ = key_now;
        run_ = 1;
        if (key_) started_ = true;
    } else {
        run_++;
    }
}

void CwMorse::onMark(float ms) {
    if (ms <= 0.0f) return;
    const bool is_dit = ms < 2.0f * dit_ms_;
    symbol_ += is_dit ? '.' : '-';
    // Pull the dit estimate toward the observed unit length (dah = 3 units).
    const float target = is_dit ? ms : ms / 3.0f;
    dit_ms_ += (target - dit_ms_) * 0.15f;
    if (dit_ms_ < 15.0f)  dit_ms_ = 15.0f;   // 80 WPM ceiling
    if (dit_ms_ > 240.0f) dit_ms_ = 240.0f;  // 5  WPM floor
}

void CwMorse::onSpace(float ms) {
    if (symbol_.empty()) return;             // nothing pending (back-to-back spaces)
    // Geometric-mean boundaries between the canonical 1u / 3u / 7u gaps:
    //   element|char = sqrt(1*3)=1.73u,  char|word = sqrt(3*7)=4.58u.
    if (ms < 1.73f * dit_ms_) return;             // element gap: stay within the symbol
    if (ms < 4.58f * dit_ms_) flushSymbol(false); // char gap
    else                      flushSymbol(true);  // word gap
}

char CwMorse::symbolToChar() const {
    const auto& t = morseTable();
    auto it = t.find(symbol_);
    return it == t.end() ? '\0' : it->second;
}

void CwMorse::flushSymbol(bool add_word_space) {
    const char c = symbolToChar();
    if (c) text_ += c;
    symbol_.clear();
    if (add_word_space) text_ += ' ';
}

void CwMorse::flush() {
    if (key_ && run_ > 0) { onMark(run_ * hop_ms_); key_ = false; run_ = 0; }
    flushSymbol(false);
}

std::string CwMorse::take() {
    std::string s = text_;
    text_.clear();
    return s;
}

// Otsu's method: the magnitude threshold that maximizes between-class variance
// of a [noise | signal] bimodal histogram.  Robust to level, no startup ramp.
static float otsu(const float* x, int n) {
    float lo = x[0], hi = x[0];
    for (int i = 1; i < n; ++i) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
    if (hi - lo < 1e-9f) return hi + 1.0f;     // flat: never keys
    const int B = 256;
    long hist[B] = {0};
    const float scale = (B - 1) / (hi - lo);
    for (int i = 0; i < n; ++i) hist[(int)((x[i] - lo) * scale)]++;
    long total = n;
    double sum = 0; for (int b = 0; b < B; ++b) sum += (double)b * hist[b];
    double sumB = 0; long wB = 0; double best = -1; int thrB = 0;
    for (int b = 0; b < B; ++b) {
        wB += hist[b]; if (!wB) continue;
        long wF = total - wB; if (!wF) break;
        sumB += (double)b * hist[b];
        double mB = sumB / wB, mF = (sum - sumB) / wF;
        double v = (double)wB * wF * (mB - mF) * (mB - mF);
        if (v > best) { best = v; thrB = b; }
    }
    return lo + (thrB + 0.5f) / scale;
}

// Offline (two-pass) decode: robust threshold, then estimate the dit length
// from the actual mark-duration clusters BEFORE classifying — so a far-off
// seed can't corrupt the estimate the way a single-pass EMA does.
std::string CwMorse::decode(const float* env, int n) {
    reset();
    if (n <= 0) return "";

    // Smooth the envelope (centered 5-tap moving average ~ matched to keying
    // edges) so per-sample noise can't erode a short inter-element gap at the
    // 5 ms hop — the dominant clean-signal error before smoothing.
    std::vector<float> s(n);
    {
        const int W = 2;                 // half-width: 5-tap
        double acc = 0; int cnt = 0;
        for (int i = -W; i <= W && i < n; ++i) { acc += env[i < 0 ? 0 : i]; cnt++; }
        for (int i = 0; i < n; ++i) {
            s[i] = (float)(acc / cnt);
            int out = i - W, in = i + W + 1;
            if (in < n)  { acc += env[in];  cnt++; }
            if (out >= 0){ acc -= env[out]; cnt--; }
        }
    }
    // AGC: divide by a slow peak envelope (forward+backward max-hold, decay ~
    // 0.5 s) so slow QSB fades flatten out before a global threshold is applied.
    // Floor the envelope at a fraction of the global peak so pure-noise regions
    // are not amplified up to signal level.
    {
        float gmax = s[0];
        for (int i = 1; i < n; ++i) if (s[i] > gmax) gmax = s[i];
        const float floor = 0.15f * gmax;
        std::vector<float> pf(n);
        float p = s[0];
        for (int i = 0; i < n; ++i) { p = std::max(s[i], p * 0.99f); pf[i] = p; }
        p = s[n - 1];
        for (int i = n - 1; i >= 0; --i) { p = std::max(s[i], p * 0.99f); if (p > pf[i]) pf[i] = p; }
        for (int i = 0; i < n; ++i) s[i] = s[i] / std::max(pf[i], floor);
    }
    const float* e = s.data();

    // Pass 1: Otsu threshold + hysteresis -> run-length list of (key, ms).
    float lo = e[0], hi = e[0];
    for (int i = 1; i < n; ++i) { if (e[i] < lo) lo = e[i]; if (e[i] > hi) hi = e[i]; }
    const float thr  = otsu(e, n);
    const float hyst = 0.08f * (hi - lo);
    const float t_hi = thr + hyst, t_lo = thr - hyst;

    std::vector<std::pair<bool, float>> runs;   // (key-down?, duration ms)
    bool k = false; int run = 0; bool seen_mark = false;
    for (int i = 0; i < n; ++i) {
        bool next = k;
        if      (e[i] >= t_hi) next = true;
        else if (e[i] <= t_lo) next = false;
        if (next != k) {
            if (k || seen_mark) runs.push_back({k, run * hop_ms_}); // drop the leading space
            if (k) seen_mark = true;
            k = next; run = 1;
        } else run++;
    }
    if (k && run > 0) runs.push_back({true, run * hop_ms_});
    if (runs.empty()) return "";

    // Estimate the unit length: 2-means on mark durations -> smaller centroid = dit.
    std::vector<float> marks;
    for (auto& r : runs) if (r.first) marks.push_back(r.second);
    if (marks.empty()) return "";
    float c1 = marks[0], c2 = marks[0];
    for (float m : marks) { if (m < c1) c1 = m; if (m > c2) c2 = m; }
    for (int it = 0; it < 20 && c2 > c1; ++it) {
        double s1 = 0, s2 = 0; int n1 = 0, n2 = 0;
        const float mid = 0.5f * (c1 + c2);
        for (float m : marks) { if (m <= mid) { s1 += m; n1++; } else { s2 += m; n2++; } }
        if (n1) c1 = (float)(s1 / n1);
        if (n2) c2 = (float)(s2 / n2);
    }
    // If the clusters didn't separate (~all one element type), treat them as dits.
    // Mark dit|dah boundary = geometric mean of the 1u/3u centroids (sqrt(c1*c2)).
    const bool split = (c2 > 1.8f * c1);
    dit_ms_ = split ? c1 : 0.5f * (c1 + c2);
    const float mark_split = split ? std::sqrt(c1 * c2) : 1e9f;

    // Pass 2: replay runs through the element/gap state machine (fixed dit).
    bool first = true;
    for (auto& r : runs) {
        if (first && !r.first) continue;   // leading space guard
        first = false;
        if (r.first) {                     // mark
            symbol_ += (r.second < mark_split) ? '.' : '-';
        } else {                           // space
            onSpace(r.second);
        }
    }
    flushSymbol(false);
    return take();
}

float CwMorse::wpm() const { return 1200.0f / dit_ms_; }

} // namespace hf
} // namespace gm
