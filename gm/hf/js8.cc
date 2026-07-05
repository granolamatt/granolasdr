#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gm/hf/js8.h"
#include "gm/hf/jsc.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/band_map.h"
#include "ft8_lib/ft8/constants.h"
#include "gm/cuda/FT8Cuda.h"       // ContScanResult, CONT_CAND_MAX
#include "gm/cuda/JS8Cuda.h"
#include "gm/cuda/JS8Osd.h"        // js8_osd_decode

// ---- CRC-12: poly 0xC06, augmented, XOR key 42 -----------------------------
// Matches JS8Call's boost::augmented_crc<12,0xC06>(data,11) ^ 42.
static uint16_t crc12_augmented(const uint8_t* data, int nbytes) {
    uint16_t crc = 0;
    for (int i = 0; i < nbytes; ++i) {
        for (int b = 7; b >= 0; --b) {
            int inp = (data[i] >> b) & 1;
            int quotient = (crc >> 11) & 1;
            crc = ((crc << 1) | inp) & 0xFFF;
            if (quotient) crc ^= 0xC06;
        }
    }
    return crc ^ 42;
}

// decoded_bits[0..86]: the K=87 info bits. Returns true if CRC matches.
static bool checkCRC12(const uint8_t* decoded_bits) {
    uint8_t packed[11] = {};
    for (int i = 0; i < 87; ++i)
        if (decoded_bits[i])
            packed[i / 8] |= (uint8_t)(1 << (7 - (i & 7)));

    uint16_t received_crc = ((uint16_t)(packed[9] & 0x1F) << 7) | (packed[10] >> 1);
    packed[9] &= 0xE0;
    packed[10] = 0x00;
    return received_crc == crc12_augmented(packed, 11);
}

// ---- Bit utilities ----------------------------------------------------------

// Extract n bits starting at offset from a bit array (MSB first).
static uint64_t bits_to_int(const uint8_t* bits, int offset, int n) {
    uint64_t v = 0;
    for (int i = 0; i < n; i++)
        v = (v << 1) | (bits[offset + i] & 1);
    return v;
}

// ---- JS8 message unpacking --------------------------------------------------

// Alphabet for JS8Call callsign encoding: 0-9, A-Z, space, /, @  (39 chars)
static const char kCallAlpha[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ /@";

// nbasecall = 37 * 36 * 10 * 27 * 27 * 27 = 262177560
// Values at or above nbasecall are special group/system callsigns.
static const uint32_t kNBasecall = 262177560u;

// Group callsigns mapped as kNBasecall + index (1-based).
static const char* const kBasecalls[] = {
    nullptr,        // 0
    "<....>",       // 1
    "@ALLCALL",     // 2
    "@JS8NET",      // 3
    "@DX/NA",       // 4
    "@DX/SA",       // 5
    "@DX/EU",       // 6
    "@DX/AS",       // 7
    "@DX/AF",       // 8
    "@DX/OC",       // 9
    "@DX/AN",       // 10
    "@REGION/1",    // 11
    "@REGION/2",    // 12
    "@REGION/3",    // 13
    "@GROUP/0",     // 14
    "@GROUP/1",     // 15
    "@GROUP/2",     // 16
    "@GROUP/3",     // 17
    "@GROUP/4",     // 18
    "@GROUP/5",     // 19
    "@GROUP/6",     // 20
    "@GROUP/7",     // 21
    "@GROUP/8",     // 22
    "@GROUP/9",     // 23
    "@COMMAND",     // 24
    "@CONTROL",     // 25
    "@NET",         // 26
    "@NTS",         // 27
    "@RESERVE/0",   // 28
    "@RESERVE/1",   // 29
    "@RESERVE/2",   // 30
    "@RESERVE/3",   // 31
    "@RESERVE/4",   // 32
    "@APRSIS",      // 33
    "@RAGCHEW",     // 34
    "@JS8",         // 35
    "@EMCOMM",      // 36
    "@ARES",        // 37
    "@MARS",        // 38
    "@AMRRON",      // 39
    "@RACES",       // 40
    "@RAYNET",      // 41
    "@RADAR",       // 42
    "@SKYWARN",     // 43
    "@CQ",          // 44
    "@HB",          // 45
    "@QSO",         // 46
    "@QSOPARTY",    // 47
};
static const int kNumBasecalls = (int)(sizeof(kBasecalls) / sizeof(kBasecalls[0]));

// Directed commands indexed 0..31 (JS8Call directed_cmds table).
static const char* const kDirCmds[32] = {
    " SNR?",          // 0
    " DIT DIT",       // 1
    " NACK",          // 2
    " HEARING?",      // 3
    " GRID?",         // 4
    ">",              // 5
    " STATUS?",       // 6
    " STATUS",        // 7
    " HEARING",       // 8
    " MSG",           // 9
    " MSG TO:",       // 10
    " QUERY",         // 11
    " QUERY MSGS",    // 12
    " QUERY CALL",    // 13
    " ACK",           // 14
    " GRID",          // 15
    " INFO?",         // 16
    " INFO",          // 17
    " FB",            // 18
    " HW CPY?",       // 19
    " SK",            // 20
    " RR",            // 21
    " QSL?",          // 22
    " QSL",           // 23
    " CMD",           // 24
    " SNR",           // 25
    " NO",            // 26
    " YES",           // 27
    " 73",            // 28
    " HEARTBEAT SNR", // 29
    " AGN?",          // 30
    " ",              // 31 (freetext)
};

// Unpack a 28-bit packed value to a standard callsign (e.g. "KF0RRR").
// Uses the same mixed-radix scheme as JS8Call's Varicode::unpackCallsign.
static std::string unpackCallsign28(uint32_t v) {
    if (v >= kNBasecall) {
        uint32_t idx = v - kNBasecall;
        if (idx > 0 && (int)idx < kNumBasecalls && kBasecalls[idx])
            return kBasecalls[idx];
        return "<unknown>";
    }
    char w[7] = {};
    w[5] = kCallAlpha[v % 27 + 10]; v /= 27;
    w[4] = kCallAlpha[v % 27 + 10]; v /= 27;
    w[3] = kCallAlpha[v % 27 + 10]; v /= 27;
    w[2] = kCallAlpha[v % 10];      v /= 10;
    w[1] = kCallAlpha[v % 36];      v /= 36;
    w[0] = kCallAlpha[v % 37];
    // Handle 3X → 3DA0 prefix (rare, skip for now)
    std::string s(w, 6);
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

// Unpack a 50-bit packed value to a compound callsign (e.g. "KF0RRR/P").
// Uses JS8Call's Varicode::unpackAlphaNumeric50.
static std::string unpackCallsign50(uint64_t v) {
    char w[12] = {};
    w[10] = kCallAlpha[v % 38]; v /= 38;
    w[9]  = kCallAlpha[v % 38]; v /= 38;
    w[8]  = kCallAlpha[v % 38]; v /= 38;
    w[7]  = (v % 2) ? '/' : ' '; v /= 2;
    w[6]  = kCallAlpha[v % 38]; v /= 38;
    w[5]  = kCallAlpha[v % 38]; v /= 38;
    w[4]  = kCallAlpha[v % 38]; v /= 38;
    w[3]  = (v % 2) ? '/' : ' '; v /= 2;
    w[2]  = kCallAlpha[v % 38]; v /= 38;
    w[1]  = kCallAlpha[v % 38]; v /= 38;
    w[0]  = kCallAlpha[v % 39];
    std::string s;
    s.reserve(11);
    for (int i = 0; i < 11; i++)
        if (w[i] != ' ') s += w[i];
    return s;
}

// Convert a 16-bit packed grid value to a 4-char Maidenhead locator.
// nbasegrid = 180*180 = 32400; values above that are not grids.
static std::string unpackGrid(uint32_t v) {
    if (v >= 32400u) return "";
    float dlat  = (float)(v % 180u) - 90.0f;
    float dlong = (float)((v / 180u) * 2u) + 2.0f - 180.0f;  // integer div intentional
    if (dlong < -180.0f) dlong += 360.0f;
    if (dlong >  180.0f) dlong -= 360.0f;
    int nlong = (int)(60.0f * (180.0f - dlong) / 5.0f);
    int nlat  = (int)(60.0f * (dlat + 90.0f)   / 2.5f);
    char g[5] = {};
    g[0] = (char)('A' + nlong / 240);
    g[1] = (char)('A' + nlat  / 240);
    g[2] = (char)('0' + (nlong % 240) / 24);
    g[3] = (char)('0' + (nlat  % 240) / 24);
    return std::string(g, 4);
}

// ---- Huffman decoder for old-format data frames ----------------------------
// Table from JS8Call's Varicode::defaultHuffTable().
static const struct { char ch; const char* code; } kHuffTable[] = {
    {' ', "01"},      {'E', "100"},     {'T', "1101"},    {'A', "0011"},
    {'O', "11111"},   {'I', "11100"},   {'N', "10111"},   {'S', "10100"},
    {'H', "00011"},   {'R', "00000"},   {'D', "111011"},  {'L', "110011"},
    {'C', "110001"},  {'U', "101101"},  {'M', "101011"},  {'W', "001011"},
    {'F', "001001"},  {'G', "000101"},  {'Y', "000011"},  {'P', "1111011"},
    {'B', "1111001"}, {'.', "1110100"}, {'V', "1100101"}, {'K', "1100100"},
    {'-', "1100001"}, {'+', "1100000"}, {'?', "1011001"}, {'!', "1011000"},
    {'"', "1010101"}, {'X', "1010100"}, {'0', "0010101"}, {'J', "0010100"},
    {'1', "0010001"}, {'Q', "0010000"}, {'2', "0001001"}, {'Z', "0001000"},
    {'3', "0000101"}, {'5', "0000100"}, {'4', "11110101"},{'9', "11110100"},
    {'8', "11110001"},{'6', "11110000"},{'7', "11101011"},{'/', "11101010"},
};
static const int kHuffTableSize = (int)(sizeof(kHuffTable) / sizeof(kHuffTable[0]));

static std::string huffDecode(const uint8_t* bits, int nbits) {
    std::string result;
    int pos = 0;
    while (pos < nbits) {
        bool found = false;
        for (int k = 0; k < kHuffTableSize; k++) {
            const char* code = kHuffTable[k].code;
            int codelen = (int)strlen(code);
            if (pos + codelen > nbits) continue;
            bool match = true;
            for (int i = 0; i < codelen; i++) {
                if ((bits[pos + i] & 1) != (uint8_t)(code[i] - '0')) {
                    match = false;
                    break;
                }
            }
            if (match) {
                result += kHuffTable[k].ch;
                pos += codelen;
                found = true;
                break;
            }
        }
        if (!found) break;
    }
    return result;
}

// ---- Raw 12-char extraction (kept for dedup key usage) ---------------------
static const char kJs8Alphabet[] =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+";

static std::string extractMessage12(const uint8_t* info) {
    char out[13] = {};
    for (int i = 0; i < 12; ++i) {
        int word = 0;
        for (int b = 0; b < 6; ++b)
            word = (word << 1) | (info[i * 6 + b] & 1);
        if (word > 63) return {};
        out[i] = kJs8Alphabet[word];
    }
    return std::string(out, 12);
}

// ---- Main message decoder ---------------------------------------------------
// Decodes the 87 info bits into a human-readable JS8 message string.
// Also extracts the "from" callsign into *from_call if non-null.
// Returns the raw 12-char string prefixed with "[raw]" if unable to decode.
static std::string decodeJs8Message(const uint8_t* info, std::string* from_call) {
    // Bits 72-74: JS8 submode (i3bit). JS8CallData=4 → fast data frame: all
    // 72 bits are payload (no packed_flag header), flagged here not in bits[0].
    int i3bit = (int)bits_to_int(info, 72, 3);
    if (i3bit & 4) {
        // Fast-data: all 72 bits are JSC content; last 0 bit is end-of-data marker.
        int last_zero = 71;
        while (last_zero > 0 && info[last_zero]) last_zero--;
        std::string text = jsc_decompress(info, last_zero);
        return text.empty() ? "[data]" : text;
    }

    // Bits 0-2: frame sub-type (packed_flag).
    // Values 4-7 (bits[0]=1) are old-format data frames (deprecated but seen).
    // Padding scheme: after content, one 0 pad bit then 1s. lastIndexOf(0) finds it.
    int packed_flag = (int)bits_to_int(info, 0, 3);
    if (packed_flag >= 4) {
        bool compressed = (info[1] & 1) != 0;
        // Find the padding marker: last 0 bit in info[1..71]
        int K = -1;
        for (int j = 71; j >= 1; j--) {
            if ((info[j] & 1) == 0) { K = j; break; }
        }
        if (K <= 1 || K - 2 <= 0) return "[data]";
        if (compressed) {
            std::string text = jsc_decompress(info + 2, K - 2);
            return text.empty() ? "[data]" : text;
        }
        std::string text = huffDecode(info + 2, K - 2);
        return text.empty() ? "[data]" : text;
    }

    switch (packed_flag) {

    case 0: // FrameHeartbeat: [3][50][11],[5][3]
    case 1: // FrameCompound
    case 2: { // FrameCompoundDirected
        uint64_t packed_cs  = bits_to_int(info, 3, 50);
        uint32_t packed_11  = (uint32_t)bits_to_int(info, 53, 11);
        uint32_t packed_5   = (uint32_t)bits_to_int(info, 64, 5);
        uint32_t num        = (packed_11 << 5) | packed_5;

        std::string cs = unpackCallsign50(packed_cs);
        if (cs.empty()) return extractMessage12(info);

        if (from_call) *from_call = cs;

        if (packed_flag == 0) { // Heartbeat
            bool isAlt       = (num >> 15) & 1;
            uint32_t grid_v  = num & 0x7FFFu;
            std::string body = isAlt ? "@ALLCALL CQ" : "@HB HEARTBEAT";
            std::string grid = unpackGrid(grid_v);
            if (!grid.empty()) body += " " + grid;
            return cs + ": " + body + " ";
        }

        if (packed_flag == 1) { // Compound — callsign [+ grid or cmd]
            std::string extra;
            const uint32_t nbasegrid = 32400u;
            const uint32_t nusergrid = nbasegrid + 10;
            const uint32_t nmaxgrid  = (1u << 15) - 1;
            if (num <= nbasegrid) {
                extra = unpackGrid(num);
            } else if (num >= nusergrid && num < nmaxgrid) {
                // Cmd encoded above nusergrid: [1-bit snr_flag][1-bit type][6-bit num]
                uint32_t cmd_bits = num - nusergrid;
                uint8_t cmd_idx   = cmd_bits & 0x1F; // low 5 bits
                if (cmd_idx < 32) extra = kDirCmds[cmd_idx];
            }
            return cs + ":" + (extra.empty() ? "" : " " + extra) + " ";
        }

        // FrameCompoundDirected (2): callsign + directed cmd
        // Extra bits encode cmd in the same way as nusergrid range above
        {
            uint32_t cmd_bits = (num >= 32410u) ? num - 32410u : 0;
            uint8_t cmd_idx   = cmd_bits & 0x1F;
            std::string cmd   = (cmd_idx < 32) ? kDirCmds[cmd_idx] : "";
            return cs + ":" + cmd + " ";
        }
    }

    case 3: { // FrameDirected: [3][28][28][5],[2][6]
        uint32_t packed_from = (uint32_t)bits_to_int(info,  3, 28);
        uint32_t packed_to   = (uint32_t)bits_to_int(info, 31, 28);
        uint8_t  packed_cmd  = (uint8_t) bits_to_int(info, 59,  5);
        bool portable_from   = (info[64] & 1) != 0;
        bool portable_to     = (info[65] & 1) != 0;
        uint8_t  extra       = (uint8_t) bits_to_int(info, 66,  6);

        std::string from = unpackCallsign28(packed_from);
        std::string to   = unpackCallsign28(packed_to);
        if (portable_from) from += "/P";
        if (portable_to)   to   += "/P";

        if (from_call) *from_call = from;

        uint8_t cmd_idx = packed_cmd % 32;
        std::string cmd = kDirCmds[cmd_idx];

        std::string result = from + ": " + to + cmd;
        if (extra != 0) {
            // SNR commands (25, 29) show signed dB; others show raw number
            int num = (int)extra - 31;
            char nbuf[16];
            if (cmd_idx == 25 || cmd_idx == 29)
                snprintf(nbuf, sizeof(nbuf), " %+d", num);
            else
                snprintf(nbuf, sizeof(nbuf), " %d", num);
            result += nbuf;
        }
        return result + " ";
    }

    default:
        return extractMessage12(info);
    }
}

// ---- JS8 class --------------------------------------------------------------

namespace gm {
namespace hf {

JS8::JS8(gm::cuda::JS8CudaBase* js8cuda, int zmq_port,
         float symbol_period, float cycle_secs, int time_osr, int rfft_size,
         const char* mode_name, int wsdict_port)
    : js8cuda_(js8cuda), zmq_port_(zmq_port),
      symbol_period_(symbol_period), cycle_secs_(cycle_secs),
      time_osr_(time_osr), rfft_size_(rfft_size), mode_name_(mode_name),
      zmq_ctx_(1),
      zmq_pub_(zmq_ctx_, zmq::socket_type::pub)
{
    if (zmq_port_ > 0) {
        char endpoint[64];
        snprintf(endpoint, sizeof(endpoint), "tcp://localhost:%d", zmq_port_);
        zmq_pub_.connect(endpoint);
        printf("[%s] ZMQ publisher → %s  topic=js8/decode\n", mode_name_, endpoint);
    }

    if (wsdict_port > 0) {
        // Build wsdict key from mode name: "JS8" → "js8", "JS8 Fast" → "js8fast"
        std::string key_part = mode_name_;
        for (char& c : key_part) c = (char)tolower((unsigned char)c);
        key_part.erase(std::remove(key_part.begin(), key_part.end(), ' '), key_part.end());
        wsdict_key_ = "granolasdr:" + key_part + ":heard:";
        try {
            ws_client_ = std::make_unique<WsDictClient>("127.0.0.1", wsdict_port);
            printf("[%s] wsdict → port=%d  key_prefix=%s\n", mode_name_, wsdict_port, wsdict_key_.c_str());
        } catch (const std::exception& e) {
            fprintf(stderr, "[%s] wsdict connect failed: %s\n", mode_name_, e.what());
        }
    }

    llr_capture_.openIfEnabled("JS8_CAPTURE_LLR", "js8", mode_name_);

    js8cuda->setDecodeCallback([this](gm::cuda::ContScanResult& r) {
        decodeAndPublishContinuous(r);
    });
    js8cuda->start();
}

JS8::~JS8()
{
    refine_run_.store(false);
    refine_cv_.notify_all();
    if (refine_thread_.joinable()) refine_thread_.join();
}

void JS8::publishDecoded(const char* text, const char* from_call, float freq_hz,
                         float snr, double unix_time, float time_offset)
{
    printf("[%s] DECODED: %-24s  freq=%7.0f Hz  snr=%+5.1f  offset=%+.3fs\n",
           mode_name_, text, (double)freq_hz, (double)snr, (double)time_offset);

    nlohmann::json v;
    v["text"]   = text;
    v["freq"]   = (double)freq_hz;
    v["snr"]    = (double)snr;
    v["unix"]   = unix_time;
    v["offset"] = (double)time_offset;
    v["mode"]   = mode_name_;
    if (from_call && from_call[0]) v["call"] = from_call;
    // dump() with error_handler_t::replace ensures valid UTF-8 even if the
    // decoder emits a false-positive with non-ASCII bytes in the message text.
    std::string json_str = v.dump(-1, ' ', false,
                                  nlohmann::json::error_handler_t::replace);
    {
        std::lock_guard<std::mutex> lk(zmq_mutex_);
        if (zmq_port_ > 0) {
            zmq::message_t topic("js8/decode", 10);
            zmq::message_t payload(json_str.data(), json_str.size());
            zmq_pub_.send(topic, zmq::send_flags::sndmore);
            auto result = zmq_pub_.send(payload, zmq::send_flags::dontwait);
            if (!result)
                fprintf(stderr, "[%s] ZMQ send queue full, dropped: %s\n", mode_name_, text);
        }
    }
    if (ws_client_ && from_call && from_call[0]) {
        std::lock_guard<std::mutex> lk(ws_mutex_);
        try {
            nlohmann::json v_ws = v;
            // One key per callsign; '/' → '-' so key doesn't contain glob separator.
            // 900s (15 min) TTL matches the display window and survives page refresh.
            std::string call_key(from_call);
            std::replace(call_key.begin(), call_key.end(), '/', '-');
            ws_client_->set_with_ttl(wsdict_key_ + call_key, v_ws, 900000);
        } catch (const std::exception& e) {
            fprintf(stderr, "[%s] wsdict publish: %s\n", mode_name_, e.what());
            ws_client_.reset();
        }
    }
}

void JS8::decodeAndPublishContinuous(gm::cuda::ContScanResult& r)
{
    uint32_t n = std::min(*r.count, gm::cuda::CONT_CAND_MAX);
    if (n == 0) return;

    auto now = std::chrono::system_clock::now();
    double unix_now = std::chrono::duration<double>(now.time_since_epoch()).count();

    {
        std::lock_guard<std::mutex> lk(pub_mu_);   // shared with the refine worker
        uint64_t epoch = (uint64_t)(unix_now / (double)cycle_secs_);
        if (epoch != last_epoch_) {
            seen_this_epoch_.clear();
            last_epoch_ = epoch;
        }
    }

    // Frequency bins that produced a valid decode this scan (pass or osd); used to
    // keep those bins out of the "fail" training set.
    std::unordered_set<int32_t> decoded_bins;

    // Pass 1: candidates the GPU star-sum SP decoder converged on.
    for (uint32_t i = 0; i < n; ++i) {
        if (!r.parity[i]) continue;
        const float*   llr  = r.log174 + (size_t)i * kFtxLdpcN;
        const uint8_t* info = r.x_hat  + (size_t)i * kFtxLdpcN + 87;
        if (publishCandidate(r, i, info, llr, unix_now, "pass", nullptr))
            decoded_bins.insert(r.fo[i]);
    }

    if (!osd_enable_) return;

    // Pass 2: OSD fallback on the strongest SP failures.  Gate hard: dedup by
    // frequency bin (keep best sync score), drop anything below the score floor,
    // sort by score, cap per cycle.  This gating — not CRC alone — is what keeps
    // the OSD-on-noise false-accept rate (~0.045%/attempt) negligible.  The same
    // gated set defines the "fail" training label: strong candidates we tried.
    std::unordered_map<int32_t, uint32_t> best_by_bin;   // fo bin -> candidate idx
    for (uint32_t i = 0; i < n; ++i) {
        if (r.parity[i]) continue;
        if ((float)r.score[i] < osd_score_floor_) continue;
        auto it = best_by_bin.find(r.fo[i]);
        if (it == best_by_bin.end() || r.score[i] > r.score[it->second])
            best_by_bin[r.fo[i]] = i;
    }

    std::vector<uint32_t> cands;
    cands.reserve(best_by_bin.size());
    for (const auto& kv : best_by_bin) cands.push_back(kv.second);
    std::sort(cands.begin(), cands.end(),
              [&](uint32_t a, uint32_t b) { return r.score[a] > r.score[b]; });
    if ((int)cands.size() > osd_max_per_cycle_) cands.resize(osd_max_per_cycle_);

    // Refine is capped per scan; a transmission reappears across many scans, so
    // strong failures still get refined over time.
    const int kRefineMaxPerCycle = 2;
    int refine_used = 0;
    uint8_t osd_xhat[kFtxLdpcN];
    for (uint32_t i : cands) {
        const float* llr = r.log174 + (size_t)i * kFtxLdpcN;
        float dist = 0.0f;
        bool ok = js8_osd_decode(llr, osd_order_, osd_xhat, &dist) &&
                  (osd_soft_thresh_ <= 0.0f || dist <= osd_soft_thresh_) &&
                  publishCandidate(r, i, osd_xhat + 87, llr, unix_now, "osd", &dist);

        if (ok) {
            decoded_bins.insert(r.fo[i]);
            continue;
        }

        // Strong-Costas candidate the OSD gate tried but no decoder cracked.
        if (!decoded_bins.count(r.fo[i])) {
            float snr = (float)r.score[i] - 26.0f;
            std::lock_guard<std::mutex> lk(pub_mu_);
            if (llr_capture_.enabled())
                llr_capture_.write(mode_name_, "fail", llr, kFtxLdpcN,
                                   r.fo[i], (int)r.to[i], (int)r.ts[i], (int)r.score[i],
                                   (double)composite_bin_to_rf_hz(r.fo[i], rfft_size_),
                                   (double)snr, unix_now, /*text=*/nullptr, &dist);
        }

        // Refine fallback runs OFF this thread: enqueue the failed strong candidate
        // for the refine worker (bounded, drop-oldest) so the heavy FFT search
        // never blocks the scan callback.
        if (refine_enable_ && refine_used < kRefineMaxPerCycle) {
            ++refine_used;
            std::lock_guard<std::mutex> lk(refine_mu_);
            if (refine_q_.size() >= kRefineQueueMax) refine_q_.pop_front();
            refine_q_.push_back({r.fo[i], (int)r.to[i], (int)r.ts[i],
                                 r.snap_start, r.score[i], unix_now});
            refine_cv_.notify_one();
        }
    }
}

// Refine worker: drains the job queue, re-extracts LLRs from the retained complex
// frame with continuous alignment, retries OSD, and publishes any decode the
// discrete-grid path missed.  Runs on its own thread + CUDA stream.
void JS8::refineWorker()
{
    uint8_t osd_xhat[kFtxLdpcN];
    while (refine_run_.load(std::memory_order_acquire)) {
        RefineJob job;
        {
            std::unique_lock<std::mutex> lk(refine_mu_);
            refine_cv_.wait(lk, [&] {
                return !refine_run_.load(std::memory_order_acquire) || !refine_q_.empty();
            });
            if (!refine_run_.load(std::memory_order_acquire)) break;
            job = refine_q_.front();
            refine_q_.pop_front();
        }

        float rllr[kFtxLdpcN];
        float rdist = 0.0f;
        if (js8cuda_->refineCandidate(job.fo, job.to, job.snap_start, rllr) &&
            js8_osd_decode(rllr, osd_order_, osd_xhat, &rdist) &&
            (osd_soft_thresh_ <= 0.0f || rdist <= osd_soft_thresh_))
            publishCandidateCore(osd_xhat + 87, rllr, job.fo, job.to, job.ts,
                                 job.score, job.unix_now, "refine", &rdist);
    }
}

bool JS8::publishCandidate(gm::cuda::ContScanResult& r, uint32_t i,
                           const uint8_t* info, const float* llr, double unix_now,
                           const char* status, const float* osd_dist)
{
    return publishCandidateCore(info, llr, r.fo[i], (int)r.to[i], (int)r.ts[i],
                                r.score[i], unix_now, status, osd_dist);
}

bool JS8::publishCandidateCore(const uint8_t* info, const float* llr, int32_t fo,
                               int to, int ts, int16_t score, double unix_now,
                               const char* status, const float* osd_dist)
{
    std::string text, from_call;
    float freq_hz, time_sec, snr;
    {
        // The JS8 message decode (jsc dictionary) + dedup set + capture file are
        // all shared with the scan thread — serialize under pub_mu_.  publishDecoded
        // (network/stdout I/O) runs afterwards, outside the lock.
        std::lock_guard<std::mutex> lk(pub_mu_);
        if (!checkCRC12(info)) return false;
        if (extractMessage12(info).empty()) return false;

        text = decodeJs8Message(info, &from_call);

        // Suppress re-publishing the same message+frequency within one epoch.  Still
        // a real decode, so the caller marks the bin decoded — just don't re-publish.
        std::string dedup_key = text + "|" + std::to_string(fo);
        if (!seen_this_epoch_.insert(dedup_key).second) return true;

        freq_hz  = composite_bin_to_rf_hz(fo, rfft_size_);
        time_sec = (to + (float)ts / (float)time_osr_) * symbol_period_;
        snr      = (float)score - 26.0f;

        llr_capture_.write(mode_name_, status, llr, kFtxLdpcN, fo, to, ts, (int)score,
                           (double)freq_hz, (double)snr, unix_now, text.c_str(), osd_dist);
    }

    publishDecoded(text.c_str(), from_call.c_str(), freq_hz, snr, unix_now, time_sec);
    return true;
}

void JS8::setOsdConfig(bool enable, int order, float score_floor,
                       int max_per_cycle, float soft_thresh)
{
    osd_enable_        = enable;
    osd_order_         = order;
    osd_score_floor_   = score_floor;
    osd_max_per_cycle_ = max_per_cycle;
    osd_soft_thresh_   = soft_thresh;
}

} // namespace hf
} // namespace gm
