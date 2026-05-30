#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>

#include "gm/hf/js8.h"
#include "gm/hf/jsc.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/hf_bands.h"
#include "ft8_lib/ft8/constants.h"
#include "gm/cuda/FT8Cuda.h"       // ContScanResult, CONT_CAND_MAX
#include "gm/cuda/JS8LdpcCuda.h"   // js8_ldpc_decode_cpu

// ---- Composite-to-RF frequency conversion (copy of ft8.cc helper) ----------
static const int kBandMapSize_js8 = kNumHFBands;
static struct { int ifft_start; int ifft_end; int wb_start; } kBandMap_js8[kNumHFBands];
static bool kBandMapInit_js8 = false;

static void init_band_map_js8() {
    if (kBandMapInit_js8) return;
    int offset = 0;
    for (int i = 0; i < kNumHFBands; ++i) {
        kBandMap_js8[i].ifft_start = offset;
        kBandMap_js8[i].ifft_end   = offset + (int)kHFBands[i].bw;
        kBandMap_js8[i].wb_start   = (int)kHFBands[i].wb_start;
        offset += (int)kHFBands[i].bw;
    }
    kBandMapInit_js8 = true;
}

static const int kIfftSize_js8        = 65536;
static const int kFt8FftSize_js8      = 1048576;
static const int kWidebandFftSize_js8 = 1400000;
static const float kWbSampleRate_js8  = 140000000.0f;

static float js8_composite_bin_to_rf_hz(int freq_offset) {
    init_band_map_js8();
    int ifft_bin = (int)roundf((float)freq_offset * kIfftSize_js8 / kFt8FftSize_js8);
    if (ifft_bin < 0) ifft_bin += kIfftSize_js8;
    for (int i = 0; i < kBandMapSize_js8; ++i) {
        if (ifft_bin >= kBandMap_js8[i].ifft_start && ifft_bin < kBandMap_js8[i].ifft_end) {
            int wb_bin = kBandMap_js8[i].wb_start + (ifft_bin - kBandMap_js8[i].ifft_start);
            return (float)wb_bin * kWbSampleRate_js8 / kWidebandFftSize_js8;
        }
    }
    return (float)freq_offset;
}

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

JS8::JS8(int zmq_port)
    : zmq_port_(zmq_port),
      zmq_ctx_(1),
      zmq_pub_(zmq_ctx_, zmq::socket_type::pub)
{
    if (zmq_port_ > 0) {
        char endpoint[64];
        snprintf(endpoint, sizeof(endpoint), "tcp://*:%d", zmq_port_);
        zmq_pub_.bind(endpoint);
        printf("[JS8] ZMQ PUB bound on tcp://*:%d\n", zmq_port_);
    }
}

void JS8::publishDecoded(const char* text, const char* from_call, float freq_hz,
                         float snr, double unix_time, float time_offset)
{
    printf("[JS8] DECODED: %-24s  freq=%7.0f Hz  snr=%+5.1f  offset=%+.3fs\n",
           text, (double)freq_hz, (double)snr, (double)time_offset);

    char buf[320];
    int len;
    if (from_call && from_call[0]) {
        len = snprintf(buf, sizeof(buf),
            "{\"call\":\"%s\",\"text\":\"%s\",\"freq\":%.0f,\"snr\":%.1f"
            ",\"unix\":%.0f,\"offset\":%.3f,\"mode\":\"JS8\"}",
            from_call, text, (double)freq_hz, (double)snr,
            unix_time, (double)time_offset);
    } else {
        len = snprintf(buf, sizeof(buf),
            "{\"text\":\"%s\",\"freq\":%.0f,\"snr\":%.1f"
            ",\"unix\":%.0f,\"offset\":%.3f,\"mode\":\"JS8\"}",
            text, (double)freq_hz, (double)snr,
            unix_time, (double)time_offset);
    }
    {
        std::lock_guard<std::mutex> lk(zmq_mutex_);
        if (zmq_port_ > 0) {
            zmq::message_t zmsg(buf, len);
            auto result = zmq_pub_.send(zmsg, zmq::send_flags::dontwait);
            if (!result)
                fprintf(stderr, "[JS8] ZMQ send queue full, dropped: %s\n", text);
        }
        if (broadcast_callback_)
            broadcast_callback_(from_call && from_call[0] ? from_call : text,
                                freq_hz, snr, unix_time);
    }
}

void JS8::decodeAndPublishContinuous(gm::cuda::ContScanResult& r)
{
    uint32_t n = std::min(*r.count, gm::cuda::CONT_CAND_MAX);
    if (n == 0) return;

    auto now = std::chrono::system_clock::now();
    double unix_now = std::chrono::duration<double>(now.time_since_epoch()).count();

    // Clear dedup set when JS8 epoch (15 s) rolls over.
    uint64_t epoch = (uint64_t)(unix_now / 15.0);
    if (epoch != last_epoch_) {
        seen_this_epoch_.clear();
        last_epoch_ = epoch;
    }

    uint8_t xhat[174];

    for (uint32_t i = 0; i < n; ++i) {
        const float* llr = r.log174 + (size_t)i * kFtxLdpcN;

        if (!js8_ldpc_decode_cpu(llr, xhat)) continue;

        const uint8_t* info = xhat + 87;
        if (!checkCRC12(info)) continue;

        std::string raw = extractMessage12(info);
        if (raw.empty()) continue;

        std::string from_call;
        std::string text = decodeJs8Message(info, &from_call);

        // Suppress re-publishing the same message+frequency within one epoch.
        std::string dedup_key = text + "|" + std::to_string(r.fo[i]);
        if (!seen_this_epoch_.insert(dedup_key).second) continue;

        float freq_hz  = js8_composite_bin_to_rf_hz(r.fo[i]);
        float time_sec = (r.to[i] + (float)r.ts[i] / FT8_TIME_OSR) * FT8_SYMBOL_PERIOD;
        float snr      = (float)r.score[i] - 26.0f;

        publishDecoded(text.c_str(), from_call.c_str(), freq_hz, snr, unix_now, time_sec);
    }
}

} // namespace hf
} // namespace gm
