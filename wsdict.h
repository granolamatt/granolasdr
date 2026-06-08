#pragma once
// wsdict.h — C++ client for the websocket_dict server.
// Canonical location: websocket_dict/include/wsdict.h
// Include in consumer projects: #include "path/to/websocket_dict/include/wsdict.h"
//
// Protocol: ws://127.0.0.1:8765/ws  (port configurable)
// All messages are JSON over WebSocket text frames (RFC 6455).
//
// Thread safety: one WsDictClient per thread. Do not share instances.

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Base64 encoder — for spectrum binary → JSON string
// ---------------------------------------------------------------------------

inline std::string wsdict_base64_encode(const uint8_t* data, size_t len)
{
    static const char T[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = (uint32_t)data[i] << 16;
        if (i + 1 < len) n |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) n |= data[i + 2];
        out += T[(n >> 18) & 63];
        out += T[(n >> 12) & 63];
        out += (i + 1 < len) ? T[(n >> 6) & 63] : '=';
        out += (i + 2 < len) ? T[n & 63] : '=';
    }
    return out;
}

// ---------------------------------------------------------------------------
// WsDictClient — minimal blocking WebSocket client for wsdict protocol
//
// Thread safety: one instance per thread.
// A single instance should either do get/set requests OR be subscribed —
// don't mix on the same connection.
// ---------------------------------------------------------------------------

class WsDictClient {
public:
    inline WsDictClient(const char* host = "127.0.0.1", int port = 8765);
    inline ~WsDictClient();

    WsDictClient(const WsDictClient&) = delete;
    WsDictClient& operator=(const WsDictClient&) = delete;

    // Send {"type":"set","key":key,"value":value}. Fire-and-forget.
    inline void set(const std::string& key, const nlohmann::json& value);

    // Send set with TTL. The key expires after ttl_ms milliseconds.
    // Minimum precision: ~100ms (server sweep interval).
    inline void set_with_ttl(const std::string& key,
                              const nlohmann::json& value,
                              int ttl_ms);

    // Send get and return the value, or null json if key not found.
    inline nlohmann::json get(const std::string& key);

    // List all currently live keys. Returns a vector of key strings.
    //
    // LIMITATION: must be called on a dedicated non-subscribed connection.
    // On a subscribed connection, async push messages (update, key_expired)
    // can arrive before the "keys" response and would be silently discarded.
    // Create a short-lived WsDictClient just for this call.
    inline std::vector<std::string> list();

    // Subscribe to keys / glob patterns. After this, recv_message() delivers
    // update, key_expired, and key_deleted messages. The server immediately
    // pushes current values for all currently matching keys.
    //
    // Glob patterns:
    //   events/*     — single segment: events/a but not events/a/b
    //   program/**   — any depth: program/a, program/a/b, program/a/b/c
    //   **           — all keys
    //   plot_*       — prefix match within a single segment
    inline void subscribe(const std::vector<std::string>& keys);

    // Block until next server→client message. Returns parsed JSON.
    inline nlohmann::json recv_message();

    // Send {"type":"delete","key":key}. Fire-and-forget.
    inline void del(const std::string& key);

private:
    int sock_{-1};
    int req_id_{0};

    inline void        send_frame(const std::string& text);
    inline std::string recv_frame();
    inline void        read_exact(uint8_t* dst, size_t n);
    inline void        send_all(const uint8_t* src, size_t n);
};

// ---------------------------------------------------------------------------
// WsDictClient — implementation
// ---------------------------------------------------------------------------

inline WsDictClient::WsDictClient(const char* host, int port)
{
    sock_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock_ < 0) throw std::runtime_error("wsdict: socket() failed");

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        ::close(sock_); sock_ = -1;
        throw std::runtime_error(std::string("wsdict: invalid host: ") + host);
    }
    if (::connect(sock_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(sock_); sock_ = -1;
        throw std::runtime_error(
            std::string("wsdict: connect failed — is websocket_dict running at ") +
            host + ":" + std::to_string(port) + "?");
    }

    // HTTP/1.1 WebSocket upgrade
    std::mt19937 rng(std::random_device{}());
    uint8_t kb[16];
    for (auto& b : kb) b = static_cast<uint8_t>(rng());
    std::string ws_key = wsdict_base64_encode(kb, 16);

    std::ostringstream req;
    req << "GET /ws HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Upgrade: websocket\r\n"
        << "Connection: Upgrade\r\n"
        << "Sec-WebSocket-Key: " << ws_key << "\r\n"
        << "Sec-WebSocket-Version: 13\r\n"
        << "\r\n";
    std::string req_str = req.str();
    send_all(reinterpret_cast<const uint8_t*>(req_str.c_str()), req_str.size());

    // Read HTTP response until \r\n\r\n
    std::string resp;
    char c;
    while (resp.size() < 4 || resp.substr(resp.size() - 4) != "\r\n\r\n") {
        if (::recv(sock_, &c, 1, 0) != 1) {
            ::close(sock_); sock_ = -1;
            throw std::runtime_error("wsdict: HTTP upgrade recv failed");
        }
        resp += c;
    }
    if (resp.find("101") == std::string::npos) {
        ::close(sock_); sock_ = -1;
        throw std::runtime_error("wsdict: WebSocket upgrade rejected by server");
    }

    // Consume the mandatory hello frame: {"type":"hello","version":"1.0"}
    recv_frame();
}

inline WsDictClient::~WsDictClient()
{
    if (sock_ >= 0) {
        // Send close frame (opcode 0x8, FIN, masked, no payload)
        uint8_t cf[6] = {0x88, 0x80, 0x00, 0x00, 0x00, 0x00};
        ::send(sock_, cf, sizeof(cf), MSG_NOSIGNAL);
        ::close(sock_);
        sock_ = -1;
    }
}

inline void WsDictClient::read_exact(uint8_t* dst, size_t n)
{
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(sock_, dst + got, n - got, 0);
        if (r <= 0) throw std::runtime_error("wsdict: recv failed (connection closed?)");
        got += r;
    }
}

inline void WsDictClient::send_all(const uint8_t* src, size_t n)
{
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = ::send(sock_, src + sent, n - sent, MSG_NOSIGNAL);
        if (r <= 0) throw std::runtime_error("wsdict: send failed");
        sent += r;
    }
}

inline void WsDictClient::send_frame(const std::string& text)
{
    // Client frames MUST be masked (RFC 6455)
    size_t len = text.size();
    uint8_t mask[4];
    {
        uint32_t m = static_cast<uint32_t>(std::rand());
        std::memcpy(mask, &m, 4);
    }

    std::vector<uint8_t> frame;
    frame.reserve(2 + 8 + 4 + len);
    frame.push_back(0x81);  // FIN + text opcode

    if (len < 126) {
        frame.push_back(0x80 | static_cast<uint8_t>(len));
    } else if (len < 65536) {
        frame.push_back(0x80 | 126);
        frame.push_back((len >> 8) & 0xff);
        frame.push_back(len & 0xff);
    } else {
        frame.push_back(0x80 | 127);
        for (int i = 7; i >= 0; i--)
            frame.push_back((len >> (i * 8)) & 0xff);
    }

    frame.insert(frame.end(), mask, mask + 4);
    for (size_t i = 0; i < len; i++)
        frame.push_back(static_cast<uint8_t>(text[i]) ^ mask[i % 4]);

    send_all(frame.data(), frame.size());
}

inline std::string WsDictClient::recv_frame()
{
    uint8_t hdr[2];
    read_exact(hdr, 2);

    int      opcode = (hdr[0] & 0x0f);
    bool     masked = (hdr[1] & 0x80) != 0;
    uint64_t len    = (hdr[1] & 0x7f);

    if (len == 126) {
        uint8_t ext[2]; read_exact(ext, 2);
        len = (uint64_t(ext[0]) << 8) | ext[1];
    } else if (len == 127) {
        uint8_t ext[8]; read_exact(ext, 8);
        len = 0;
        for (int i = 0; i < 8; i++) len = (len << 8) | ext[i];
    }

    uint8_t mask[4] = {};
    if (masked) read_exact(mask, 4);

    std::vector<uint8_t> payload(len);
    if (len > 0) read_exact(payload.data(), len);
    if (masked)
        for (size_t i = 0; i < len; i++) payload[i] ^= mask[i % 4];

    if (opcode == 0x8) throw std::runtime_error("wsdict: server closed connection");
    if (opcode == 0x9) {
        // Ping — respond with pong (opcode 0xA, FIN, masked, echo payload)
        std::vector<uint8_t> pong;
        uint8_t pmask[4] = {};
        pong.push_back(0x8A);
        pong.push_back(0x80 | static_cast<uint8_t>(std::min(len, uint64_t(125))));
        pong.insert(pong.end(), pmask, pmask + 4);
        send_all(pong.data(), pong.size());
        return recv_frame();  // tail-recurse to get the real message
    }

    return std::string(payload.begin(), payload.end());
}

inline void WsDictClient::set(const std::string& key, const nlohmann::json& value)
{
    nlohmann::json msg = {{"type", "set"}, {"key", key}, {"value", value}};
    send_frame(msg.dump());
}

inline void WsDictClient::set_with_ttl(const std::string& key,
                                        const nlohmann::json& value,
                                        int ttl_ms)
{
    nlohmann::json msg = {
        {"type", "set"}, {"key", key}, {"value", value}, {"ttl_ms", ttl_ms}
    };
    send_frame(msg.dump());
}

inline void WsDictClient::del(const std::string& key)
{
    nlohmann::json msg = {{"type", "delete"}, {"key", key}};
    send_frame(msg.dump());
}

inline nlohmann::json WsDictClient::get(const std::string& key)
{
    std::string id = "g" + std::to_string(++req_id_);
    nlohmann::json req = {{"type", "get"}, {"key", key}, {"id", id}};
    send_frame(req.dump());

    // Read until we get the response with our id
    while (true) {
        nlohmann::json resp = nlohmann::json::parse(recv_frame());
        if (resp.value("id", std::string{}) == id) {
            if (resp["type"] == "value") return resp["value"];
            return nlohmann::json(nullptr);  // error = key not found
        }
        // Ignore unrelated messages (shouldn't happen on a non-subscribe connection)
    }
}

inline std::vector<std::string> WsDictClient::list()
{
    nlohmann::json msg = {{"type", "list"}};
    send_frame(msg.dump());
    // Read until we get the "keys" response.
    // Safe only on a non-subscribed connection — see LIMITATION in the declaration.
    while (true) {
        nlohmann::json resp = nlohmann::json::parse(recv_frame());
        if (resp.value("type", "") == "keys") {
            return resp["keys"].get<std::vector<std::string>>();
        }
    }
}

inline void WsDictClient::subscribe(const std::vector<std::string>& keys)
{
    nlohmann::json msg = {{"type", "subscribe"}, {"keys", keys}};
    send_frame(msg.dump());
    recv_frame();  // read and discard the "subscribed" ack
    // Note: server immediately pushes current values for matching keys.
    // The caller is responsible for reading those via recv_message().
}

inline nlohmann::json WsDictClient::recv_message()
{
    return nlohmann::json::parse(recv_frame());
}

// ---------------------------------------------------------------------------
// Pipeline config data types
// ---------------------------------------------------------------------------

struct ChannelConfig {
    double center_hz      {0.0};
    int    symbol_rate_hz {2'000'000};
    bool   is_burst       {false};
};

struct PipelineConfig {
    std::string              device        {"addr=127.0.0.1,master_clock_rate=200e6"};
    double                   freq_hz       {1e9};
    double                   gain_db       {30.0};
    double                   sample_rate_hz{200e6};
    double                   duration_sec  {0.0};
    int                      waterfall_interval_ms{500};
    std::string              source              {"uhd"};
    bool                     gpu_direct_open_loop{false};
    int                      gpu_direct_udp_port {0};
    uint16_t                 gpu_direct_fc_epid  {0};
    std::vector<ChannelConfig> channels;
};

// ---------------------------------------------------------------------------
// write_defaults_if_missing — write default config to wsdict if not present.
// Called at pipeline startup before load_config().
// ---------------------------------------------------------------------------

inline void write_defaults_if_missing(const char* host = "127.0.0.1", int port = 8765)
{
    for (int attempt = 0; ; attempt++) {
        try {
            WsDictClient c(host, port);

            nlohmann::json existing = c.get("uhd:config");
            if (!existing.is_null()) return;

            c.set("uhd:config", {
                {"device",                 "addr=127.0.0.1,master_clock_rate=200e6"},
                {"freq_hz",                1e9},
                {"gain_db",                30.0},
                {"sample_rate_hz",         200e6},
                {"duration_sec",           0.0},
                {"waterfall_interval_ms",  500},
                {"source",                 "uhd"},
                {"gpu_direct_open_loop",   false},
                {"gpu_direct_udp_port",    0},
                {"gpu_direct_fc_epid",     0}
            });
            std::cout << "[config] wrote default uhd:config\n";

            c.set("uhd:channels", nlohmann::json::array({
                {{"center_hz",  0.0},   {"symbol_rate_hz", 2000000}, {"is_burst", false}},
                {{"center_hz",  10e6},  {"symbol_rate_hz", 2000000}, {"is_burst", false}},
                {{"center_hz", -10e6},  {"symbol_rate_hz", 2000000}, {"is_burst", false}},
                {{"center_hz",  20e6},  {"symbol_rate_hz", 2000000}, {"is_burst", false}},
            }));
            std::cout << "[config] wrote default uhd:channels\n";
            return;
        } catch (const std::exception& e) {
            if (attempt >= 10) {
                std::cerr << "[config] write_defaults_if_missing: " << e.what()
                          << " — giving up\n";
                return;
            }
            std::cerr << "[config] wsdict not ready: " << e.what() << " — retrying\n";
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
}

// ---------------------------------------------------------------------------
// load_config — read PipelineConfig from wsdict at startup
// ---------------------------------------------------------------------------

inline PipelineConfig load_config(const char* host = "127.0.0.1", int port = 8765)
{
    PipelineConfig cfg;

    for (int attempt = 0; ; attempt++) {
        try {
            WsDictClient c(host, port);

            nlohmann::json j = c.get("uhd:config");
            if (!j.is_null()) {
                if (j.contains("device"))                cfg.device                = j["device"].get<std::string>();
                if (j.contains("freq_hz"))               cfg.freq_hz               = j["freq_hz"].get<double>();
                if (j.contains("gain_db"))               cfg.gain_db               = j["gain_db"].get<double>();
                if (j.contains("sample_rate_hz"))        cfg.sample_rate_hz        = j["sample_rate_hz"].get<double>();
                if (j.contains("duration_sec"))          cfg.duration_sec          = j["duration_sec"].get<double>();
                if (j.contains("waterfall_interval_ms")) cfg.waterfall_interval_ms = j["waterfall_interval_ms"].get<int>();
                if (j.contains("source"))                cfg.source                = j["source"].get<std::string>();
                if (j.contains("gpu_direct_open_loop"))  cfg.gpu_direct_open_loop  = j["gpu_direct_open_loop"].get<bool>();
                if (j.contains("gpu_direct_udp_port"))   cfg.gpu_direct_udp_port   = j["gpu_direct_udp_port"].get<int>();
                if (j.contains("gpu_direct_fc_epid"))    cfg.gpu_direct_fc_epid    = static_cast<uint16_t>(j["gpu_direct_fc_epid"].get<int>());
            }

            nlohmann::json chs = c.get("uhd:channels");
            if (!chs.is_null() && chs.is_array()) {
                for (const auto& ch_j : chs) {
                    ChannelConfig ch;
                    if (ch_j.contains("center_hz"))      ch.center_hz      = ch_j["center_hz"].get<double>();
                    if (ch_j.contains("symbol_rate_hz")) ch.symbol_rate_hz = ch_j["symbol_rate_hz"].get<int>();
                    if (ch_j.contains("is_burst"))       ch.is_burst       = ch_j["is_burst"].get<bool>();
                    cfg.channels.push_back(ch);
                }
            }

            break;
        } catch (const std::exception& e) {
            if (attempt >= 10) throw;
            std::cerr << "[config] load_config: " << e.what() << " — retrying\n";
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    std::cout << "[config] device="  << cfg.device
              << "  freq="    << cfg.freq_hz / 1e6        << " MHz"
              << "  gain="    << cfg.gain_db               << " dB"
              << "  rate="    << cfg.sample_rate_hz / 1e6  << " Msps"
              << "  channels=" << cfg.channels.size()      << "\n";
    for (size_t i = 0; i < cfg.channels.size(); i++) {
        std::cout << "  ch[" << i << "]"
                  << "  center=" << cfg.channels[i].center_hz / 1e6 << " MHz"
                  << "  rate="   << cfg.channels[i].symbol_rate_hz  << " Baud\n";
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// start_config_watcher — background thread that restarts the pipeline when
// uhd:config or uhd:channels changes or expires.
// ---------------------------------------------------------------------------

inline std::thread start_config_watcher(std::function<void()> restart_cb,
                                         const char* host = "127.0.0.1",
                                         int port = 8765)
{
    return std::thread([=]() {
        // Fetch baseline values on a separate short-lived connection
        nlohmann::json baseline_cfg, baseline_chs;
        while (true) {
            try {
                WsDictClient c(host, port);
                baseline_cfg = c.get("uhd:config");
                baseline_chs = c.get("uhd:channels");
                break;
            } catch (const std::exception& e) {
                std::cerr << "[config watcher] baseline fetch failed: " << e.what()
                          << " — retrying\n";
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }

        // Subscribe and watch on a persistent connection
        while (true) {
            try {
                WsDictClient watcher(host, port);
                watcher.subscribe({"uhd:config", "uhd:channels"});
                std::cout << "[config watcher] watching uhd:config and uhd:channels\n";

                while (true) {
                    nlohmann::json msg = watcher.recv_message();
                    const std::string type = msg.value("type", std::string{});

                    if (type == "update") {
                        std::string key = msg["key"];
                        const nlohmann::json& val = msg["value"];
                        bool changed = false;
                        if (key == "uhd:config"   && val != baseline_cfg) changed = true;
                        if (key == "uhd:channels" && val != baseline_chs) changed = true;
                        if (changed) {
                            std::cout << "[config watcher] " << key
                                      << " changed — restarting pipeline\n";
                            restart_cb();
                            return;
                        }
                    } else if (type == "key_deleted") {
                        std::string key = msg["key"];
                        if (key == "uhd:config" || key == "uhd:channels") {
                            std::cout << "[config watcher] " << key
                                      << " deleted — restarting pipeline\n";
                            restart_cb();
                            return;
                        }
                    } else if (type == "key_expired") {
                        std::string key = msg["key"];
                        if (key == "uhd:config" || key == "uhd:channels") {
                            std::cout << "[config watcher] " << key
                                      << " expired — restarting pipeline\n";
                            restart_cb();
                            return;
                        }
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[config watcher] lost connection: " << e.what()
                          << " — reconnecting\n";
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
    });
}
