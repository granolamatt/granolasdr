#pragma once
#include <cstdint>
#include <string>

// Lightweight wrapper around JSC's (s,c)-dense word compression used by JS8.
// Table data lives in jsc_map.cc and jsc_list.cc (adapted from JS8Call).

struct JscTuple {
    char const* str;
    int size;
    int index;
};

static const uint32_t kJscSize       = 262144;
static const uint32_t kJscPrefixSize = 103;

extern const JscTuple jsc_map   [kJscSize];
extern const JscTuple jsc_list  [kJscSize];
extern const JscTuple jsc_prefix[kJscPrefixSize];

// Decompress JSC-encoded bit stream. bits[i] must be 0 or 1.
// Returns decoded text, or empty string on failure.
std::string jsc_decompress(const uint8_t* bits, int nbits);
