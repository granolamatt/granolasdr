#include "gm/hf/jsc.h"

#include <vector>

// Ported from JSC::decompress() in JS8Call (Jordan Sherer KN4CRD, GPLv3).
// Algorithm: (s,c)-dense prefix-free code with b=4 bits/symbol, s=7, c=9.
// Each group of 4 bits is either a terminal (<s) or continuation (>=s).
// A 1-bit separator follows each terminal to indicate an inter-word space.
std::string jsc_decompress(const uint8_t* bits, int nbits) {
    const uint32_t s = 7;
    const uint32_t c = 9;   // (1<<4) - s

    // Pre-compute base offsets for each codeword depth k:
    //   base[0]=0, base[k] = base[k-1] + s * c^(k-1)
    uint32_t base[8];
    base[0] = 0;
    base[1] = s;
    uint32_t cpow = 1;
    for (int k = 2; k < 8; k++) {
        cpow *= c;
        base[k] = base[k - 1] + s * cpow;
    }

    // Pass 1: decode 4-bit groups and record separator positions.
    std::vector<uint32_t> grps;
    std::vector<uint32_t> seps;
    grps.reserve(nbits / 4);

    int i = 0;
    while (i + 4 <= nbits) {
        uint32_t g = (bits[i] << 3) | (bits[i+1] << 2) | (bits[i+2] << 1) | bits[i+3];
        grps.push_back(g);
        i += 4;
        if (g < s) {
            // One separator bit follows the terminal group.
            if (i < nbits && bits[i])
                seps.push_back((uint32_t)grps.size() - 1);
            i += 1;
        }
    }

    // Pass 2: walk groups to reconstruct words.
    std::string out;
    uint32_t start = 0;
    uint32_t nseps = 0;
    while (start < (uint32_t)grps.size()) {
        uint32_t k = 0;
        uint32_t j = 0;

        while (start + k < (uint32_t)grps.size() && grps[start + k] >= s) {
            j = j * c + (grps[start + k] - s);
            k++;
            if (k >= 7 || base[k] + j * s >= kJscSize) break;
        }
        if (start + k >= (uint32_t)grps.size()) break;

        j = j * s + grps[start + k] + base[k];
        if (j >= kJscSize) break;

        out += jsc_map[j].str;

        // Append space separator if flagged for this terminal position.
        if (nseps < (uint32_t)seps.size() && seps[nseps] == start + k) {
            out += ' ';
            nseps++;
        }

        start += k + 1;
    }

    return out;
}
