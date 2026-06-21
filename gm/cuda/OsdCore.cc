#include "gm/cuda/OsdCore.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace {

constexpr int N     = 174;
constexpr int K_MAX = 91;   // FT8 (largest); JS8 uses 87
constexpr int WORDS = 3;    // 174 bits -> 3 x uint64

struct Bits {
    uint64_t w[WORDS] = {0, 0, 0};
    inline int get(int i) const { return (w[i >> 6] >> (i & 63)) & 1u; }
    inline void set(int i)      { w[i >> 6] |= (uint64_t)1 << (i & 63); }
    inline void xorRow(const Bits& o) {
        w[0] ^= o.w[0]; w[1] ^= o.w[1]; w[2] ^= o.w[2];
    }
};

} // namespace

bool osd_decode_n174(const float* llr, int K, const uint8_t* gen,
                     int order, uint8_t* xhat, float* out_soft_dist)
{
    if (K < 1 || K > K_MAX) return false;
    if (order < 0) order = 0;
    if (order > 2) order = 2;

    // 1. Order positions by reliability (|llr|) descending.  perm[newcol] = oldcol.
    int perm[N];
    std::iota(perm, perm + N, 0);
    std::stable_sort(perm, perm + N, [&](int a, int b) {
        return std::fabs(llr[a]) > std::fabs(llr[b]);
    });

    // 2. Build the generator with columns permuted into reliability order, then
    //    row-reduce (RREF) to find the most-reliable basis (MRB) -- scanning
    //    columns left-to-right takes the most reliable independent columns first.
    Bits rows[K_MAX];
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            if (gen[(size_t)r * N + perm[c]]) rows[r].set(c);

    int pivot_col[K_MAX];
    int rank = 0;
    for (int col = 0; col < N && rank < K; ++col) {
        int sel = -1;
        for (int r = rank; r < K; ++r) {
            if (rows[r].get(col)) { sel = r; break; }
        }
        if (sel < 0) continue;                  // column dependent -> skip
        std::swap(rows[rank], rows[sel]);
        for (int r = 0; r < K; ++r) {           // eliminate this column elsewhere
            if (r != rank && rows[r].get(col)) rows[r].xorRow(rows[rank]);
        }
        pivot_col[rank] = col;
        ++rank;
    }
    if (rank < K) return false;                 // should not happen for a full-rank code

    // After RREF each pivot column carries exactly one 1 (in its row).  A codeword
    // whose pivot-column bits equal y is exactly XOR of rows where y_i == 1.

    // 3. OSD-0: hard decision at the MRB (pivot) columns, in permuted space.
    auto hard = [&](int permcol) -> int { return llr[perm[permcol]] > 0.0f ? 1 : 0; };

    uint8_t y[K_MAX];
    for (int i = 0; i < K; ++i) y[i] = (uint8_t)hard(pivot_col[i]);

    float rel[N];
    uint8_t hd[N];
    for (int c = 0; c < N; ++c) { rel[c] = std::fabs(llr[perm[c]]); hd[c] = (uint8_t)hard(c); }

    auto encode = [&](const uint8_t* yv, Bits& out) {
        out = Bits{};
        for (int i = 0; i < K; ++i) if (yv[i]) out.xorRow(rows[i]);
    };
    auto softDist = [&](const Bits& c) -> float {
        float d = 0.0f;
        for (int p = 0; p < N; ++p) if (c.get(p) != hd[p]) d += rel[p];
        return d;
    };

    Bits best;
    encode(y, best);
    float best_dist = softDist(best);

    // 4. Order-w: test all MRB error patterns of weight 1..order (full basis),
    //    keeping the minimum-soft-distance codeword.
    uint8_t yt[K_MAX];
    if (order >= 1) {
        for (int i = 0; i < K; ++i) {
            std::memcpy(yt, y, K);
            yt[i] ^= 1;
            Bits c; encode(yt, c);
            float d = softDist(c);
            if (d < best_dist) { best_dist = d; best = c; }
        }
    }
    if (order >= 2) {
        for (int i = 0; i < K; ++i) {
            for (int j = i + 1; j < K; ++j) {
                std::memcpy(yt, y, K);
                yt[i] ^= 1; yt[j] ^= 1;
                Bits c; encode(yt, c);
                float d = softDist(c);
                if (d < best_dist) { best_dist = d; best = c; }
            }
        }
    }

    // 5. Map the winning codeword back to natural bit order.
    std::memset(xhat, 0, N);
    for (int c = 0; c < N; ++c) xhat[perm[c]] = (uint8_t)best.get(c);

    if (out_soft_dist) *out_soft_dist = best_dist;
    return true;
}
