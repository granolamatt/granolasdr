// GPU Sum-Product Belief Propagation LDPC decoder for JS8 Normal (174,87).
//
// Structurally identical to the BP decoder in FT8LdpcCuda.cu; only the
// H-matrix constants differ (M=87 check nodes vs FT8's M=83).
// H-matrix extracted from JS8Call JS8.cpp Nm[87] / Mn[174][3] tables.

#include <cuda_runtime.h>
#include <cstdint>
#include "gm/cuda/JS8LdpcCuda.h"

namespace {
constexpr int N     = 174;
constexpr int M     = 87;
constexpr int D_MAX = 7;
constexpr int BP_BLOCK    = 192;
constexpr int BP_MAX_ITER = 25;

// Shared memory: s_scale(4) + sx[N](696) + stotal[N](696)
//   + stoc[M*D_MAX](2436) + stov[M*D_MAX](2436) + sc_ok(4) = 6272 bytes
constexpr int BP_SMEM_BYTES = 4 + N*4 + N*4 + M*D_MAX*4 + M*D_MAX*4 + 4;
static_assert(BP_SMEM_BYTES <= 49152, "BP shared memory exceeds 48 KB");
} // namespace

__constant__ int g_js8_check_var_j[M][D_MAX];
__constant__ int g_js8_check_deg[M];
__constant__ int g_js8_var_adj[N][3][2];
__constant__ int g_js8_deg_v[N];

// H-matrix for JS8 Normal (174,87) — from JS8Call JS8.cpp Nm[87]/Mn[174][3].
static const int k_check_var_j[87][7] = {
    {   0,   29,   59,   88,  117,  146,   -1}, // check 0, deg 6
    {   1,   30,   60,   89,  118,  146,   -1}, // check 1, deg 6
    {   2,   31,   61,   90,  119,  147,   -1}, // check 2, deg 6
    {   3,   32,   62,   91,  120,  148,   -1}, // check 3, deg 6
    {   1,   33,   63,   92,  121,  149,   -1}, // check 4, deg 6
    {   4,   32,   64,   93,  122,  147,   -1}, // check 5, deg 6
    {   5,   33,   65,   94,  123,  150,   -1}, // check 6, deg 6
    {   6,   34,   66,   95,  119,  151,   -1}, // check 7, deg 6
    {   7,   35,   67,   96,  124,  152,   -1}, // check 8, deg 6
    {   8,   36,   68,   97,  125,  151,   -1}, // check 9, deg 6
    {   9,   37,   69,   98,  126,  153,   -1}, // check 10, deg 6
    {  10,   38,   70,   99,  125,  154,   -1}, // check 11, deg 6
    {  11,   39,   60,  100,  127,  144,   -1}, // check 12, deg 6
    {   9,   32,   59,   94,  127,  155,   -1}, // check 13, deg 6
    {  12,   40,   71,   96,  125,  156,   -1}, // check 14, deg 6
    {  12,   41,   72,   89,  128,  155,   -1}, // check 15, deg 6
    {  13,   38,   73,   98,  129,  157,   -1}, // check 16, deg 6
    {  14,   42,   74,  101,  130,  158,   -1}, // check 17, deg 6
    {  15,   42,   70,  102,  117,  159,   -1}, // check 18, deg 6
    {  16,   43,   75,   97,  129,  155,   -1}, // check 19, deg 6
    {  17,   44,   59,   95,  131,  160,   -1}, // check 20, deg 6
    {  18,   45,   72,   82,  132,  161,   -1}, // check 21, deg 6
    {  11,   37,   76,  101,  133,  162,   -1}, // check 22, deg 6
    {  18,   46,   77,  103,  134,  146,   -1}, // check 23, deg 6
    {   0,   31,   76,  104,  135,  163,   -1}, // check 24, deg 6
    {  19,   47,   72,  105,  122,  162,   -1}, // check 25, deg 6
    {  20,   40,   78,  106,  136,  164,   -1}, // check 26, deg 6
    {  21,   41,   65,  107,  137,  151,   -1}, // check 27, deg 6
    {  17,   41,   79,  108,  138,  153,   -1}, // check 28, deg 6
    {  22,   48,   80,  109,  134,  165,   -1}, // check 29, deg 6
    {  15,   49,   81,   90,  128,  157,   -1}, // check 30, deg 6
    {   2,   47,   62,  106,  123,  166,   -1}, // check 31, deg 6
    {   5,   50,   66,  110,  133,  154,   -1}, // check 32, deg 6
    {  23,   34,   76,   99,  121,  161,   -1}, // check 33, deg 6
    {  19,   44,   75,  111,  139,  156,   -1}, // check 34, deg 6
    {  20,   35,   63,   91,  129,  158,   -1}, // check 35, deg 6
    {   7,   51,   82,  110,  117,  165,   -1}, // check 36, deg 6
    {  20,   52,   83,  112,  137,  167,   -1}, // check 37, deg 6
    {  24,   50,   78,   88,  121,  157,   -1}, // check 38, deg 6
    {  21,   43,   74,  106,  132,  154,  171},  // check 39, deg 7
    {   8,   53,   83,   89,  140,  168,   -1}, // check 40, deg 6
    {  21,   53,   84,  109,  135,  160,   -1}, // check 41, deg 6
    {   7,   36,   64,  101,  128,  169,   -1}, // check 42, deg 6
    {  18,   38,   84,  113,  138,  149,   -1}, // check 43, deg 6
    {  25,   54,   70,   92,  141,  166,   -1}, // check 44, deg 6
    {  26,   55,   64,   95,  132,  159,  173},  // check 45, deg 7
    {  27,   30,   85,   99,  116,  170,   -1}, // check 46, deg 6
    {  27,   51,   69,  103,  131,  143,   -1}, // check 47, deg 6
    {  23,   56,   67,   94,  136,  141,   -1}, // check 48, deg 6
    {   6,   29,   71,  109,  142,  150,   -1}, // check 49, deg 6
    {   3,   50,   75,  114,  126,  167,   -1}, // check 50, deg 6
    {  15,   44,   86,  113,  124,  171,   -1}, // check 51, deg 6
    {  14,   29,   85,  114,  122,  149,   -1}, // check 52, deg 6
    {  22,   45,   63,   90,  143,  172,   -1}, // check 53, deg 6
    {  22,   34,   74,  112,  144,  152,   -1}, // check 54, deg 6
    {  13,   40,   86,  107,  116,  148,  169},  // check 55, deg 7
    {  24,   39,   84,   93,  123,  158,   -1}, // check 56, deg 6
    {  24,   57,   68,  115,  142,  173,   -1}, // check 57, deg 6
    {  28,   42,   60,  115,  131,  161,   -1}, // check 58, deg 6
    {  14,   57,   87,  111,  120,  163,   -1}, // check 59, deg 6
    {   3,   58,   71,  113,  118,  162,  172},  // check 60, deg 7
    {  26,   46,   85,   97,  133,  152,   -1}, // check 61, deg 6
    {   4,   43,   77,  108,  140,   -1,   -1}, // check 62, deg 5
    {   9,   45,   68,  102,  135,  164,   -1}, // check 63, deg 6
    {   8,   49,   58,   92,  127,  163,   -1}, // check 64, deg 6
    {  13,   56,   57,  108,  119,  165,   -1}, // check 65, deg 6
    {  16,   54,   61,  115,  124,  153,   -1}, // check 66, deg 6
    {   2,   53,   69,  100,  139,  169,   -1}, // check 67, deg 6
    {   0,   35,   81,  107,  126,  173,   -1}, // check 68, deg 6
    {   4,   52,   80,  104,  139,   -1,   -1}, // check 69, deg 5
    {  28,   52,   66,   98,  141,  172,   -1}, // check 70, deg 6
    {  17,   48,   73,   96,  114,  166,   -1}, // check 71, deg 6
    {   1,   56,   62,  102,  137,  156,   -1}, // check 72, deg 6
    {  25,   37,   78,  111,  134,  170,   -1}, // check 73, deg 6
    {  10,   51,   65,   87,  118,  147,   -1}, // check 74, deg 6
    {  19,   39,   67,  116,  140,  159,   -1}, // check 75, deg 6
    {  10,   47,   80,   88,  145,  168,   -1}, // check 76, deg 6
    {  28,   46,   79,   91,  145,  171,   -1}, // check 77, deg 6
    {   5,   31,   86,  103,  144,  168,   -1}, // check 78, deg 6
    {  26,   33,   73,  105,  130,  164,   -1}, // check 79, deg 6
    {  11,   55,   83,   87,  138,   -1,   -1}, // check 80, deg 5
    {  12,   55,   61,  110,  145,  170,   -1}, // check 81, deg 6
    {  25,   36,   79,  104,  143,  150,   -1}, // check 82, deg 6
    {  16,   30,   81,  112,  120,  160,   -1}, // check 83, deg 6
    {  27,   48,   58,   93,  136,   -1,   -1}, // check 84, deg 5
    {   6,   54,   82,  100,  130,  167,   -1}, // check 85, deg 6
    {  23,   49,   77,  105,  142,  148,   -1}, // check 86, deg 6
};
static const int k_check_deg[87] = {
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 7,
    6, 6, 6, 6, 6, 7, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 7, 6, 6, 6, 6,
    7, 6, 5, 6, 6, 6, 6, 6, 6, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    5, 6, 6, 6, 5, 6, 6
};
static const int k_var_adj[174][3][2] = {
    {{0,0}, {24,0}, {68,0}},   // var 0
    {{1,0}, {4,0}, {72,0}},    // var 1
    {{2,0}, {31,0}, {67,0}},   // var 2
    {{3,0}, {50,0}, {60,0}},   // var 3
    {{5,0}, {62,0}, {69,0}},   // var 4
    {{6,0}, {32,0}, {78,0}},   // var 5
    {{7,0}, {49,0}, {85,0}},   // var 6
    {{8,0}, {36,0}, {42,0}},   // var 7
    {{9,0}, {40,0}, {64,0}},   // var 8
    {{10,0}, {13,0}, {63,0}},  // var 9
    {{11,0}, {74,0}, {76,0}},  // var 10
    {{12,0}, {22,0}, {80,0}},  // var 11
    {{14,0}, {15,0}, {81,0}},  // var 12
    {{16,0}, {55,0}, {65,0}},  // var 13
    {{17,0}, {52,0}, {59,0}},  // var 14
    {{18,0}, {30,0}, {51,0}},  // var 15
    {{19,0}, {66,0}, {83,0}},  // var 16
    {{20,0}, {28,0}, {71,0}},  // var 17
    {{21,0}, {23,0}, {43,0}},  // var 18
    {{25,0}, {34,0}, {75,0}},  // var 19
    {{26,0}, {35,0}, {37,0}},  // var 20
    {{27,0}, {39,0}, {41,0}},  // var 21
    {{29,0}, {53,0}, {54,0}},  // var 22
    {{33,0}, {48,0}, {86,0}},  // var 23
    {{38,0}, {56,0}, {57,0}},  // var 24
    {{44,0}, {73,0}, {82,0}},  // var 25
    {{45,0}, {61,0}, {79,0}},  // var 26
    {{46,0}, {47,0}, {84,0}},  // var 27
    {{58,0}, {70,0}, {77,0}},  // var 28
    {{0,1}, {49,1}, {52,1}},   // var 29
    {{1,1}, {46,1}, {83,1}},   // var 30
    {{2,1}, {24,1}, {78,1}},   // var 31
    {{3,1}, {5,1}, {13,1}},    // var 32
    {{4,1}, {6,1}, {79,1}},    // var 33
    {{7,1}, {33,1}, {54,1}},   // var 34
    {{8,1}, {35,1}, {68,1}},   // var 35
    {{9,1}, {42,1}, {82,1}},   // var 36
    {{10,1}, {22,1}, {73,1}},  // var 37
    {{11,1}, {16,1}, {43,1}},  // var 38
    {{12,1}, {56,1}, {75,1}},  // var 39
    {{14,1}, {26,1}, {55,1}},  // var 40
    {{15,1}, {27,1}, {28,1}},  // var 41
    {{17,1}, {18,1}, {58,1}},  // var 42
    {{19,1}, {39,1}, {62,1}},  // var 43
    {{20,1}, {34,1}, {51,1}},  // var 44
    {{21,1}, {53,1}, {63,1}},  // var 45
    {{23,1}, {61,1}, {77,1}},  // var 46
    {{25,1}, {31,1}, {76,1}},  // var 47
    {{29,1}, {71,1}, {84,1}},  // var 48
    {{30,1}, {64,1}, {86,1}},  // var 49
    {{32,1}, {38,1}, {50,1}},  // var 50
    {{36,1}, {47,1}, {74,1}},  // var 51
    {{37,1}, {69,1}, {70,1}},  // var 52
    {{40,1}, {41,1}, {67,1}},  // var 53
    {{44,1}, {66,1}, {85,1}},  // var 54
    {{45,1}, {80,1}, {81,1}},  // var 55
    {{48,1}, {65,1}, {72,1}},  // var 56
    {{57,1}, {59,1}, {65,2}},  // var 57
    {{60,1}, {64,2}, {84,2}},  // var 58
    {{0,2}, {13,2}, {20,2}},   // var 59
    {{1,2}, {12,2}, {58,2}},   // var 60
    {{2,2}, {66,2}, {81,2}},   // var 61
    {{3,2}, {31,2}, {72,2}},   // var 62
    {{4,2}, {35,2}, {53,2}},   // var 63
    {{5,2}, {42,2}, {45,2}},   // var 64
    {{6,2}, {27,2}, {74,2}},   // var 65
    {{7,2}, {32,2}, {70,2}},   // var 66
    {{8,2}, {48,2}, {75,2}},   // var 67
    {{9,2}, {57,2}, {63,2}},   // var 68
    {{10,2}, {47,2}, {67,2}},  // var 69
    {{11,2}, {18,2}, {44,2}},  // var 70
    {{14,2}, {49,2}, {60,2}},  // var 71
    {{15,2}, {21,2}, {25,2}},  // var 72
    {{16,2}, {71,2}, {79,2}},  // var 73
    {{17,2}, {39,2}, {54,2}},  // var 74
    {{19,2}, {34,2}, {50,2}},  // var 75
    {{22,2}, {24,2}, {33,2}},  // var 76
    {{23,2}, {62,2}, {86,2}},  // var 77
    {{26,2}, {38,2}, {73,2}},  // var 78
    {{28,2}, {77,2}, {82,2}},  // var 79
    {{29,2}, {69,2}, {76,2}},  // var 80
    {{30,2}, {68,2}, {83,2}},  // var 81
    {{21,3}, {36,2}, {85,2}},  // var 82
    {{37,2}, {40,2}, {80,2}},  // var 83
    {{41,2}, {43,2}, {56,2}},  // var 84
    {{46,2}, {52,2}, {61,2}},  // var 85
    {{51,2}, {55,2}, {78,2}},  // var 86
    {{59,2}, {74,3}, {80,3}},  // var 87
    {{0,3}, {38,3}, {76,3}},   // var 88
    {{1,3}, {15,3}, {40,3}},   // var 89
    {{2,3}, {30,3}, {53,3}},   // var 90
    {{3,3}, {35,3}, {77,3}},   // var 91
    {{4,3}, {44,3}, {64,3}},   // var 92
    {{5,3}, {56,3}, {84,3}},   // var 93
    {{6,3}, {13,3}, {48,3}},   // var 94
    {{7,3}, {20,3}, {45,3}},   // var 95
    {{8,3}, {14,3}, {71,3}},   // var 96
    {{9,3}, {19,3}, {61,3}},   // var 97
    {{10,3}, {16,3}, {70,3}},  // var 98
    {{11,3}, {33,3}, {46,3}},  // var 99
    {{12,3}, {67,3}, {85,3}},  // var 100
    {{17,3}, {22,3}, {42,3}},  // var 101
    {{18,3}, {63,3}, {72,3}},  // var 102
    {{23,3}, {47,3}, {78,3}},  // var 103
    {{24,3}, {69,3}, {82,3}},  // var 104
    {{25,3}, {79,3}, {86,3}},  // var 105
    {{26,3}, {31,3}, {39,3}},  // var 106
    {{27,3}, {55,3}, {68,3}},  // var 107
    {{28,3}, {62,3}, {65,3}},  // var 108
    {{29,3}, {41,3}, {49,3}},  // var 109
    {{32,3}, {36,3}, {81,3}},  // var 110
    {{34,3}, {59,3}, {73,3}},  // var 111
    {{37,3}, {54,3}, {83,3}},  // var 112
    {{43,3}, {51,3}, {60,3}},  // var 113
    {{50,3}, {52,3}, {71,4}},  // var 114
    {{57,3}, {58,3}, {66,3}},  // var 115
    {{46,4}, {55,4}, {75,3}},  // var 116
    {{0,4}, {18,4}, {36,4}},   // var 117
    {{1,4}, {60,4}, {74,4}},   // var 118
    {{2,4}, {7,4}, {65,4}},    // var 119
    {{3,4}, {59,4}, {83,4}},   // var 120
    {{4,4}, {33,4}, {38,4}},   // var 121
    {{5,4}, {25,4}, {52,4}},   // var 122
    {{6,4}, {31,4}, {56,4}},   // var 123
    {{8,4}, {51,4}, {66,4}},   // var 124
    {{9,4}, {11,4}, {14,4}},   // var 125
    {{10,4}, {50,4}, {68,4}},  // var 126
    {{12,4}, {13,4}, {64,4}},  // var 127
    {{15,4}, {30,4}, {42,4}},  // var 128
    {{16,4}, {19,4}, {35,4}},  // var 129
    {{17,4}, {79,4}, {85,4}},  // var 130
    {{20,4}, {47,4}, {58,4}},  // var 131
    {{21,4}, {39,4}, {45,4}},  // var 132
    {{22,4}, {32,4}, {61,4}},  // var 133
    {{23,4}, {29,4}, {73,4}},  // var 134
    {{24,4}, {41,4}, {63,4}},  // var 135
    {{26,4}, {48,4}, {84,4}},  // var 136
    {{27,4}, {37,4}, {72,4}},  // var 137
    {{28,4}, {43,4}, {80,4}},  // var 138
    {{34,4}, {67,4}, {69,4}},  // var 139
    {{40,4}, {62,4}, {75,4}},  // var 140
    {{44,4}, {48,5}, {70,4}},  // var 141
    {{49,4}, {57,4}, {86,4}},  // var 142
    {{47,5}, {53,4}, {82,4}},  // var 143
    {{12,5}, {54,4}, {78,4}},  // var 144
    {{76,4}, {77,4}, {81,4}},  // var 145
    {{0,5}, {1,5}, {23,5}},    // var 146
    {{2,5}, {5,5}, {74,5}},    // var 147
    {{3,5}, {55,5}, {86,5}},   // var 148
    {{4,5}, {43,5}, {52,5}},   // var 149
    {{6,5}, {49,5}, {82,5}},   // var 150
    {{7,5}, {9,5}, {27,5}},    // var 151
    {{8,5}, {54,5}, {61,5}},   // var 152
    {{10,5}, {28,5}, {66,5}},  // var 153
    {{11,5}, {32,5}, {39,5}},  // var 154
    {{13,5}, {15,5}, {19,5}},  // var 155
    {{14,5}, {34,5}, {72,5}},  // var 156
    {{16,5}, {30,5}, {38,5}},  // var 157
    {{17,5}, {35,5}, {56,5}},  // var 158
    {{18,5}, {45,5}, {75,5}},  // var 159
    {{20,5}, {41,5}, {83,5}},  // var 160
    {{21,5}, {33,5}, {58,5}},  // var 161
    {{22,5}, {25,5}, {60,5}},  // var 162
    {{24,5}, {59,5}, {64,5}},  // var 163
    {{26,5}, {63,5}, {79,5}},  // var 164
    {{29,5}, {36,5}, {65,5}},  // var 165
    {{31,5}, {44,5}, {71,5}},  // var 166
    {{37,5}, {50,5}, {85,5}},  // var 167
    {{40,5}, {76,5}, {78,5}},  // var 168
    {{42,5}, {55,6}, {67,5}},  // var 169
    {{46,5}, {73,5}, {81,5}},  // var 170
    {{39,6}, {51,5}, {77,5}},  // var 171
    {{53,5}, {60,6}, {70,5}},  // var 172
    {{45,6}, {57,5}, {68,5}},  // var 173
};
static const int k_deg_v[174] = {
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};

__launch_bounds__(BP_BLOCK)
__global__ void bp_js8_kernel(
    int          B,
    const float* llr,
    uint8_t*     x_hat,
    bool*        parity
)
{
    const int b   = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B) return;

    __shared__ float s_scale;
    __shared__ float sx[N];
    __shared__ float stotal[N];
    __shared__ float stoc[M * D_MAX];
    __shared__ float stov[M * D_MAX];
    __shared__ bool  sc_ok;

    const float* llr_b = llr + (ptrdiff_t)b * N;
    const float  eps   = 1e-7f;

    if (tid == 0) {
        float sum = 0.0f, sum2 = 0.0f;
        for (int j = 0; j < N; ++j) {
            float v = llr_b[j];
            sum  += v;
            sum2 += v * v;
        }
        const float inv_n = 1.0f / N;
        const float var   = (sum2 - sum * sum * inv_n) * inv_n;
        s_scale = (var > 1e-6f) ? sqrtf(24.0f / var) : 1.0f;
    }
    __syncthreads();

    if (tid < N)
        sx[tid] = llr_b[tid] * s_scale;

    for (int idx = tid; idx < M * D_MAX; idx += BP_BLOCK)
        stov[idx] = 0.0f;
    if (tid == 0) sc_ok = false;
    __syncthreads();

    for (int it = 0; it < BP_MAX_ITER; ++it) {

        if (tid < N) {
            float tot = sx[tid];
            for (int e = 0; e < 3; ++e)
                tot += stov[g_js8_var_adj[tid][e][0] * D_MAX + g_js8_var_adj[tid][e][1]];
            stotal[tid] = tot;
        }
        __syncthreads();

        if (tid == 0) {
            int plain_sum = 0;
            for (int n = 0; n < N; ++n)
                plain_sum += (stotal[n] > 0.0f) ? 1 : 0;
            bool ok = (plain_sum > 0);
            if (ok) {
                for (int i = 0; i < M; ++i) {
                    int row_parity = 0;
                    for (int k = 0; k < g_js8_check_deg[i]; ++k)
                        row_parity ^= (stotal[g_js8_check_var_j[i][k]] > 0.0f) ? 1 : 0;
                    if (row_parity) { ok = false; break; }
                }
            }
            sc_ok = ok;
        }
        __syncthreads();
        if (sc_ok) break;

        if (tid < N) {
            for (int e = 0; e < 3; ++e) {
                const int ci  = g_js8_var_adj[tid][e][0];
                const int sk  = g_js8_var_adj[tid][e][1];
                const float extrinsic = stotal[tid] - stov[ci * D_MAX + sk];
                stoc[ci * D_MAX + sk] = tanhf(-extrinsic * 0.5f);
            }
        }
        __syncthreads();

        if (tid < M) {
            const int d = g_js8_check_deg[tid];
            float pref[D_MAX + 1], suf[D_MAX + 1];
            pref[0] = 1.0f;
            for (int k = 0; k < d; ++k)
                pref[k + 1] = pref[k] * stoc[tid * D_MAX + k];
            suf[d] = 1.0f;
            for (int k = d - 1; k >= 0; --k)
                suf[k] = suf[k + 1] * stoc[tid * D_MAX + k];
            for (int k = 0; k < d; ++k) {
                float excl = pref[k] * suf[k + 1];
                excl = fmaxf(-1.0f + eps, fminf(1.0f - eps, excl));
                stov[tid * D_MAX + k] = -2.0f * atanhf(excl);
            }
        }
        __syncthreads();
    }

    if (tid < N)
        x_hat[(ptrdiff_t)b * N + tid] = (stotal[tid] > 0.0f) ? 1u : 0u;
    if (tid == 0)
        parity[b] = sc_ok;
}

void js8_ldpc_init_constants()
{
    cudaMemcpyToSymbol(g_js8_check_var_j, k_check_var_j, sizeof(k_check_var_j));
    cudaMemcpyToSymbol(g_js8_check_deg,   k_check_deg,   sizeof(k_check_deg));
    cudaMemcpyToSymbol(g_js8_var_adj,     k_var_adj,     sizeof(k_var_adj));
    cudaMemcpyToSymbol(g_js8_deg_v,       k_deg_v,       sizeof(k_deg_v));
}

void js8_bp_decode_batch(
    uint32_t     n_candidates,
    const float* log174_d,
    uint8_t*     x_hat_d,
    bool*        parity_d,
    cudaStream_t ldpc_stream)
{
    if (n_candidates == 0) return;
    bp_js8_kernel<<<(int)n_candidates, BP_BLOCK, 0, ldpc_stream>>>(
        (int)n_candidates, log174_d, x_hat_d, parity_d);
    cudaGetLastError();
}

// ---------------------------------------------------------------------------
// CPU single-candidate BP decoder — mirrors bp_js8_kernel logic exactly.
// Uses the same k_* static tables defined above.
// ---------------------------------------------------------------------------

bool js8_ldpc_decode_cpu(const float* llr, uint8_t* xhat)
{
    const int Nc  = N;   // 174
    const int Mc  = M;   // 87
    const float eps = 1e-7f;

    // Normalize to variance=24 (matching GPU kernel)
    float sum = 0.0f, sum2 = 0.0f;
    for (int j = 0; j < Nc; ++j) { sum += llr[j]; sum2 += llr[j] * llr[j]; }
    const float inv_n = 1.0f / Nc;
    const float var   = (sum2 - sum * sum * inv_n) * inv_n;
    const float scale = (var > 1e-6f) ? sqrtf(24.0f / var) : 1.0f;

    float sx[N];
    for (int j = 0; j < Nc; ++j) sx[j] = llr[j] * scale;

    // stov[check][slot] = c2v LLR messages;  stoc[check][slot] = v2c tanh messages
    float stov[M][D_MAX] = {};
    float stoc[M][D_MAX];
    float stotal[N];

    for (int iter = 0; iter < BP_MAX_ITER; ++iter) {
        // Compute total LLR per variable
        for (int n = 0; n < Nc; ++n) {
            float tot = sx[n];
            for (int e = 0; e < 3; ++e)
                tot += stov[k_var_adj[n][e][0]][k_var_adj[n][e][1]];
            stotal[n] = tot;
        }

        // Parity check (all-zeros guard same as GPU)
        int ones = 0;
        for (int n = 0; n < Nc; ++n) ones += (stotal[n] > 0.0f) ? 1 : 0;
        bool ok = (ones > 0);
        if (ok) {
            for (int i = 0; i < Mc && ok; ++i) {
                int row_p = 0;
                for (int k = 0; k < k_check_deg[i]; ++k)
                    row_p ^= (stotal[k_check_var_j[i][k]] > 0.0f) ? 1 : 0;
                if (row_p) ok = false;
            }
        }
        if (ok) {
            for (int n = 0; n < Nc; ++n) xhat[n] = (stotal[n] > 0.0f) ? 1u : 0u;
            return true;
        }

        // v2c: stoc[ci][sk] = tanh(-extrinsic/2)
        for (int n = 0; n < Nc; ++n) {
            for (int e = 0; e < 3; ++e) {
                int ci = k_var_adj[n][e][0], sk = k_var_adj[n][e][1];
                float ext = stotal[n] - stov[ci][sk];
                stoc[ci][sk] = tanhf(-ext * 0.5f);
            }
        }

        // c2v: stov[m][k] = -2*atanh(product excl k), prefix-suffix trick
        for (int m = 0; m < Mc; ++m) {
            const int d = k_check_deg[m];
            float pref[D_MAX + 1], suf[D_MAX + 1];
            pref[0] = 1.0f;
            for (int k = 0; k < d; ++k)
                pref[k + 1] = pref[k] * stoc[m][k];
            suf[d] = 1.0f;
            for (int k = d - 1; k >= 0; --k)
                suf[k] = suf[k + 1] * stoc[m][k];
            for (int k = 0; k < d; ++k) {
                float excl = fmaxf(-1.0f + eps, fminf(1.0f - eps, pref[k] * suf[k + 1]));
                stov[m][k] = -2.0f * atanhf(excl);
            }
        }
    }

    // Final hard decisions
    for (int n = 0; n < Nc; ++n) xhat[n] = (stotal[n] > 0.0f) ? 1u : 0u;
    return false;
}
