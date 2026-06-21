#pragma once
#include <cstdint>

// Generic most-reliable-basis Ordered-Statistics Decoder for the N=174 LDPC
// codes used here (JS8 (174,87) and FT8 (174,91) differ only in K).  Host-only.
//
// Order the 174 bit positions by reliability, Gaussian-eliminate the generator
// over the most-reliable basis, re-encode the hard MRB decisions (order 0), then
// test all MRB error patterns of weight <= order, keeping the minimum-soft-
// distance codeword.  Every output is a linear combination of generator rows, so
// it is always a valid codeword (H*c^T == 0); the caller MUST gate on CRC.
//
//   llr:   N=174 channel LLRs (positive => bit likely 1).
//   K:     number of message rows (87 JS8 / 91 FT8); must be <= 91.
//   gen:   K x 174 row-major systematic generator (codeword = msg * gen).
//   order: 0..2.
//   xhat:  output 174-bit codeword.
//   out_soft_dist: optional winning soft distance (sum of |llr| over disagreements).
//
// Returns true if a codeword was produced.
bool osd_decode_n174(const float* llr, int K, const uint8_t* gen,
                     int order, uint8_t* xhat, float* out_soft_dist);
