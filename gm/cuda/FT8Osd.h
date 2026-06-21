#pragma once
#include <cstdint>

// Ordered-Statistics Decoding (OSD) fallback for the FT8/FT4 (174,91) LDPC code.
// Host-only.  Second chance for candidates that ftx_decode_from_llr (BP) fails.
//
// Unlike JS8, FT8 already ships a parity-generator table (kFTX_LDPC_generator),
// so the systematic generator is built directly from it -- no offline derivation.
// FT8 codewords are message-first: c[0..90] = payload+CRC, c[91..173] = parity.
//
// Every output is a valid codeword (H*c^T == 0); the caller MUST validate with
// ftx_decode_from_bits (CRC-14), the only real/false-alarm discriminator.
//
//   llr:   N=174 channel LLRs (positive => bit likely 1).
//   order: 0..2 (2 is the production default).
//   plain174: output 174-bit codeword (0/1 per byte), ready for ftx_decode_from_bits.
//   out_soft_dist: optional winning soft distance for an acceptance gate.
//
// Returns true if a codeword was produced.
bool ft8_osd_decode(const float* llr, int order, uint8_t* plain174,
                    float* out_soft_dist = nullptr);
