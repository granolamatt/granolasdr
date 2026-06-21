#pragma once
#include <cstdint>

// Ordered-Statistics Decoding (OSD) fallback for the JS8 Normal (174,87) LDPC
// code.  Host-only; no CUDA.  Used as a second chance for candidates the GPU
// star-sum SP decoder fails to converge on (see js8.cc decodeAndPublishContinuous).
//
// Generator-based MRB-OSD (Fossorier): order the 174 bit positions by
// reliability, Gaussian-eliminate the generator over the most-reliable basis,
// re-encode the hard MRB decisions (order 0), then test all MRB error patterns
// of weight <= order, keeping the codeword with minimum soft distance.
//
// Because every output is a linear combination of generator rows it is always a
// valid codeword (H*c^T == 0); the caller MUST still gate on CRC-12, which is the
// only thing distinguishing a real decode from an OSD-on-noise false alarm.
//
//   llr:   N=174 channel LLRs (positive => bit likely 1), same convention as the
//          SP decoder and the captured js8_llr_*.jsonl files.
//   order: 0..2 (2 is the production default).
//   xhat:  output 174-bit codeword (parity [0..86], message [87..173]).
//   out_soft_dist: optional, set to the winning candidate's soft distance
//          (sum of |llr| over disagreeing positions) for an acceptance gate.
//
// Returns true if a codeword was produced.
bool js8_osd_decode(const float* llr, int order, uint8_t* xhat,
                    float* out_soft_dist = nullptr);
