#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Upper bound on syndrome scan candidates per epoch.
static const uint32_t FT8_GPU_SYNDROME_MAX = 100000;

// Maximum failed LDPC parity checks to still pass as a candidate (data tones).
// 83 = disabled; Costas is the sole GPU gate. Syndrome filtering done by BP on CPU.
static const int FT8_SYN_MAX_ERRORS = 83;

// Minimum Costas sync tone matches (out of 21 total) required to pass as a candidate.
// Random noise expected matches: ~21 * 1/8 = 2.6. Real signals at decodable SNR: 14-21.
// At 12, false-positive rate across ~5M checks/scan is ~0 — tight enough to control
// candidate throughput without losing decodable signals.
static const int FT8_SYN_MIN_SYNC = 12;

// Launch the syndrome + Costas sync scan kernel over one frequency sub-range.
// A candidate must satisfy BOTH:
//   failed LDPC parity checks <= max_syn_errors  (data tones)
//   Costas argmax matches      >= min_sync_matches (sync tones)
//
// Call once per HF band, looping over all bands to cover the full spectrum.
// Caller must reset cand_count_d to 0 (cudaMemsetAsync) BEFORE the first band call;
// this function does NOT reset it, so candidates accumulate across band calls.
//
//   total_bins  — ring buffer frequency stride (= rfft_length, never changes)
//   scan_bins   — number of bins to search in this call
//   bin_start   — first bin index for this call (absolute, 0-based)
void ft8_syndrome_scan(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    int32_t*  cand_fo_d,
    uint8_t*  cand_to_d,
    uint8_t*  cand_ts_d,
    uint8_t*  cand_fs_d,
    uint32_t* cand_count_d,
    uint32_t  max_cands,
    int total_bins, int scan_bins, int bin_start,
    int num_blocks, int time_osr, int freq_osr,
    int max_syn_errors, int min_sync_matches,
    cudaStream_t stream);
