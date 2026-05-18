#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Upper bound on syndrome scan candidates per epoch.
static const uint32_t FT8_GPU_SYNDROME_MAX = 100000;

// Maximum failed LDPC parity checks to still pass as a candidate (data tones).
// Each hard-decision bit error flips ~3 check equations; BP decode handles the rest.
// 0 = strict, 20 = ~6-7 bit error tolerance.
// WARNING: above ~25 noise false-positive rate explodes — use min_sync_matches to compensate.
static const int FT8_SYN_MAX_ERRORS = 40;

// Minimum Costas sync tone matches (out of 21 total) required to pass as a candidate.
// Random noise expected matches: ~21 * 1/8 = 2.6. Real signals at decodable SNR: 14-21.
// The sync gate suppresses noise that slips through a loose syndrome threshold.
static const int FT8_SYN_MIN_SYNC = 9;

// Launch the combined syndrome + Costas sync scan kernel.
// A candidate must satisfy BOTH:
//   failed LDPC parity checks <= max_syn_errors  (data tones)
//   Costas argmax matches      >= min_sync_matches (sync tones)
// cand_count_d is reset to 0 by this call before the kernel runs.
void ft8_syndrome_scan(
    const uint8_t* mag_d,
    int snap_start, int ring_size,
    int32_t*  cand_fo_d,
    uint8_t*  cand_to_d,
    uint8_t*  cand_ts_d,
    uint8_t*  cand_fs_d,
    uint32_t* cand_count_d,
    uint32_t  max_cands,
    int num_bins, int num_blocks, int time_osr, int freq_osr,
    int max_syn_errors, int min_sync_matches,
    cudaStream_t stream);
