#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Decimates the t=0,f=0 sub-block of a ring slot to out_bins output bytes
// using a quadratic frequency mapping: out[b] = slot[round((rfft_length-1)*(b/(out_bins-1))^2)].
// Runs asynchronously on the given stream.
void ft8_waterfall_decimate(
    const uint8_t* ring_slot_d,
    uint8_t*       out_d,
    int            rfft_length,
    int            out_bins,
    cudaStream_t   stream);
