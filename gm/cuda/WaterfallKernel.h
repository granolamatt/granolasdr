#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Build a 256-bin histogram of the uint8 magnitudes in [bin_start, bin_end) across
// all FT8_TIME_OSR time sub-arrays of one slot.  The waterfall uses a low percentile
// of this as an auto-tracked noise-floor reference, so the CW and FT8 waterfalls
// (which land at different absolute magnitudes due to their FFT sizes) self-align to
// the same scale and share one WF_GAIN / WF_OFFSET.
// hist_out: device buffer of 256 uints; the caller zeroes it before each call.
void waterfall_histogram(
    const uint8_t* slot_base,
    int bin_start, int bin_end,
    int row_stride,
    unsigned int* hist_out,
    cudaStream_t stream);

// Decimate [bin_start, bin_end) bins from all FT8_TIME_OSR sub-arrays and colormap
// each output pixel as: v = (mag - noise_ref + wf_offset) * wf_gain, clamped [0,255].
//
// slot_base  : first byte of the ring slot (device)
// row_stride : elements between time sub-arrays = freq_osr * num_bins
// noise_ref  : auto-tracked noise floor (uint8 units); mag - noise_ref = dB above noise
// wf_gain    : contrast (dB-above-noise → brightness)
// wf_offset  : noise-floor lift (uint8 units) so the floor renders visibly, not black
void waterfall_rgba(
    const uint8_t* slot_base,
    int bin_start, int bin_end,
    uint8_t* rgba_out, int out_bins,
    int num_bins, int row_stride,
    float noise_ref, float wf_gain, int wf_offset,
    cudaStream_t stream);
