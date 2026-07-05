#pragma once
#include <cufft.h>

// CPU: fit a degree-D Legendre polynomial to log_mag[0..N-1] (log10 of per-bin
// magnitudes) and compute equalization gains.  gains_out[i] = 10^(poly_center -
// poly(xi)) so the noise floor is flattened to the band's center-frequency level.
void norm_fit_gains(const float* log_mag, int N, int D, float* gains_out);

// GPU: multiply data[i] by gains[i] in-place (applies equalization to packed
// composite complex bins before the IFFT).
void norm_apply_gains(cufftComplex* data, const float* gains, int N,
                      cudaStream_t stream);

// ── Wideband equalization (whole-spectrum, GPU-only) ─────────────────────────
// Flattens the entire wideband FFT before any bin-selection, so every consumer
// (FT8/JS8, CW, audio sinks tuned anywhere) inherits a flat noise floor.

// Update the per-bin asymmetric noise-floor EMA from |fft|.  a_down (fast) tracks
// the floor when a signal drops; a_up (slow) resists inflation by carriers.
void weq_update_ema(const cufftComplex* fft, float* ema, int N,
                    float a_down, float a_up, bool cold, cudaStream_t stream);

// Compute per-bin gains from the EMA: logema[i] = log10(ema[i]); a wide box filter
// of logema estimates the local noise floor (narrow carriers averaged out, not
// notched); gain[i] = 10^(global_mean_log - floor_log[i]), clamped to
// [1/max_gain, max_gain].  Self-calibrating target (global mean) preserves the
// overall level.  sum is a 1-float device scratch buffer.
void weq_compute_gains(const float* ema, float* logema, float* gain, float* sum,
                       int N, int half_win, float max_gain, cudaStream_t stream);

// Apply per-bin gains interpolated between two tap sets: g = gprev + t*(gtarg-gprev),
// t in [0,1] across the tap-update interval.  Ramping the gains every frame (instead
// of stepping every ~320 ms) removes any discontinuity from the equalized audio.
void weq_apply_interp(cufftComplex* data, const float* gprev, const float* gtarg,
                      int N, float t, cudaStream_t stream);
