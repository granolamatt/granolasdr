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
