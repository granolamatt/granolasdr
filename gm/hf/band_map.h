#pragma once
// Composite-bin → RF Hz / band-index conversion, shared by ft8.cc and js8.cc.
// init_band_map() is idempotent; the other functions call it automatically.
// rfft_size: FFT length used to produce freq_offset bins (default 1048576 = Normal/FT8).
// Fast mode callers must pass 655360; otherwise RF Hz will be 62.5% of correct.
void  init_band_map();
float composite_bin_to_rf_hz(int freq_offset, int rfft_size = 1048576);
// Returns the kHFBands index for freq_offset, or -1 if no band matches.
int   composite_bin_to_band_idx(int freq_offset, int rfft_size = 1048576);
