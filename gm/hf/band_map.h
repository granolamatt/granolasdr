#pragma once
// Composite-bin → RF Hz / band-index conversion, shared by ft8.cc and js8.cc.
// init_band_map() is idempotent; the other functions call it automatically.
// rfft_size: FFT length used (32768 for Normal 6.25 Hz/bin, 20480 for Fast 10 Hz/bin).
void  init_band_map();
float composite_bin_to_rf_hz(int freq_offset, int rfft_size = 32768);
int   composite_bin_to_band_idx(int freq_offset, int rfft_size = 32768);
