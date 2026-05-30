#pragma once
// Composite-bin → RF Hz / band-index conversion, shared by ft8.cc and js8.cc.
// init_band_map() is idempotent; the other functions call it automatically.
void  init_band_map();
float composite_bin_to_rf_hz(int freq_offset);
// Returns the kHFBands index for freq_offset, or -1 if no band matches.
int   composite_bin_to_band_idx(int freq_offset);
