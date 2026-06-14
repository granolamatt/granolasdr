#include "gm/hf/band_map.h"
#include "gm/hf/hf_bands.h"
#include <cmath>

// The HFChannelizer packs FT8/JS8 sub-band windows into a 4096-bin composite IFFT,
// outputs 2048 complex samples at 409.6 kHz.  The FT8/JS8 FFT (65536 points)
// has bin_hz = 409,600 / 65,536 = 6.25 Hz.  Convert freq_offset (an FFT bin)
// back to RF using the HFChannelizer band table.

static const int   kBandMapSize     = kNumHFBands;
static const int   kIfftSize        = 4096;
static const int   kFt8FftSize      = 65536;
static const int   kWidebandFftSize = 1400000; // 2 * NLARGE
static const float kWbSampleRate    = 140000000.0f;

static struct { int ifft_start; int ifft_end; int wb_start; } kBandMap[kNumHFBands];
static bool kBandMapReady = false;

void init_band_map() {
    if (kBandMapReady) return;
    int offset = 0;
    for (int i = 0; i < kNumHFBands; ++i) {
        kBandMap[i].ifft_start = offset;
        kBandMap[i].ifft_end   = offset + (int)kHFBands[i].bw;
        kBandMap[i].wb_start   = (int)kHFBands[i].wb_start;
        offset += (int)kHFBands[i].bw;
    }
    kBandMapReady = true;
}

static int band_map_ifft_bin(int freq_offset, int rfft_size) {
    int ifft_bin = (int)roundf((float)freq_offset * kIfftSize / rfft_size);
    if (ifft_bin < 0) ifft_bin += kIfftSize;
    return ifft_bin;
}

float composite_bin_to_rf_hz(int freq_offset, int rfft_size) {
    init_band_map();
    int ifft_bin = band_map_ifft_bin(freq_offset, rfft_size);
    for (int i = 0; i < kBandMapSize; ++i) {
        if (ifft_bin >= kBandMap[i].ifft_start && ifft_bin < kBandMap[i].ifft_end) {
            int wb_bin = kBandMap[i].wb_start + (ifft_bin - kBandMap[i].ifft_start);
            return (float)wb_bin * kWbSampleRate / kWidebandFftSize;
        }
    }
    return (float)freq_offset;
}

int composite_bin_to_band_idx(int freq_offset, int rfft_size) {
    init_band_map();
    int ifft_bin = band_map_ifft_bin(freq_offset, rfft_size);
    for (int i = 0; i < kBandMapSize; ++i) {
        if (ifft_bin >= kBandMap[i].ifft_start && ifft_bin < kBandMap[i].ifft_end)
            return i;
    }
    return -1;
}
