#pragma once
#include <complex>

namespace gm {
namespace hf {

// Shared FT8/JS8 per-candidate refine core (CPU, no CUDA).
//
// Input: a decimated complex baseband frame of the candidate (79 symbols ×
// NSPS samples at 12.8 kHz, coarse-centered near DC by the caller's downconvert).
// Does the continuous fine freq/time alignment that beats the discrete STFT grid
// (Costas-energy search), then extracts 174 max-log LLRs (non-coherent |z|;
// coherent phase is a dead end for FT8/JS8 CPM — see docs/coherent_overlap_findings.md).
//
// mode selects the Costas pattern (drives sync) and the tone→bit map (drives the
// LLR bit order): FT8 is Gray, JS8 is natural binary. Caller runs LDPC/CRC on the
// returned LLRs (ft8_lib for FT8, js8 for JS8).
struct RefineMode {
    int costas[7];       // FT8 {3,1,4,0,6,5,2}; JS8 Normal {4,2,5,6,1,3,0}
    int tone_map[8];     // FT8 Gray {0,1,3,2,5,6,4,7}; JS8 natural {0..7}
};

extern const RefineMode kFT8Refine;
extern const RefineMode kJS8Refine;

constexpr int kRefineSr   = 12800;           // decimated frame sample rate
constexpr int kRefineNsps = 2048;            // samples/symbol (sr * 0.16)
constexpr int kRefineNsym = 79;
constexpr int kRefineFrame = kRefineNsym * kRefineNsps;

// frame: kRefineFrame complex samples. Fills log174[174] (un-normalized; the
// LDPC layer normalizes). Returns the best Costas sync energy (higher = better).
// dechirp (default off): also search a linear frequency drift — experimental,
// opt-in via REFINE_DECHIRP=1 at the call site (see the DE-CHIRP block in .cc).
float refine_llr(const std::complex<float>* frame, int frame_len,
                 const RefineMode& mode, float* log174, bool dechirp = false);

// Reads the REFINE_DECHIRP env toggle once (default off). Production call sites
// pass this into refine_llr's `dechirp`; tests pass an explicit bool.
bool dechirp_enabled();

// Downconvert (mix by -freq_hz) + decimate a raw composite frame at sr_in to the
// kRefineSr baseband frame refine_llr expects. Windowed-sinc FIR lowpass. `out`
// holds n_out complex samples (typically kRefineFrame). This is the CPU
// extraction the pipeline runs on a D2H'd complex-ring frame before refine_llr.
void extract_frame(const std::complex<float>* in, int n_in, float freq_hz,
                   int sr_in, std::complex<float>* out, int n_out);

} // namespace hf
} // namespace gm
