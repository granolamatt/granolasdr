// coherent_offline.cc — offline harness for coherent-demod experiments.
//
// Reads a --record GNLH capture (the complex FT8/JS8 composite, 409.6 kHz,
// full phase) so we can prototype coherent vs non-coherent demod off-line,
// deterministically, with no radio and no CUDA.  This first stage is just the
// reader + a sanity report on the stream; candidate detection and the
// coherent/non-coherent demod comparison build on top of it.
//
// Build (host-only): add_executable(coherent_offline coherent_offline.cc)
// Run:   ./coherent_offline raw.dat [--stats]

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Must match gm/buffer/BufferFile.h BufferFileHeader (packed, 32 bytes).
#pragma pack(push, 1)
struct GnlhHeader {
    uint32_t magic;             // 0x474E4C48 'GNLH'
    uint32_t version;
    uint32_t block_samples;     // complex<float> per block
    uint32_t element_bytes;     // sizeof(complex<float>) == 8
    uint64_t block_interval_ns;
    uint32_t sample_rate_hz;
    uint32_t rx_sample_rate;
};
#pragma pack(pop)
static const uint32_t kMagic = 0x474E4C48u;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <recording.dat> [--stats]\n", argv[0]);
        return 1;
    }
    const char* path  = argv[1];
    bool        stats = (argc > 2 && strcmp(argv[2], "--stats") == 0);

    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    GnlhHeader h;
    if (fread(&h, sizeof(h), 1, fp) != 1) {
        fprintf(stderr, "short read on header\n"); fclose(fp); return 1;
    }
    if (h.magic != kMagic) {
        fprintf(stderr, "bad magic 0x%08x (expected 0x%08x)\n", h.magic, kMagic);
        fclose(fp); return 1;
    }
    if (h.element_bytes != sizeof(std::complex<float>)) {
        fprintf(stderr, "element_bytes=%u, expected %zu (complex<float>)\n",
                h.element_bytes, sizeof(std::complex<float>));
        fclose(fp); return 1;
    }

    // File geometry.
    fseek(fp, 0, SEEK_END);
    long file_bytes = ftell(fp);
    long data_bytes = file_bytes - (long)sizeof(GnlhHeader);
    size_t bytes_per_block = (size_t)h.block_samples * h.element_bytes;
    long   n_blocks = data_bytes / (long)bytes_per_block;
    double block_s  = (double)h.block_interval_ns / 1e9;
    double dur_s    = n_blocks * block_s;

    printf("GNLH recording: %s\n", path);
    printf("  version         %u\n", h.version);
    printf("  block_samples   %u  (%.3f ms/block)\n", h.block_samples, block_s * 1e3);
    printf("  sample_rate     %u Hz\n", h.sample_rate_hz);
    printf("  rx_sample_rate  %u Hz\n", h.rx_sample_rate);
    printf("  blocks          %ld   (%.1f s, %.2f min)\n", n_blocks, dur_s, dur_s / 60.0);
    printf("  data            %.1f MB\n", data_bytes / 1e6);

    if (!stats) { fclose(fp); return 0; }

    // Sanity pass: per-block RMS + global peak, to confirm the stream has signal
    // and see its level range (also a quick look for dead/clipped stretches).
    fseek(fp, sizeof(GnlhHeader), SEEK_SET);
    std::vector<std::complex<float>> blk(h.block_samples);
    double sum_rms = 0, min_rms = 1e30, max_rms = 0, peak = 0;
    long read_blocks = 0;
    while (fread(blk.data(), bytes_per_block, 1, fp) == 1) {
        double e = 0;
        for (auto& c : blk) {
            double m = std::norm(c);   // |c|^2
            e += m;
            if (m > peak) peak = m;
        }
        double rms = std::sqrt(e / h.block_samples);
        sum_rms += rms;
        if (rms < min_rms) min_rms = rms;
        if (rms > max_rms) max_rms = rms;
        ++read_blocks;
    }
    fclose(fp);
    if (read_blocks) {
        printf("\nstats over %ld blocks:\n", read_blocks);
        printf("  block RMS   min=%.3g  mean=%.3g  max=%.3g\n",
               min_rms, sum_rms / read_blocks, max_rms);
        printf("  sample peak |c|=%.3g\n", std::sqrt(peak));
    }
    return 0;
}
