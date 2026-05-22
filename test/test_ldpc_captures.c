// test_ldpc_captures.c — Validate LLR capture file by re-running ftx_decode_from_llr.
//
// Usage: ./test_ldpc_captures <capture_file.bin>
//
// Reads the binary capture file written by granolasdr (LDPC_CAPTURE env var),
// calls ftx_decode_from_llr() on each vector, and reports the decode rate.
// This proves the captures contain exactly what ft8_lib's decoder sees.
// Expected: ~100% decode rate (by construction — captures were saved on decode success).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/constants.h"

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <capture_file.bin>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 1; }

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (sz % (FTX_LDPC_N * sizeof(float)) != 0) {
        fprintf(stderr, "File size %ld not divisible by %d\n",
                sz, (int)(FTX_LDPC_N * sizeof(float)));
        fclose(f);
        return 1;
    }

    int n_vecs = (int)(sz / (FTX_LDPC_N * sizeof(float)));
    printf("Loaded %d LLR vectors from %s\n", n_vecs, argv[1]);

    float* buf = malloc(FTX_LDPC_N * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM\n"); return 1; }

    int n_pass = 0, n_fail = 0;
    for (int i = 0; i < n_vecs; ++i) {
        if (fread(buf, sizeof(float), FTX_LDPC_N, f) != (size_t)FTX_LDPC_N) {
            fprintf(stderr, "Short read at vector %d\n", i);
            break;
        }
        ftx_message_t msg;
        ftx_decode_status_t status;
        bool ok = ftx_decode_from_llr(buf, 25, &msg, &status);
        if (ok) ++n_pass; else ++n_fail;
    }

    printf("Results: %d / %d decoded (%.1f%%)\n", n_pass, n_vecs,
           n_vecs > 0 ? 100.0 * n_pass / n_vecs : 0.0);
    printf("  (expect ~100%% — captures were saved on decode success)\n");

    free(buf);
    fclose(f);
    return (n_pass >= n_vecs * 9 / 10) ? 0 : 1;  // pass if ≥90% decode
}
