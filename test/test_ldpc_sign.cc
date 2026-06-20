// test_ldpc_sign.cc — Smoke tests for FT8 SP-flooded LDPC decoder.
//
// These tests primarily verify that the decoder runs without crashing and that
// the all-zeros guard is active.  A full functional decode test requires a real
// FT8 codeword (encoder not available here).
//
// Tests 1, 2, 4: all-zero hard decisions → all-zeros guard fires → parity=false.
// Test 3: sign sanity — all-one hard decisions → H*1 has parity violations → false.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include "gm/cuda/FT8LdpcCuda.h"
#include "ft8_lib/ft8/constants.h"

static bool run_test(const char* name,
                     const float* llr_h,
                     bool expect_parity)
{
    float*   llr_batch_d   = nullptr;
    uint8_t* x_hat_batch_d = nullptr;
    bool*    parity_batch_d = nullptr;

    cudaMalloc(&llr_batch_d,    FTX_LDPC_N * sizeof(float));
    cudaMalloc(&x_hat_batch_d,  FTX_LDPC_N * sizeof(uint8_t));
    cudaMalloc(&parity_batch_d, sizeof(bool));

    cudaMemcpy(llr_batch_d, llr_h, FTX_LDPC_N * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t s;
    cudaStreamCreate(&s);
    ft8_sp_flooded_decode_batch(1, llr_batch_d, x_hat_batch_d, parity_batch_d, s);
    cudaStreamSynchronize(s);

    bool    got_parity = false;
    uint8_t x_hat_h[FTX_LDPC_N] = {};
    cudaMemcpy(&got_parity, parity_batch_d,               sizeof(bool),    cudaMemcpyDeviceToHost);
    cudaMemcpy(x_hat_h,     x_hat_batch_d, FTX_LDPC_N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    bool ok = (got_parity == expect_parity);
    printf("[%s] %s  parity=%s (expected %s)\n",
           ok ? "PASS" : "FAIL", name,
           got_parity ? "true" : "false",
           expect_parity ? "true" : "false");

    cudaFree(llr_batch_d); cudaFree(x_hat_batch_d); cudaFree(parity_batch_d);
    cudaStreamDestroy(s);
    return ok;
}

int main()
{
    cudaSetDevice(0);
    ft8_ldpc_init_constants();

    int pass = 0, total = 0;

    // Test 1: strong bit-0 LLRs → all-zero hard decisions → all-zeros guard fires.
    {
        float llr[FTX_LDPC_N];
        for (int i = 0; i < FTX_LDPC_N; ++i) llr[i] = -10.0f;
        bool ok = run_test("all-zero codeword (llr=-10, all-zeros guard)", llr, /*expect_parity=*/false);
        if (ok) ++pass;
        ++total;
    }

    // Test 2: all-zero LLR → zero variance path (s_scale=1), zero hard decisions.
    {
        float llr[FTX_LDPC_N] = {};
        bool ok = run_test("all-zero LLR (zero variance path, no crash)", llr, /*expect_parity=*/false);
        if (ok) ++pass;
        ++total;
    }

    // Test 3: all-one hard decisions → H*1 has parity violations (most rows have odd degree).
    {
        float llr[FTX_LDPC_N];
        for (int i = 0; i < FTX_LDPC_N; ++i) llr[i] = +10.0f;
        bool ok = run_test("all-ones hard decision (parity must fail)", llr, /*expect_parity=*/false);
        if (ok) ++pass;
        ++total;
    }

    // Test 4: over-scale LLRs → normalization branch, still all-zero decisions.
    {
        float llr[FTX_LDPC_N];
        for (int i = 0; i < FTX_LDPC_N; ++i) llr[i] = -80.0f;
        bool ok = run_test("raw-scale codeword (llr=-80, tests norm path)", llr, /*expect_parity=*/false);
        if (ok) ++pass;
        ++total;
    }

    printf("\n%d / %d tests passed\n", pass, total);
    return (pass == total) ? 0 : 1;
}
