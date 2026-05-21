// test_ldpc_sign.cc — Verify FT8LdpcCuda sign convention and basic decode.
//
// Four test vectors:
//   1. All-zero codeword (llr[i] = -10): parity must pass, x_hat must be all zeros.
//      (All-zero is always a valid LDPC codeword; H*0=0 mod 2.)
//   2. All-zero LLR (llr[i] = 0): gradient is zero, ADMM trivially converges to
//      all-zeros codeword, so parity=true. Validates init path doesn't NaN/crash.
//   3. Sign sanity: flipped LLRs from test 1 (llr[i] = +10) must produce parity=false.
//      (If the sign fix were missing, test 1 and 3 outcomes would be swapped.)
//   4. Normalization: all-zero codeword with FT8SoftCuda-scale raw LLRs (llr[i] = -80).
//      Without normalization, over-scaled hard decisions still converge (trivial case)
//      but this validates the scale path. More importantly tests that var>0 branch works.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include "gm/cuda/FT8LdpcCuda.h"
#include "ft8_lib/ft8/constants.h"

static bool run_test(const char* name,
                     const float* llr_h,     // 174 floats
                     bool expect_parity)
{
    float*   llr_d    = nullptr;
    uint8_t* x_hat_d  = nullptr;
    bool*    parity_d = nullptr;

    cudaMalloc(&llr_d,    FTX_LDPC_N * sizeof(float));
    cudaMalloc(&x_hat_d,  FTX_LDPC_N * sizeof(uint8_t));
    cudaMalloc(&parity_d, sizeof(bool));

    // Pad to FT8_LDPC_BATCH; only slot 0 has a real input.
    float* llr_batch_d = nullptr;
    cudaMalloc(&llr_batch_d, (size_t)FT8_LDPC_BATCH * FTX_LDPC_N * sizeof(float));
    cudaMemset(llr_batch_d, 0, (size_t)FT8_LDPC_BATCH * FTX_LDPC_N * sizeof(float));
    cudaMemcpy(llr_batch_d, llr_h, FTX_LDPC_N * sizeof(float), cudaMemcpyHostToDevice);

    uint8_t* x_hat_batch_d = nullptr;
    bool*    parity_batch_d = nullptr;
    cudaMalloc(&x_hat_batch_d,  (size_t)FT8_LDPC_BATCH * FTX_LDPC_N * sizeof(uint8_t));
    cudaMalloc(&parity_batch_d, (size_t)FT8_LDPC_BATCH * sizeof(bool));

    cudaStream_t s;
    cudaStreamCreate(&s);
    ft8_ldpc_decode_batch(llr_batch_d, x_hat_batch_d, parity_batch_d, s);
    cudaStreamSynchronize(s);

    bool got_parity = false;
    uint8_t x_hat_h[FTX_LDPC_N] = {};
    cudaMemcpy(&got_parity, parity_batch_d,               sizeof(bool),    cudaMemcpyDeviceToHost);
    cudaMemcpy(x_hat_h,     x_hat_batch_d, FTX_LDPC_N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    bool ok = (got_parity == expect_parity);
    printf("[%s] %s  parity=%s (expected %s)\n",
           ok ? "PASS" : "FAIL", name,
           got_parity ? "true" : "false",
           expect_parity ? "true" : "false");

    if (expect_parity && got_parity) {
        bool all_zero = true;
        for (int i = 0; i < FTX_LDPC_N; ++i)
            if (x_hat_h[i] != 0) { all_zero = false; break; }
        printf("       x_hat all-zero: %s\n", all_zero ? "yes" : "NO (unexpected)");
        if (!all_zero) ok = false;
    }

    cudaFree(llr_d); cudaFree(x_hat_d); cudaFree(parity_d);
    cudaFree(llr_batch_d); cudaFree(x_hat_batch_d); cudaFree(parity_batch_d);
    cudaStreamDestroy(s);
    return ok;
}

int main()
{
    cudaSetDevice(0);
    ft8_ldpc_init_constants();

    int pass = 0, total = 0;

    // Test 1: all-zero codeword. Strong LLRs for bit 0 in FT8SoftCuda convention
    // (positive = bit 1, so negative = bit 0). Parity must pass; x_hat all zeros.
    {
        float llr[FTX_LDPC_N];
        for (int i = 0; i < FTX_LDPC_N; ++i) llr[i] = -10.0f;
        bool ok = run_test("all-zero codeword (llr=-10)", llr, /*expect_parity=*/true);
        if (ok) ++pass;
        ++total;
    }

    // Test 2: all-zero LLR — zero gradient, ADMM trivially settles at all-zeros codeword.
    // parity=true confirms the init path doesn't crash or NaN.
    {
        float llr[FTX_LDPC_N] = {};
        bool ok = run_test("all-zero LLR (converges to zero codeword)", llr, /*expect_parity=*/true);
        if (ok) ++pass;
        ++total;
    }

    // Test 3: sign sanity — flipped LLRs (positive = bit 0 wrong convention).
    // Without the sign fix this would decode to all zeros; with fix it must fail.
    {
        float llr[FTX_LDPC_N];
        for (int i = 0; i < FTX_LDPC_N; ++i) llr[i] = +10.0f;
        bool ok = run_test("sign sanity (flipped, llr=+10, must fail)", llr, /*expect_parity=*/false);
        if (ok) ++pass;
        ++total;
    }

    // Test 4: all-zero codeword with FT8SoftCuda-scale raw LLRs (var >> 24).
    // Before normalization fix, |llr|=80 → |qj|=40 >> rho*deg=3, x hard-clamps.
    // This still decodes (trivial codeword), but validates the normalization branch runs.
    {
        float llr[FTX_LDPC_N];
        for (int i = 0; i < FTX_LDPC_N; ++i) llr[i] = -80.0f;
        bool ok = run_test("raw-scale codeword (llr=-80, tests norm path)", llr, /*expect_parity=*/true);
        if (ok) ++pass;
        ++total;
    }

    printf("\n%d / %d tests passed\n", pass, total);
    return (pass == total) ? 0 : 1;
}
