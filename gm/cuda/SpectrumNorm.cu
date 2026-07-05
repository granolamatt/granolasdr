#include "gm/cuda/SpectrumNorm.h"
#include <cmath>
#include <cstdlib>
#include <vector>

// ── CPU: Legendre polynomial fitting ─────────────────────────────────────────
// Fits P(x) = Σ c[k] * L_k(x) to data y[0..N-1], x_i = 2i/(N-1) - 1.
// Legendre basis is orthogonal on [-1,1] → well-conditioned normal equations.

static std::vector<double> legendre_fit(const float* y, int N, int D)
{
    int m = D + 1;
    std::vector<std::vector<double>> M(m, std::vector<double>(m, 0.0));
    std::vector<double> rhs(m, 0.0);

    for (int i = 0; i < N; i++) {
        double x = (N > 1) ? (2.0 * i / (N - 1) - 1.0) : 0.0;

        // Evaluate L_0 .. L_D via three-term recurrence.
        std::vector<double> L(m);
        L[0] = 1.0;
        if (D >= 1) L[1] = x;
        for (int k = 2; k <= D; k++)
            L[k] = ((2*k - 1) * x * L[k-1] - (k-1) * L[k-2]) / k;

        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) M[j][k] += L[j] * L[k];
            rhs[j] += L[j] * (double)y[i];
        }
    }

    // Gaussian elimination with partial pivoting.
    for (int col = 0; col < m; col++) {
        int pivot = col;
        for (int row = col + 1; row < m; row++)
            if (std::abs(M[row][col]) > std::abs(M[pivot][col])) pivot = row;
        std::swap(M[col], M[pivot]);
        std::swap(rhs[col], rhs[pivot]);
        if (std::abs(M[col][col]) < 1e-30) continue;
        double inv = 1.0 / M[col][col];
        for (int row = col + 1; row < m; row++) {
            double f = M[row][col] * inv;
            for (int k = col; k < m; k++) M[row][k] -= f * M[col][k];
            rhs[row] -= f * rhs[col];
        }
    }

    std::vector<double> c(m, 0.0);
    for (int i = m - 1; i >= 0; i--) {
        if (std::abs(M[i][i]) < 1e-30) continue;
        c[i] = rhs[i];
        for (int k = i + 1; k < m; k++) c[i] -= M[i][k] * c[k];
        c[i] /= M[i][i];
    }
    return c;
}

static double legendre_eval(const std::vector<double>& c, double x)
{
    int D = (int)c.size() - 1;
    if (D < 0) return 0.0;
    double p0 = 1.0, p1 = x, val = c[0];
    if (D >= 1) val += c[1] * x;
    for (int k = 2; k <= D; k++) {
        double pk = ((2*k - 1) * x * p1 - (k - 1) * p0) / k;
        val += c[k] * pk;
        p0 = p1; p1 = pk;
    }
    return val;
}

// Public: fit poly to log_mag, write per-bin equalization gains.
// gain[i] = 10^(poly(0) - poly(xi)) → all bins equalized to center-band level.
void norm_fit_gains(const float* log_mag, int N, int D, float* gains_out)
{
    if (N <= 0) return;
    if (N == 1) { gains_out[0] = 1.0f; return; }

    auto c = legendre_fit(log_mag, N, D);
    double ref_log = legendre_eval(c, 0.0);  // target = center of band

    for (int i = 0; i < N; i++) {
        double x = 2.0 * i / (N - 1) - 1.0;
        double gain_log = ref_log - legendre_eval(c, x);
        gains_out[i] = (float)std::pow(10.0, gain_log);
    }
}

// ── GPU: apply gains ──────────────────────────────────────────────────────────

__global__ static void apply_gains_kernel(cufftComplex* data,
                                           const float* gains, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    data[i].x *= gains[i];
    data[i].y *= gains[i];
}

void norm_apply_gains(cufftComplex* data, const float* gains, int N,
                      cudaStream_t stream)
{
    if (N <= 0) return;
    apply_gains_kernel<<<(N + 255) / 256, 256, 0, stream>>>(data, gains, N);
}
