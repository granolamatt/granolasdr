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

// ── Wideband equalization (whole-spectrum) ───────────────────────────────────

__global__ static void weq_ema_kernel(const cufftComplex* fft, float* ema, int N,
                                      float a_down, float a_up, int cold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float re = fft[i].x, im = fft[i].y;
    float mag = sqrtf(re * re + im * im);
    if (cold) { ema[i] = mag; return; }
    float e = ema[i];
    float a = (mag < e) ? a_down : a_up;   // track down fast, up slow (noise floor)
    ema[i] = (1.0f - a) * e + a * mag;
}

void weq_update_ema(const cufftComplex* fft, float* ema, int N,
                    float a_down, float a_up, bool cold, cudaStream_t stream)
{
    if (N <= 0) return;
    weq_ema_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
        fft, ema, N, a_down, a_up, cold ? 1 : 0);
}

// Pass 1: logema[i] = log10(ema[i]); accumulate the global sum (self-cal target).
__global__ static void weq_sumlog_kernel(const float* ema, float* logema, int N,
                                         float* sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float l = log10f(ema[i] + 1e-30f);
    logema[i] = l;
    atomicAdd(sum, l);
}

// Pass 2: box-filter logema over ±hw for the local floor; gain flattens to the
// global mean (target).  Wide window => narrow carriers averaged out, not notched.
__global__ static void weq_gain_kernel(const float* logema, float* gain, int N,
                                       int hw, const float* sum, float max_gain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float target_log = *sum / (float)N;

    int lo = i - hw; if (lo < 0) lo = 0;
    int hi = i + hw; if (hi > N - 1) hi = N - 1;
    float s = 0.0f; int c = 0;
    for (int j = lo; j <= hi; ++j) { s += logema[j]; ++c; }
    float floor_log = s / (float)c;

    float g = powf(10.0f, target_log - floor_log);
    float mn = 1.0f / max_gain;
    if (g > max_gain) g = max_gain;
    if (g < mn)       g = mn;
    gain[i] = g;
}

void weq_compute_gains(const float* ema, float* logema, float* gain, float* sum,
                       int N, int half_win, float max_gain, cudaStream_t stream)
{
    if (N <= 0) return;
    cudaMemsetAsync(sum, 0, sizeof(float), stream);
    weq_sumlog_kernel<<<(N + 255) / 256, 256, 0, stream>>>(ema, logema, N, sum);
    weq_gain_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
        logema, gain, N, half_win, sum, max_gain);
}

__global__ static void weq_apply_interp_kernel(cufftComplex* data,
                                               const float* gprev, const float* gtarg,
                                               int N, float t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = gprev[i] + t * (gtarg[i] - gprev[i]);
    data[i].x *= g;
    data[i].y *= g;
}

void weq_apply_interp(cufftComplex* data, const float* gprev, const float* gtarg,
                      int N, float t, cudaStream_t stream)
{
    if (N <= 0) return;
    weq_apply_interp_kernel<<<(N + 255) / 256, 256, 0, stream>>>(
        data, gprev, gtarg, N, t);
}
