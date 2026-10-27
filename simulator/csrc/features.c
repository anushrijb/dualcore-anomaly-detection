#include <math.h>
#include <string.h>
#include "dsp_features.h"

#ifndef FFT_MAX
#define FFT_MAX 4096          /* edge build: -DFFT_MAX=256 (only the 256-pt FFT is needed) */
#endif
static float TW_C[FFT_MAX / 2], TW_S[FFT_MAX / 2], HANN[WIN];
static int g_fft_ready = 0;

const char *FEATURE_NAMES[N_FEAT] = {
    "ax_rms", "ax_p2p", "ax_kurt", "ax_crest", "ay_rms", "ay_p2p", "ay_kurt", "ay_crest",
    "az_rms", "az_p2p", "az_kurt", "az_crest", "gx_rms", "gx_p2p", "gx_kurt", "gx_crest",
    "gy_rms", "gy_p2p", "gy_kurt", "gy_crest", "gz_rms", "gz_p2p", "gz_kurt", "gz_crest",
    "band0", "band1", "band2", "band3", "band4", "band5", "temp"};

void fft_init(void)
{
    if (g_fft_ready) return;
    for (int i = 0; i < FFT_MAX / 2; i++) {
        TW_C[i] = (float)cos(-6.283185307179586 * i / FFT_MAX);
        TW_S[i] = (float)sin(-6.283185307179586 * i / FFT_MAX);
    }
    for (int i = 0; i < WIN; i++) HANN[i] = (float)(0.5 - 0.5 * cos(6.283185307179586 * i / (WIN - 1)));
    g_fft_ready = 1;
}

void fft_radix2(float *re, float *im, int n)
{
    for (int i = 1, j = 0; i < n; i++) {                       /* bit reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int step = FFT_MAX / len, half = len >> 1;
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < half; k++) {
                float wr = TW_C[k * step], wi = TW_S[k * step];
                float xr = re[i + k + half], xi = im[i + k + half];
                float tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
                re[i + k + half] = re[i + k] - tr; im[i + k + half] = im[i + k] - ti;
                re[i + k] += tr; im[i + k] += ti;
            }
    }
}

/* band edges as fractions of fs: 0-40, 40-80, 80-150, 150-250, 250-350, 350-500 Hz at 1 kHz */
static const float BAND_EDGE[N_BANDS + 1] = {0.0f, 0.04f, 0.08f, 0.15f, 0.25f, 0.35f, 0.5001f};

void features_window(const float x[WIN * MPU_REGS], float fs, float out[N_FEAT])
{
    static const int AXIS[6] = {0, 1, 2, 4, 5, 6};             /* skip temp channel */
    float ch[WIN];
    for (int a = 0; a < 6; a++) {
        int c = AXIS[a];
        float mean = 0.0f;
        for (int i = 0; i < WIN; i++) { ch[i] = x[i * MPU_REGS + c]; mean += ch[i]; }
        mean /= WIN;
        float m2 = 0.0f, m4 = 0.0f, mx = -1e30f, mn = 1e30f, amax = 0.0f;
        for (int i = 0; i < WIN; i++) {
            float v = ch[i] - mean, v2 = v * v;
            m2 += v2; m4 += v2 * v2;
            if (ch[i] > mx) mx = ch[i];
            if (ch[i] < mn) mn = ch[i];
            if (fabsf(v) > amax) amax = fabsf(v);
        }
        m2 /= WIN; m4 /= WIN;
        float rms = sqrtf(m2) + 1e-9f;
        out[a * 4 + 0] = rms;
        out[a * 4 + 1] = mx - mn;
        out[a * 4 + 2] = m4 / (m2 * m2 + 1e-18f);
        out[a * 4 + 3] = amax / rms;
    }
    /* spectrum of radial accel */
    float re[WIN], im[WIN], mean = 0.0f;
    for (int i = 0; i < WIN; i++) mean += x[i * MPU_REGS + 0];
    mean /= WIN;
    for (int i = 0; i < WIN; i++) { re[i] = (x[i * MPU_REGS + 0] - mean) * HANN[i]; im[i] = 0.0f; }
    fft_radix2(re, im, WIN);
    float band[N_BANDS] = {0};
    for (int k = 1; k <= WIN / 2; k++) {
        float f = (float)k / WIN;                                /* fraction of fs */
        float p = (re[k] * re[k] + im[k] * im[k]) / (WIN * WIN);
        for (int b = 0; b < N_BANDS; b++)
            if (f >= BAND_EDGE[b] && f < BAND_EDGE[b + 1]) { band[b] += p; break; }
    }
    for (int b = 0; b < N_BANDS; b++) out[24 + b] = log10f(band[b] + 1e-12f);
    float tm = 0.0f;
    for (int i = 0; i < WIN; i++) tm += x[i * MPU_REGS + 3];
    out[30] = tm / WIN;
    (void)fs;
}

void features_window_raw(const int16_t raw[WIN * MPU_REGS], float fs, float out[N_FEAT])
{
    float x[WIN * MPU_REGS];
    for (int i = 0; i < WIN; i++) mpu6050_to_phys(&raw[i * MPU_REGS], &x[i * MPU_REGS]);
    features_window(x, fs, out);
}

float envelope_bpfo_ratio(const float *ax, int n, float fs, float shaft_hz)
{
    float re[FFT_MAX], im[FFT_MAX];
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += ax[i];
    mean /= n;
    for (int i = 0; i < n; i++) { re[i] = ax[i] - mean; im[i] = 0.0f; }
    fft_radix2(re, im, n);
    /* analytic signal: keep 150-500 Hz (resonance band), double positive freqs, zero negatives */
    for (int k = 0; k < n; k++) {
        float f = (k <= n / 2 ? k : n - k) * fs / n;
        int keep = (k > 0 && k < n / 2 && f >= 150.0f);
        if (keep) { re[k] *= 2.0f; im[k] *= 2.0f; } else { re[k] = 0.0f; im[k] = 0.0f; }
    }
    for (int k = 0; k < n; k++) im[k] = -im[k];                  /* inverse via conj */
    fft_radix2(re, im, n);
    for (int i = 0; i < n; i++) {                                /* envelope */
        float r = re[i] / n, q = -im[i] / n;
        re[i] = sqrtf(r * r + q * q);
    }
    mean = 0.0f;
    for (int i = 0; i < n; i++) mean += re[i];
    mean /= n;
    for (int i = 0; i < n; i++) { re[i] -= mean; im[i] = 0.0f; }
    fft_radix2(re, im, n);
    float bpfo = 3.585f * shaft_hz, on = 0.0f, off = 0.0f;
    int non = 0, noff = 0;
    for (int k = 1; k < n / 2; k++) {
        float f = k * fs / n, p = re[k] * re[k] + im[k] * im[k];
        int harm = 0;
        for (int h = 1; h <= 3; h++) if (fabsf(f - h * bpfo) < 2.5f * fs / n) harm = 1;
        if (harm) { on += p; non++; } else if (f < 4 * bpfo) { off += p; noff++; }
    }
    return (on / (non ? non : 1)) / (off / (noff ? noff : 1) + 1e-20f);
}
