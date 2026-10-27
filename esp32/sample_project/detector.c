#include <math.h>
#include <string.h>
#include "detector.h"

void det_init(detector_t *d, int kind, int dim, float center, float sens)
{
    memset(d, 0, sizeof *d);
    d->kind = kind;
    d->dim = dim > DET_MAX_DIM ? DET_MAX_DIM : dim;
    d->center = center;
    d->sens = sens;
    d->shrink = 0.10f;      /* Ledoit-Wolf style shrinkage towards the diagonal */
    d->sigma_min = 1e-4f;
}

void det_learn(detector_t *d, const float *x)
{
    int D = d->dim;
    float delta[DET_MAX_DIM];
    d->n++;
    for (int i = 0; i < D; i++) {
        delta[i] = x[i] - d->mean[i];
        d->mean[i] += delta[i] / (float)d->n;
    }
    if (d->kind == DET_ZSM) {
        for (int i = 0; i < D; i++) d->m2[i * D + i] += delta[i] * (x[i] - d->mean[i]);
    } else {
        for (int i = 0; i < D; i++) {
            float post = x[i] - d->mean[i];
            for (int j = 0; j < D; j++) d->m2[i * D + j] += delta[j] * post;
        }
    }
    d->ready = 0;
}

int det_finalize(detector_t *d)
{
    int D = d->dim;
    float nm1 = d->n > 1 ? (float)(d->n - 1) : 1.0f;
    for (int i = 0; i < D; i++) {
        float v = d->m2[i * D + i] / nm1;
        float s = sqrtf(v > 0 ? v : 0);
        d->scale[i] = s > d->sigma_min ? s : d->sigma_min;
    }
    if (d->kind == DET_MAHA) {
        /* correlation matrix of standardised features, shrunk towards identity */
        float *L = d->L;
        for (int i = 0; i < D; i++)
            for (int j = 0; j < D; j++) {
                float c = d->m2[i * D + j] / nm1 / (d->scale[i] * d->scale[j]);
                L[i * D + j] = (i == j) ? 1.0f : (1.0f - d->shrink) * c;
            }
        for (int j = 0; j < D; j++) {                  /* in-place Cholesky, lower */
            float s = L[j * D + j];
            for (int k = 0; k < j; k++) s -= L[j * D + k] * L[j * D + k];
            if (s <= 1e-8f) return -1;
            L[j * D + j] = sqrtf(s);
            for (int i = j + 1; i < D; i++) {
                float t = L[i * D + j];
                for (int k = 0; k < j; k++) t -= L[i * D + k] * L[j * D + k];
                L[i * D + j] = t / L[j * D + j];
            }
            for (int i = 0; i < j; i++) L[i * D + j] = 0.0f;
        }
    }
    d->ready = 1;
    return 0;
}

float det_distance(const detector_t *d, const float *x)
{
    int D = d->dim;
    float z[DET_MAX_DIM], acc = 0.0f;
    for (int i = 0; i < D; i++) z[i] = (x[i] - d->mean[i]) / d->scale[i];
    if (d->kind == DET_MAHA) {                          /* solve L y = z (forward) */
        const float *L = d->L;
        for (int i = 0; i < D; i++) {
            float t = z[i];
            for (int k = 0; k < i; k++) t -= L[i * D + k] * z[k];
            z[i] = t / L[i * D + i];
        }
    }
    float zmax = 0.0f;
    for (int i = 0; i < D; i++) { acc += z[i] * z[i]; if (fabsf(z[i]) > zmax) zmax = fabsf(z[i]); }
    /* RMS of the (whitened) deviations catches broad changes; the largest single
       deviation / 3 catches a change confined to one feature (e.g. temperature),
       which the RMS alone would dilute by sqrt(D). */
    float rms = sqrtf(acc / D), peak = zmax / 3.0f;
    return rms > peak ? rms : peak;
}

uint8_t det_similarity(const detector_t *d, const float *x)
{
    float s = d->sens * (det_distance(d, x) - d->center);
    if (s > 60.0f) s = 60.0f;
    if (s < -60.0f) s = -60.0f;
    s = floorf(100.0f / (1.0f + expf(s)) + 0.5f);
    return (uint8_t)(s < 0 ? 0 : (s > 100 ? 100 : s));
}

uint32_t det_model_bytes(const detector_t *d)
{
    int D = d->dim;
    uint32_t b = 4u * (2 * D) + 16;                      /* mean + scale + header */
    if (d->kind == DET_MAHA) b += 4u * (2 * D * D);      /* co-moment + Cholesky factor */
    else b += 4u * D;
    return b;
}

void det_adapt(detector_t *d, const float *x, float alpha)
{
    if (!d->ready || d->kind != DET_ZSM) return;
    for (int i = 0; i < d->dim; i++) {
        float delta = x[i] - d->mean[i];
        d->mean[i] += alpha * delta;
        float var = (1.0f - alpha) * (d->scale[i] * d->scale[i] + alpha * delta * delta);
        float sd = sqrtf(var);
        d->scale[i] = sd > d->sigma_min ? sd : d->sigma_min;
    }
}
