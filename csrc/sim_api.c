/* Batch helpers exported to Python (ctypes) - same C code as the edge node. */
#include <stdlib.h>
#include <string.h>
#include "mpu6050_sim.h"
#include "dsp_features.h"
#include "detector.h"

int api_n_feat(void) { return N_FEAT; }
int api_win(void) { return WIN; }
int api_hop(void) { return HOP; }
int api_det_sizeof(void) { return (int)sizeof(detector_t); }
int api_cfg_sizeof(void) { return (int)sizeof(motor_cfg_t); }
void api_motor_default(motor_cfg_t *c, unsigned long long seed) { motor_default(c, seed); }
float api_i2c_us(void) { return mpu6050_i2c_read_us(); }

/* features of n_windows windows (window k starts at sample k*HOP), OpenMP parallel */
void api_session_features(const motor_cfg_t *c, int n_windows, float *feat)
{
    fft_init();
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n_windows; k++) {
        int16_t raw[WIN * MPU_REGS];
        for (int i = 0; i < WIN; i++)
            mpu6050_read(c, (uint64_t)k * HOP + i, &raw[i * MPU_REGS]);
        features_window_raw(raw, c->fs, &feat[(size_t)k * N_FEAT]);
    }
}

void api_raw_samples(const motor_cfg_t *c, long long n0, int n, short *out)
{
    for (int i = 0; i < n; i++) mpu6050_read(c, (uint64_t)(n0 + i), (int16_t *)&out[i * MPU_REGS]);
}

void api_case_temp(const motor_cfg_t *c, int n, double t0, double dt, float *out)
{
    for (int i = 0; i < n; i++) out[i] = motor_case_temp(c, t0 + i * dt);
}

void api_severity(const motor_cfg_t *c, int n, double t0, double dt, float *out)
{
    for (int i = 0; i < n; i++) out[i] = motor_severity(c, t0 + i * dt);
}

void api_det_init(detector_t *d, int kind, int dim, float center, float sens) { det_init(d, kind, dim, center, sens); }
void api_det_learn_batch(detector_t *d, const float *X, int n)
{
    for (int i = 0; i < n; i++) det_learn(d, &X[(size_t)i * d->dim]);
}
int api_det_finalize(detector_t *d) { return det_finalize(d); }
void api_det_score_batch(const detector_t *d, const float *X, int n, float *dist, unsigned char *sim)
{
    for (int i = 0; i < n; i++) {
        dist[i] = det_distance(d, &X[(size_t)i * d->dim]);
        sim[i] = det_similarity(d, &X[(size_t)i * d->dim]);
    }
}
unsigned api_det_model_bytes(const detector_t *d) { return det_model_bytes(d); }
void api_det_set_floor(detector_t *d, int i, float v) { if (i >= 0 && i < DET_MAX_DIM) d->floor[i] = v; }
