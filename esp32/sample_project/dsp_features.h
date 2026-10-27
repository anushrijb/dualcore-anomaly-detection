/* ---------------------------------------------------------------------------
 * features.h - window feature extraction (runs on ESP32 Core 1 and on fog/cloud)
 *
 * Window = WIN samples of the 7 MPU6050 channels (physical units).
 * Feature vector (N_FEAT = 31):
 *   per axis ax ay az gx gy gz (6 x 4 = 24): RMS (AC), peak-to-peak, kurtosis, crest factor
 *   radial accel (ax) spectrum, Hann + 256-pt FFT, log10 energy in 6 bands  (6)
 *   mean temperature                                                        (1)
 * ------------------------------------------------------------------------- */
#ifndef DSP_FEATURES_H
#define DSP_FEATURES_H
#include <stdint.h>
#include "mpu6050_sim.h"

#define WIN      256
#define HOP      128
#define N_FEAT   31
#define N_BANDS  6

void fft_init(void);                                   /* call once before use */
void fft_radix2(float *re, float *im, int n);          /* in place, n = 2^k <= 4096 */

/* x: WIN samples x MPU_REGS channels, interleaved as read from the sensor */
void features_window(const float x[WIN * MPU_REGS], float fs, float out[N_FEAT]);
void features_window_raw(const int16_t raw[WIN * MPU_REGS], float fs, float out[N_FEAT]);

/* Fog "deep" analysis: envelope spectrum of radial accel over n samples
 * (n = 4 * WIN); returns the envelope energy ratio at the bearing fault
 * frequency band (BPFO harmonics) vs background. */
float envelope_bpfo_ratio(const float *ax, int n, float fs, float shaft_hz);

extern const char *FEATURE_NAMES[N_FEAT];
#endif
