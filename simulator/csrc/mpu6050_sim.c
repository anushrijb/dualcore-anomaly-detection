#include <math.h>
#include "mpu6050_sim.h"

#define TWO_PI 6.283185307179586

/* -------- counter-based RNG: deterministic noise for any sample index ---- */
static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static inline double u01(uint64_t h) { return ((h >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static inline double gauss(uint64_t key)          /* Box-Muller from two hashes */
{
    double u1 = u01(splitmix64(key)), u2 = u01(splitmix64(key ^ 0xA5A5A5A5A5A5A5A5ull));
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

void motor_default(motor_cfg_t *c, uint64_t seed)
{
    c->seed = seed;
    c->fs = 1000.0f;
    c->rpm = 1480.0f;
    c->speed_drift = 0.01f;
    c->load_amp = 0.25f;
    c->load_period_s = 240.0f;
    c->a1 = 0.05f;
    c->case_temp_c = 40.0f;
    c->ambient_amp = 0.5f;
    c->ambient_period_s = 3600.0f;
    c->fault = FAULT_NONE;
    c->onset_s = 1e9f;
    c->ramp_s = 600.0f;
    c->tau_s = 240.0f;
}

float motor_severity(const motor_cfg_t *c, double t)
{
    if (c->fault == FAULT_NONE || t <= c->onset_s) return 0.0f;
    double s = (t - c->onset_s) / c->ramp_s;
    return (float)(s > 1.0 ? 1.0 : s);
}

/* steady-state temperature rise (degC) of each fault at full severity */
static const float HEAT_C[FAULT_N] = {0.0f, 3.0f, 5.0f, 6.0f, 2.0f, 16.0f};

/* first-order thermal lag of a linear ramp 0 -> K over R seconds, closed form */
static double lagged_ramp(double tp, double R, double tau, double K)
{
    if (tp <= 0) return 0.0;
    if (tp <= R) return K / R * (tp - tau * (1.0 - exp(-tp / tau)));
    double yR = K / R * (R - tau * (1.0 - exp(-R / tau)));
    double e = exp(-(tp - R) / tau);
    return yR * e + K * (1.0 - e);
}

static double load_frac(const motor_cfg_t *c, double t)
{
    double ph = (double)(c->seed % 997) * 0.0063;
    return sin(TWO_PI * t / c->load_period_s + ph);
}

float motor_case_temp(const motor_cfg_t *c, double t)
{
    double w = TWO_PI / c->load_period_s, wt = w * c->tau_s;
    double ph = (double)(c->seed % 997) * 0.0063;
    /* load effect passed through the thermal lag (steady-state sinusoid) */
    double load = 0.8 * c->load_amp / 0.25 * sin(TWO_PI * t / c->load_period_s + ph - atan(wt)) / sqrt(1 + wt * wt);
    double amb = c->ambient_amp * sin(TWO_PI * t / c->ambient_period_s + (double)(c->seed % 389) * 0.016);
    double heat = 0.0;
    if (c->fault != FAULT_NONE)
        heat = lagged_ramp(t - c->onset_s, c->ramp_s, c->tau_s, HEAT_C[c->fault]);
    return (float)(c->case_temp_c + load * 3.0 + amb + heat);
}

/* shaft angle with slow speed drift, closed form */
static double shaft_phase(const motor_cfg_t *c, double t, double *f_inst)
{
    double f0 = c->rpm / 60.0, Ts = 97.0 + (double)(c->seed % 53);
    double d = c->speed_drift;
    *f_inst = f0 * (1.0 + d * sin(TWO_PI * t / Ts));
    return TWO_PI * f0 * (t - d * Ts / TWO_PI * cos(TWO_PI * t / Ts)) + (double)(c->seed % 101) * 0.06;
}

void mpu6050_read(const motor_cfg_t *c, uint64_t n, int16_t raw[MPU_REGS])
{
    double t = (double)n / c->fs, f_r;
    double th = shaft_phase(c, t, &f_r);
    double s = motor_severity(c, t);
    double load = 1.0 + c->load_amp * load_frac(c, t);
    double A1 = c->a1 * load;
    uint64_t key = splitmix64(c->seed * 0x100000001B3ull + n);

    /* ---- healthy machine: 1x orbit, small 2x, 100 Hz electrical, noise --- */
    double ax = A1 * sin(th) + 0.15 * A1 * sin(2 * th + 0.4) + 0.006 * sin(TWO_PI * 100.0 * t);
    double ay = A1 * cos(th) + 0.15 * A1 * cos(2 * th + 0.4) + 0.006 * cos(TWO_PI * 100.0 * t);
    double az = 0.25 * A1 * sin(th + 1.1);

    /* ---- faults (amplitudes at full severity) ----------------------------- */
    switch (c->fault) {
    case FAULT_IMBALANCE:
        ax += 3.0 * s * A1 * sin(th);
        ay += 3.0 * s * A1 * cos(th);
        break;
    case FAULT_MISALIGNMENT:
        ax += 0.12 * s * sin(2 * th + 0.9);
        ay += 0.12 * s * cos(2 * th + 0.9);
        az += 0.06 * s * sin(th + 0.3) + 0.08 * s * sin(2 * th + 1.7);
        break;
    case FAULT_BEARING: {               /* BPFO = 3.585 x shaft, resonance 330 Hz */
        double pb = 3.585 * th / TWO_PI, frac = pb - floor(pb);
        double fb = 3.585 * f_r, wr = TWO_PI * 330.0, zeta = 0.06;
        double imp = 0.0;
        for (int k = 0; k < 2; k++) {                       /* last two impacts */
            double dt = (frac + k) / fb;
            uint64_t idx = (uint64_t)(floor(pb) - k + 1e6);
            double amp = 0.5 * (0.7 + 0.6 * u01(splitmix64(c->seed ^ (idx * 0x9E37ull))));
            imp += amp * exp(-zeta * wr * dt) * sin(wr * dt);
        }
        ax += s * imp;
        ay += s * 0.6 * imp;
        az += s * 0.3 * imp;
        ax += s * 0.01 * gauss(key ^ 0x77);
        break; }
    case FAULT_LOOSENESS: {
        double rect = fabs(sin(th)) - 0.6366;               /* rectified: rich harmonics */
        ax += s * (0.25 * rect + 0.07 * sin(0.5 * th) + 0.06 * sin(3 * th) + 0.05 * sin(4 * th));
        ay += s * (0.15 * rect + 0.05 * cos(0.5 * th) + 0.05 * cos(3 * th));
        az += s * 0.10 * rect;
        break; }
    default: break;
    }

    /* gyroscope: base rocking driven by the radial vibration */
    double gx = 8.0 * ay + 0.4 * sin(TWO_PI * 0.7 * t);
    double gy = -8.0 * ax;
    double gz = 2.0 * az;

    /* sensor noise (MPU6050 datasheet densities over 500 Hz bandwidth) */
    ax += 0.009 * gauss(key ^ 1); ay += 0.009 * gauss(key ^ 2); az += 0.009 * gauss(key ^ 3);
    gx += 0.11 * gauss(key ^ 4);  gy += 0.11 * gauss(key ^ 5);  gz += 0.11 * gauss(key ^ 6);
    az += 1.0;                                         /* gravity, sensor flat */
    double temp = motor_case_temp(c, t) + 0.05 * gauss(key ^ 7);

    double v[MPU_REGS] = {ax * MPU_ACCEL_LSB_PER_G, ay * MPU_ACCEL_LSB_PER_G, az * MPU_ACCEL_LSB_PER_G,
                          (temp - MPU_TEMP_OFFSET_C) * MPU_TEMP_LSB_PER_C,
                          gx * MPU_GYRO_LSB_PER_DPS, gy * MPU_GYRO_LSB_PER_DPS, gz * MPU_GYRO_LSB_PER_DPS};
    for (int i = 0; i < MPU_REGS; i++) {
        double r = floor(v[i] + 0.5);
        raw[i] = (int16_t)(r > 32767 ? 32767 : (r < -32768 ? -32768 : r));
    }
}

void mpu6050_to_phys(const int16_t raw[MPU_REGS], float out[MPU_REGS])
{
    out[0] = raw[0] / MPU_ACCEL_LSB_PER_G;
    out[1] = raw[1] / MPU_ACCEL_LSB_PER_G;
    out[2] = raw[2] / MPU_ACCEL_LSB_PER_G;
    out[3] = raw[3] / MPU_TEMP_LSB_PER_C + MPU_TEMP_OFFSET_C;
    out[4] = raw[4] / MPU_GYRO_LSB_PER_DPS;
    out[5] = raw[5] / MPU_GYRO_LSB_PER_DPS;
    out[6] = raw[6] / MPU_GYRO_LSB_PER_DPS;
}

float mpu6050_i2c_read_us(void)
{
    /* START + addr(W) + reg + RESTART + addr(R) + 14 data bytes = 17 bytes x 9 bits */
    return (17.0f * 9.0f) / 400e3f * 1e6f + 2.5f;   /* + start/stop/turnaround */
}
