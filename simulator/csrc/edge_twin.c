#include <math.h>
#include <string.h>
#include "edge_twin.h"
#include "dsp_features.h"
#include "detector.h"

const char *TWIN_FAULT_NAMES[MSF_N] = {"none", "imbalance", "bearing", "misalign", "looseness",
                                       "overload", "cooling", "humidity"};

static ms_plant_t P;               /* Core 0 only */
static ms_windower_t WD;           /* Core 1 only */
static ms_window_t W;
static detector_t DET;
static ms_ctrl_t CTRL;
static float LEARN_S, RPM_NOM;
static uint32_t WIN_N, RX_AT_LAST;
static volatile int RESTART_REQ;
static float PH_B, PH_H;           /* bearing-pass and half-speed phases (float plant) */
static volatile twin_status_t ST;  /* written by Core 1, read by the UI task */

void twin_init(float learn_s, float center, float sens, uint64_t seed)
{
    ms_cfg_t c;
    ms_cfg_default(&c, seed);
    c.load_amp = 0.15f;            /* load cycle short enough to be seen while learning */
    c.load_period_s = 20.0f;
    c.onset_s = 1e9f;              /* faults come from the potentiometer, not a timer */
    ms_plant_init(&P, &c);
    PH_B = PH_H = 0.0f;
    RPM_NOM = c.rpm_nom;

    fft_init();
    det_init(&DET, DET_ZSM, MS_NF, center, sens);
    ms_detector_setup(&DET);
    ms_ctrl_init(&CTRL, CTRL_AI);
    memset(&WD, 0, sizeof WD);
    LEARN_S = learn_s;
    WIN_N = RX_AT_LAST = 0;
    RESTART_REQ = 0;
    memset((void *)&ST, 0, sizeof ST);
    ST.learning = 1;
    ST.sim = -1;
    ST.duty = 1.0f;
}

float twin_learn_s(void) { return LEARN_S; }

/* ---------------------------------------------------------------------------
 * Single-precision version of ms_plant_step() for the ESP32.
 * The ESP32 has a float FPU but does double maths in software (~20 us per
 * sin/exp), and the PC model makes ~70 such calls per 1 ms sample - more than
 * the whole 1 ms budget. Same physics, but phases are kept wrapped to [0, 2*pi)
 * (separate accumulators for the bearing cage and the 1/2x looseness
 * sub-harmonic) so float precision does not decay over long runs.
 * ------------------------------------------------------------------------- */
#define TWO_PI_F 6.2831853f

static inline uint32_t xs32(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return (uint32_t)(x >> 32);
}
static inline float u01f(uint64_t *s) { return ((xs32(s) >> 8) + 0.5f) * (1.0f / 16777216.0f); }
static inline float gaussf(uint64_t *s)
{
    /* sum of 4 uniforms (mean 2, variance 1/3): close to Gaussian, no log/sqrt/cos */
    return (u01f(s) + u01f(s) + u01f(s) + u01f(s) - 2.0f) * 1.7320508f;
}

static void plant_step_f(ms_plant_t *p, ms_tick_t *o)
{
    const ms_cfg_t *c = &p->c;
    const float dt = 1.0f / MS_FS;
    float t = (float)p->n * dt;
    float wn = c->rpm_nom / 60.0f * TWO_PI_F;
    int f = c->fault;
    float s = (float)p->sev;
    float s_ov = f == MSF_OVERLOAD ? s : 0.0f;

    float load = c->load_amp * sinf(TWO_PI_F * fmodf(t / c->load_period_s, 1.0f) + (c->seed % 97) * 0.07f);
    float omega = (float)p->omega;
    float w_target = wn * (float)p->duty * (1.0f - 0.04f * load) * (1.0f - 0.35f * s_ov);
    omega += (w_target - omega) * dt / 0.25f;
    if (omega < 0) omega = 0;
    p->omega = omega;
    float r = omega / wn, r2 = r * r;

    float th0 = (float)p->theta, th = th0 + omega * dt;
    o->hall = th >= TWO_PI_F;                  /* one magnet: edge at each full turn */
    o->hall_us = o->hall ? (TWO_PI_F - th0) / (th - th0 + 1e-12f) * 1000.0f : 0.0f;
    if (th >= TWO_PI_F) th -= TWO_PI_F;
    p->theta = th;
    PH_B += 3.585f * omega * dt / TWO_PI_F; if (PH_B >= 1.0f) PH_B -= 1.0f;
    PH_H += 0.5f * omega * dt;              if (PH_H >= TWO_PI_F) PH_H -= TWO_PI_F;

    float A1 = c->a1 * r2 * (1.0f + 0.3f * load);
    float s1 = sinf(th), c1 = cosf(th);
    float ax = A1 * s1 + 0.15f * A1 * sinf(2 * th + 0.4f) + 0.012f * r * sinf(7 * th + 0.2f);
    float ay = A1 * c1 + 0.15f * A1 * cosf(2 * th + 0.4f) + 0.012f * r * cosf(7 * th + 0.2f);
    float az = 0.25f * A1 * sinf(th + 1.1f);
    float fb = 3.585f * omega / TWO_PI_F + 1e-6f;
    switch (f) {
    case MSF_IMBALANCE: ax += 3.0f * s * A1 * s1; ay += 3.0f * s * A1 * c1; break;
    case MSF_MISALIGN:
        ax += 0.12f * s * r2 * sinf(2 * th + 0.9f); ay += 0.12f * s * r2 * cosf(2 * th + 0.9f);
        az += (0.06f * sinf(th + 0.3f) + 0.08f * sinf(2 * th + 1.7f)) * s * r2; break;
    case MSF_BEARING: {
        float wr = TWO_PI_F * 330.0f, ts = PH_B / fb;
        float imp = s * r * expf(-0.06f * wr * ts), sw = sinf(wr * ts);
        ax += 0.5f * imp * sw; ay += 0.3f * imp * sw; az += 0.15f * imp * sw;
        break; }
    case MSF_LOOSENESS: {
        float rect = fabsf(s1) - 0.6366f;
        ax += s * r2 * (0.25f * rect + 0.07f * sinf(PH_H) + 0.06f * sinf(3 * th));
        ay += s * r2 * (0.15f * rect + 0.05f * cosf(PH_H)); az += s * r2 * 0.10f * rect; break; }
    case MSF_OVERLOAD: ax += 0.5f * s * A1 * s1; ay += 0.5f * s * A1 * c1; break;
    default: break;
    }
    float gx = 8.0f * ay, gy = -8.0f * ax, gz = 2.0f * az;
    ax += 0.009f * gaussf(&p->rng); ay += 0.009f * gaussf(&p->rng); az += 0.009f * gaussf(&p->rng) + 1.0f;
    gx += 0.11f * gaussf(&p->rng); gy += 0.11f * gaussf(&p->rng); gz += 0.11f * gaussf(&p->rng);

    /* motor temperature (the Wokwi MPU6050 temperature normally replaces it) */
    float t_amb = c->ambient_c;
    float heat = 12.0f * (float)(p->duty * p->duty) * (1.0f + 0.6f * s_ov) * (1.0f + 0.6f * s_ov) * (1.0f + 0.15f * load)
               + (f == MSF_BEARING ? 5.0f * s * r : 0.0f) + (f == MSF_MISALIGN ? 3.0f * s * r : 0.0f);
    p->t_motor += (t_amb + heat - (float)p->t_motor) * dt / 180.0f;
    float t_die = (float)p->t_motor + 0.05f * gaussf(&p->rng);

    float rawv[7] = {ax * 16384, ay * 16384, az * 16384, (t_die - 36.53f) * 340, gx * 131, gy * 131, gz * 131};
    for (int i = 0; i < 7; i++) {
        float v = floorf(rawv[i] + 0.5f);
        o->mpu[i] = (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
    }

    o->current_a = 0.25f + 0.55f * (float)p->duty * (1.0f + 0.3f * load) * (1.0f + 1.2f * s_ov)
                 + (f == MSF_BEARING ? 0.08f * s : 0.0f) + (f == MSF_MISALIGN ? 0.06f * s : 0.0f)
                 + 0.05f * (float)p->duty * sinf(7 * th) + 0.03f * gaussf(&p->rng);

    /* sound: 8 samples per tick at 8 kHz */
    float pb0 = PH_B - 3.585f * omega * dt / TWO_PI_F;
    if (pb0 < 0) pb0 += 1.0f;
    for (int k = 0; k < MS_AUD; k++) {
        float tk = k / (float)(MS_FS * MS_AUD), thk = th0 + omega * tk;
        float a = 0.05f * r * sinf(thk) + 0.08f * r * sinf(7 * thk) * (1.0f + 0.5f * s_ov) + 0.03f * r * sinf(14 * thk)
                + (0.006f + 0.014f * r) * gaussf(&p->rng);
        if (f == MSF_BEARING) {
            float pb = pb0 + 3.585f * omega * tk / TWO_PI_F;
            if (pb >= 1.0f) pb -= 1.0f;
            float ts = pb / fb;
            a += 0.25f * s * r * expf(-900.0f * ts) * sinf(TWO_PI_F * 2200.0f * ts);
        }
        if (f == MSF_LOOSENESS) a += 0.10f * s * r * (fabsf(sinf(thk)) - 0.6366f) * (1.0f + gaussf(&p->rng) * 0.5f);
        if (f == MSF_MISALIGN) a += 0.05f * s * r * sinf(2 * thk);
        if (f == MSF_IMBALANCE) a += 0.04f * s * r * sinf(thk);
        float v = floorf(a * 8000.0f + 0.5f);
        o->audio[k] = (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
    }

    o->dht_new = (p->n % MS_DHT_TICKS) == 0;
    o->dht_t = c->ambient_c;
    o->dht_rh = c->rh_pct;
    o->seq = (uint32_t)p->n;
    p->n++;
}

void twin_plant_skip(uint32_t k)
{
    /* samples the sensor task could not take in time (FIFO overflow on a
       real MPU6050): the motor keeps turning, the samples are lost and
       Core 1 counts them as gaps */
    float th = (float)P.theta + (float)P.omega * k / MS_FS;
    P.theta = fmodf(th, TWO_PI_F);
    P.n += k;
}

void twin_plant_tick(const twin_inputs_t *in, float duty, ms_tick_t *out)
{
    int learning = P.n < (uint64_t)(LEARN_S * MS_FS);
    int f = in->fault_type;
    float s = in->fault_sev;
    if (learning || s < 0.02f || f <= MSF_NONE || f > MSF_OVERLOAD) { f = MSF_NONE; s = 0.0f; }
    P.c.fault = f;
    P.sev = s;                     /* severity is set directly by the slider */
    ms_plant_set_duty(&P, duty);
    plant_step_f(&P, out);
    if (in->have_mpu_temp) out->mpu[3] = in->mpu_temp_raw;
    if (in->have_dht) { out->dht_t = in->dht_t; out->dht_rh = in->dht_rh; }
}

void twin_request_restart(void) { RESTART_REQ = 1; }

int twin_process(const ms_tick_t *tk, float *duty, twin_packet_t *pkt)
{
    if (RESTART_REQ) {
        RESTART_REQ = 0;
        ms_limits_t keep = CTRL.L;           /* keep the learned healthy references */
        ms_ctrl_init(&CTRL, CTRL_AI);
        CTRL.L = keep;
        *duty = 1.0f;
    }
    if (!ms_windower_push(&WD, tk, *duty, &W)) return 0;

    float f[MS_NF], raw[6];
    float t = (tk->seq + 1) / (float)MS_FS;
    ms_features(&W, RPM_NOM, f);
    ms_raw_values(&W, f, raw);
    int sim = -1;
    if (t <= LEARN_S) {
        det_learn(&DET, f);
        ms_ctrl_learn(&CTRL, raw);
    } else {
        if (!DET.ready) det_finalize(&DET);
        sim = det_similarity(&DET, f);
        *duty = ms_ctrl_step(&CTRL, t, raw, sim);
    }

    pkt->win = WIN_N++;
    pkt->seq_last = tk->seq;
    pkt->n_new = (uint32_t)(WD.received - RX_AT_LAST);
    RX_AT_LAST = (uint32_t)WD.received;
    pkt->sim = sim;
    pkt->state = CTRL.state;
    pkt->duty = *duty;
    pkt->vib = raw[0];
    pkt->t_motor = raw[2];
    pkt->t_amb = raw[3];
    pkt->rh = raw[4];
    pkt->rpm = raw[1] * RPM_NOM * W.duty;
    pkt->current = raw[5];
    pkt->reason = CTRL.reason;
    pkt->gaps = WD.gaps;

    ST.learning = sim < 0; ST.sim = sim; ST.state = CTRL.state; ST.t = t; ST.duty = *duty;
    ST.vib = raw[0]; ST.t_motor = raw[2]; ST.t_amb = raw[3]; ST.rh = raw[4];
    ST.rpm = pkt->rpm; ST.current = raw[5]; ST.reason = CTRL.reason;
    return 1;
}

void twin_status(twin_status_t *s) { memcpy(s, (const void *)&ST, sizeof *s); }
