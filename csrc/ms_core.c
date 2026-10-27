#include <math.h>
#include <string.h>
#include "ms_core.h"
#include "dsp_features.h"            /* fft_radix2 */

#define TWO_PI 6.283185307179586

/* ------------------------------------------------------------ RNG -------- */
static inline uint64_t xs(uint64_t *s)
{
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}
static inline double u01(uint64_t *s) { return ((xs(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static inline double gauss(uint64_t *s)
{
    double a = u01(s), b = u01(s);
    return sqrt(-2.0 * log(a)) * cos(TWO_PI * b);
}

void ms_cfg_default(ms_cfg_t *c, uint64_t seed)
{
    memset(c, 0, sizeof *c);
    c->seed = seed;
    c->rpm_nom = 3000.0f;
    c->a1 = 0.05f;
    c->ambient_c = 28.0f;
    c->rh_pct = 55.0f;
    c->drift_s = 1.0f;
    c->load_amp = 0.2f;
    c->load_period_s = 240.0f;
    c->fault = MSF_NONE;
    c->onset_s = 1e9f;
    c->ramp_s = 900.0f;
}

void ms_plant_init(ms_plant_t *p, const ms_cfg_t *c)
{
    memset(p, 0, sizeof *p);
    p->c = *c;
    p->rng = c->seed * 0x9E3779B97F4A7C15ull + 0x1234567ull;
    if (!p->rng) p->rng = 1;
    p->duty = 1.0;
    p->omega = c->rpm_nom / 60.0 * TWO_PI;
    p->t_motor = c->ambient_c + 12.0;
    p->dht_t = c->ambient_c;
    p->dht_rh = c->rh_pct;
}

void ms_plant_set_duty(ms_plant_t *p, float duty)
{
    p->duty = duty < 0 ? 0 : (duty > 1 ? 1 : duty);
}

float ms_plant_rpm(const ms_plant_t *p) { return (float)(p->omega * 60.0 / TWO_PI); }

static int mechanical(int f)
{
    return f == MSF_IMBALANCE || f == MSF_BEARING || f == MSF_MISALIGN || f == MSF_LOOSENESS || f == MSF_OVERLOAD;
}

void ms_plant_step(ms_plant_t *p, ms_tick_t *o)
{
    const ms_cfg_t *c = &p->c;
    const double dt = 1.0 / MS_FS;
    double t = p->n * dt;
    double wn = c->rpm_nom / 60.0 * TWO_PI;
    int f = c->fault;

    /* ---- fault severity: mechanical wear grows with speed^2 --------------- */
    if (f != MSF_NONE && t >= c->onset_s && p->sev < 1.0) {
        double r = p->omega / wn;
        p->sev += dt / c->ramp_s * (mechanical(f) ? r * r : 1.0);
        if (p->sev > 1.0) p->sev = 1.0;
    }
    double s = p->sev;
    double s_ov = f == MSF_OVERLOAD ? s : 0.0;

    /* ---- DC motor speed follows the L298N duty (tau 0.25 s) -------------- */
    double load = c->load_amp * sin(TWO_PI * t / c->load_period_s + (c->seed % 97) * 0.07);
    double w_target = wn * p->duty * (1.0 - 0.04 * load) * (1.0 - 0.35 * s_ov);
    p->omega += (w_target - p->omega) * dt / 0.25;
    if (p->omega < 0) p->omega = 0;
    double r = p->omega / wn;
    double th0 = p->theta;
    p->theta += p->omega * dt;

    /* ---- Hall sensor: one magnet, edge when theta crosses 2*pi*k ---------- */
    double k0 = floor(th0 / TWO_PI), k1 = floor(p->theta / TWO_PI);
    o->hall = k1 > k0;
    o->hall_us = o->hall ? (float)(((k1 * TWO_PI) - th0) / (p->theta - th0 + 1e-12) * 1000.0) : 0.0f;

    /* ---- vibration (g), amplitudes scale with speed^2 ----------------------- */
    double th = p->theta, r2 = r * r;
    double A1 = c->a1 * r2 * (1.0 + 0.3 * load);
    double ax = A1 * sin(th) + 0.15 * A1 * sin(2 * th + 0.4) + 0.012 * r * sin(7 * th + 0.2);
    double ay = A1 * cos(th) + 0.15 * A1 * cos(2 * th + 0.4) + 0.012 * r * cos(7 * th + 0.2);
    double az = 0.25 * A1 * sin(th + 1.1);
    double bear_imp = 0.0;
    switch (f) {
    case MSF_IMBALANCE: ax += 3.0 * s * A1 * sin(th); ay += 3.0 * s * A1 * cos(th); break;
    case MSF_MISALIGN:
        ax += 0.12 * s * r2 * sin(2 * th + 0.9); ay += 0.12 * s * r2 * cos(2 * th + 0.9);
        az += (0.06 * sin(th + 0.3) + 0.08 * sin(2 * th + 1.7)) * s * r2; break;
    case MSF_BEARING: {
        double pb = 3.585 * th / TWO_PI, frac = pb - floor(pb), fb = 3.585 * p->omega / TWO_PI + 1e-6;
        double wr = TWO_PI * 330.0, ts = frac / fb;
        bear_imp = s * r * exp(-0.06 * wr * ts);
        ax += 0.5 * bear_imp * sin(wr * ts); ay += 0.3 * bear_imp * sin(wr * ts); az += 0.15 * bear_imp * sin(wr * ts);
        break; }
    case MSF_LOOSENESS: {
        double rect = fabs(sin(th)) - 0.6366;
        ax += s * r2 * (0.25 * rect + 0.07 * sin(0.5 * th) + 0.06 * sin(3 * th));
        ay += s * r2 * (0.15 * rect + 0.05 * cos(0.5 * th)); az += s * r2 * 0.10 * rect; break; }
    case MSF_OVERLOAD: ax += 0.5 * s * A1 * sin(th); ay += 0.5 * s * A1 * cos(th); break;
    default: break;
    }
    double gx = 8.0 * ay, gy = -8.0 * ax, gz = 2.0 * az;
    ax += 0.009 * gauss(&p->rng); ay += 0.009 * gauss(&p->rng); az += 0.009 * gauss(&p->rng) + 1.0;
    gx += 0.11 * gauss(&p->rng); gy += 0.11 * gauss(&p->rng); gz += 0.11 * gauss(&p->rng);

    /* ---- temperatures --------------------------------------------------- */
    double frac_d = c->drift_s > 0 ? fmin(t / c->drift_s, 1.0) : 1.0;
    double t_amb = c->ambient_c + c->drift_c * frac_d;
    double rh = c->rh_pct + c->drift_rh * frac_d + (f == MSF_HUMIDITY ? 35.0 * s : 0.0);
    if (rh > 99.0) rh = 99.0;
    double heat = 12.0 * p->duty * p->duty * (1.0 + 0.6 * s_ov) * (1.0 + 0.6 * s_ov) * (1.0 + 0.15 * load)
                + (f == MSF_COOLING ? 18.0 * s : 0.0) + (f == MSF_BEARING ? 5.0 * s * r : 0.0)
                + (f == MSF_MISALIGN ? 3.0 * s * r : 0.0);
    p->t_motor += (t_amb + heat - p->t_motor) * dt / 180.0;
    double t_die = p->t_motor + 0.05 * gauss(&p->rng);

    double raw[7] = {ax * 16384, ay * 16384, az * 16384, (t_die - 36.53) * 340, gx * 131, gy * 131, gz * 131};
    for (int i = 0; i < 7; i++) {
        double v = floor(raw[i] + 0.5);
        o->mpu[i] = (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
    }

    /* ---- current (ACS712): load, overload, friction, commutation ripple ---- */
    o->current_a = (float)(0.25 + 0.55 * p->duty * (1.0 + 0.3 * load) * (1.0 + 1.2 * s_ov)
                           + (f == MSF_BEARING ? 0.08 * s : 0.0) + (f == MSF_MISALIGN ? 0.06 * s : 0.0)
                           + 0.05 * p->duty * sin(7 * th) + 0.03 * gauss(&p->rng));

    /* ---- sound (INMP441, 8 kHz) ------------------------------------------- */
    for (int k = 0; k < MS_AUD; k++) {
        double tk = k / (double)(MS_FS * MS_AUD), thk = th0 + p->omega * tk;
        double a = 0.05 * r * sin(thk) + 0.08 * r * sin(7 * thk) * (1.0 + 0.5 * s_ov) + 0.03 * r * sin(14 * thk)
                 + (0.006 + 0.014 * r) * gauss(&p->rng);
        if (f == MSF_BEARING) {
            double pb = 3.585 * thk / TWO_PI, fr = pb - floor(pb), fb = 3.585 * p->omega / TWO_PI + 1e-6;
            double ts = fr / fb;
            a += 0.25 * s * r * exp(-900.0 * ts) * sin(TWO_PI * 2200.0 * ts);
        }
        if (f == MSF_LOOSENESS) a += 0.10 * s * r * (fabs(sin(thk)) - 0.6366) * (1.0 + gauss(&p->rng) * 0.5);
        if (f == MSF_MISALIGN) a += 0.05 * s * r * sin(2 * thk);
        if (f == MSF_IMBALANCE) a += 0.04 * s * r * sin(thk);
        double v = floor(a * 8000.0 + 0.5);
        o->audio[k] = (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
    }

    /* ---- DHT22 every 2 s ------------------------------------------------- */
    o->dht_new = (p->n % MS_DHT_TICKS) == 0;
    if (o->dht_new) {
        p->dht_t = (float)(floor((t_amb + 0.15 * gauss(&p->rng)) * 10 + 0.5) / 10);
        p->dht_rh = (float)(floor((rh + 0.8 * gauss(&p->rng)) * 10 + 0.5) / 10);
    }
    o->dht_t = p->dht_t;
    o->dht_rh = p->dht_rh;
    o->seq = (uint32_t)p->n;
    p->n++;
}

/* ============================================================ features === */
const char *MS_FEATURE_NAMES[MS_NF] = {
    "vib_ax_rms", "vib_ay_rms", "vib_az_rms", "vib_ax_kurt", "vib_ax_crest", "vib_gx_rms",
    "vib_band_lo", "vib_band_mid", "vib_band_hi", "temp_motor", "temp_ambient", "humidity",
    "rpm_ratio", "current_mean", "current_ripple", "sound_db", "sound_centroid_khz", "sound_hf_ratio"};
const int MS_FEATURE_GROUP[MS_NF] = {
    MSG_VIB, MSG_VIB, MSG_VIB, MSG_VIB, MSG_VIB, MSG_VIB, MSG_VIB, MSG_VIB, MSG_VIB,
    MSG_TEMP, MSG_TEMP, MSG_HUM, MSG_RPM, MSG_CUR, MSG_CUR, MSG_SND, MSG_SND, MSG_SND};

static float ch_rms(const int16_t (*m)[7], int c, float scale, float *kurt, float *crest)
{
    float mean = 0, m2 = 0, m4 = 0, amax = 0;
    for (int i = 0; i < MS_WIN; i++) mean += m[i][c] / scale;
    mean /= MS_WIN;
    for (int i = 0; i < MS_WIN; i++) {
        float v = m[i][c] / scale - mean, v2 = v * v;
        m2 += v2; m4 += v2 * v2;
        if (fabsf(v) > amax) amax = fabsf(v);
    }
    m2 /= MS_WIN; m4 /= MS_WIN;
    float rms = sqrtf(m2) + 1e-9f;
    if (kurt) *kurt = m4 / (m2 * m2 + 1e-18f);
    if (crest) *crest = amax / rms;
    return rms;
}

void ms_features(const ms_window_t *w, float rpm_nom, float out[MS_NF])
{
    /* measured speed from Hall edge times (period average) */
    float rpm = 0.0f;
    if (w->n_hall >= 2)
        rpm = (float)(60.0 * (w->n_hall - 1) / (w->hall_t[w->n_hall - 1] - w->hall_t[0]));
    float r = rpm / rpm_nom;
    float rn = r > 0.2f ? r : 0.2f, s2 = rn * rn;
    float kurt, crest;
    out[0] = ch_rms(w->mpu, 0, 16384.0f, &kurt, &crest) / s2;
    out[3] = kurt;
    out[4] = crest;
    out[1] = ch_rms(w->mpu, 1, 16384.0f, NULL, NULL) / s2;
    out[2] = ch_rms(w->mpu, 2, 16384.0f, NULL, NULL) / s2;
    out[5] = ch_rms(w->mpu, 4, 131.0f, NULL, NULL) / s2;

    float re[MS_WIN], im[MS_WIN], mean = 0;
    for (int i = 0; i < MS_WIN; i++) mean += w->mpu[i][0] / 16384.0f;
    mean /= MS_WIN;
    for (int i = 0; i < MS_WIN; i++) {
        float h = 0.5f - 0.5f * cosf(6.2831853f * i / (MS_WIN - 1));
        re[i] = (w->mpu[i][0] / 16384.0f - mean) * h; im[i] = 0;
    }
    fft_radix2(re, im, MS_WIN);
    float b[3] = {0, 0, 0};
    for (int k = 1; k <= MS_WIN / 2; k++) {
        float fr = (float)k / MS_WIN, pw = re[k] * re[k] + im[k] * im[k];
        b[fr < 0.08f ? 0 : (fr < 0.25f ? 1 : 2)] += pw / (MS_WIN * MS_WIN);
    }
    for (int i = 0; i < 3; i++) out[6 + i] = log10f(b[i] + 1e-12f) - 4.0f * log10f(rn);

    float tm = 0;
    for (int i = 0; i < MS_WIN; i++) tm += w->mpu[i][3] / 340.0f + 36.53f;
    out[9] = tm / MS_WIN;
    out[10] = w->dht_t;
    out[11] = w->dht_rh;
    float expect = w->duty > 0.05f ? w->duty : 0.05f;
    out[12] = r / expect;

    float cm = 0, cv = 0;
    for (int i = 0; i < MS_WIN; i++) cm += w->cur[i];
    cm /= MS_WIN;
    for (int i = 0; i < MS_WIN; i++) cv += (w->cur[i] - cm) * (w->cur[i] - cm);
    out[13] = (cm - 0.25f) / expect;
    out[14] = sqrtf(cv / MS_WIN);

    /* sound: level, spectral centroid, high-frequency ratio (2048-pt FFT) */
    enum { NA = MS_WIN * MS_AUD };
    static float are[NA], aim[NA];
    float am = 0, e = 0;
    for (int i = 0; i < NA; i++) am += w->aud[i];
    am /= NA;
    for (int i = 0; i < NA; i++) {
        float v = (w->aud[i] - am) / 8000.0f;
        e += v * v;
        are[i] = v * (0.5f - 0.5f * cosf(6.2831853f * i / (NA - 1)));
        aim[i] = 0;
    }
    fft_radix2(are, aim, NA);
    double sp = 0, spf = 0, hf = 0;
    for (int k = 1; k < NA / 2; k++) {
        double pw = are[k] * are[k] + aim[k] * aim[k], fr = k * 8000.0 / NA;
        sp += pw; spf += pw * fr;
        if (fr > 1500.0) hf += pw;
    }
    out[15] = 10.0f * log10f(e / NA + 1e-12f) - 20.0f * log10f(rn);
    out[16] = (float)(spf / (sp + 1e-20) / 1000.0);
    out[17] = (float)log10(hf / (sp + 1e-20) + 1e-9);
}

int ms_windower_push(ms_windower_t *wd, const ms_tick_t *tk, float duty, ms_window_t *out)
{
    if (wd->started && tk->seq != wd->next_seq)
        wd->gaps += (uint32_t)(tk->seq - wd->next_seq);       /* missing samples */
    wd->started = 1;
    wd->next_seq = tk->seq + 1;
    wd->received++;
    wd->ring[wd->head] = *tk;
    wd->head = (wd->head + 1) % MS_WIN;
    if (wd->fill < MS_WIN) wd->fill++;
    wd->since++;
    if (wd->fill < MS_WIN || wd->since < MS_HOP) return 0;
    wd->since = 0;
    out->n_hall = 0;
    for (int i = 0; i < MS_WIN; i++) {
        const ms_tick_t *k = &wd->ring[(wd->head + i) % MS_WIN];
        memcpy(out->mpu[i], k->mpu, sizeof k->mpu);
        out->cur[i] = k->current_a;
        memcpy(&out->aud[i * MS_AUD], k->audio, sizeof k->audio);
        if (k->hall && out->n_hall < 64)
            out->hall_t[out->n_hall++] = (k->seq + k->hall_us / 1000.0) / MS_FS;
    }
    const ms_tick_t *last = &wd->ring[(wd->head + MS_WIN - 1) % MS_WIN];
    out->dht_t = last->dht_t;
    out->dht_rh = last->dht_rh;
    out->duty = duty;
    return 1;
}

void ms_raw_values(const ms_window_t *w, const float feat[MS_NF], float raw[6])
{
    raw[0] = ch_rms(w->mpu, 0, 16384.0f, NULL, NULL);
    raw[1] = feat[12];
    raw[2] = feat[9];
    raw[3] = feat[10];
    raw[4] = feat[11];
    float cm = 0;
    for (int i = 0; i < MS_WIN; i++) cm += w->cur[i];
    raw[5] = cm / MS_WIN;
}

/* ========================================================= controller === */
void ms_limits_default(ms_limits_t *L)
{
    L->vib_warn_g = 0.075f;  L->vib_stop_g = 0.15f;      /* 2x and 4x healthy RMS     */
    L->rpm_dev_warn = 0.10f;
    L->tmotor_warn_c = 52.0f; L->tmotor_stop_c = 60.0f;
    L->tamb_warn_c = 40.0f;   L->tamb_stop_c = 45.0f;
    L->rh_warn = 80.0f;       L->rh_stop = 90.0f;
    L->cur_warn_a = 1.0f;     L->cur_stop_a = 1.3f;
    L->step_s = 10.0f;
    L->debounce = 3;
}

void ms_detector_setup(detector_t *d)
{
    d->floor[9] = 0.5f;          /* motor temperature, degC  */
    d->floor[10] = 0.5f;         /* DHT22 ambient, degC (sensor accuracy +-0.5) */
    d->floor[11] = 2.0f;         /* DHT22 humidity, %RH (sensor accuracy +-2)   */
    d->adapt_mask = (1u << 9) | (1u << 10) | (1u << 11);
}

static float g_learn_n = 0;
void ms_ctrl_learn(ms_ctrl_t *c, const float raw[6])
{
    float n = ++c->ok_streak;                  /* reuse as a counter while learning */
    c->L.vib_ref += (raw[0] - c->L.vib_ref) / n;
    c->L.cur_ref += (raw[5] - c->L.cur_ref) / n;
    (void)g_learn_n;
}

void ms_ctrl_init(ms_ctrl_t *c, int mode)
{
    memset(c, 0, sizeof *c);
    c->mode = mode;
    c->duty = 1.0f;
    c->t_warn = c->t_stop = -1.0f;
    ms_limits_default(&c->L);
}

float ms_ctrl_step(ms_ctrl_t *c, float t, const float raw[6], int sim)
{
    const ms_limits_t *L = &c->L;
    if (c->mode == CTRL_NONE || c->state == ST_STOPPED) return c->duty;
    if (!c->started) { c->started = 1; c->ok_streak = 0; }
    float vib = raw[0], rr = raw[1], tm = raw[2], ta = raw[3], rh = raw[4], cur = raw[5];

    /* hard limits = the professor's sample-project thresholds (always active) */
    uint32_t warn = 0, stop = 0;
    if (vib > L->vib_warn_g) warn |= MSG_VIB;
    if (vib > L->vib_stop_g) stop |= MSG_VIB;
    if (fabsf(rr - 1.0f) > L->rpm_dev_warn && c->duty > 0.05f) warn |= MSG_RPM;
    if (tm > L->tmotor_warn_c || ta > L->tamb_warn_c) warn |= MSG_TEMP;
    if (tm > L->tmotor_stop_c || ta > L->tamb_stop_c) stop |= MSG_TEMP;
    if (rh > L->rh_warn) warn |= MSG_HUM;
    if (rh > L->rh_stop) stop |= MSG_HUM;
    if (cur > L->cur_warn_a) warn |= MSG_CUR;
    if (cur > L->cur_stop_a) stop |= MSG_CUR;

    /* AI: debounced one-class anomaly, graded by how far from normal it is
       sim < 90 -> warning (schedule maintenance, keep running)
       sim < 50 -> reduce speed one step per step_s
       sim < 10 while already at 25 % -> stop                                  */
    int ai = 0, ai_severe = 0;
    if (c->mode == CTRL_AI && sim >= 0 && sim <= 100) {
        c->streak = sim < 90 ? c->streak + 1 : 0;
        ai = c->streak >= L->debounce;
        if (ai) warn |= 64;
        /* self-optimising speed: keep the fastest speed at which the motor's raw
           vibration and current stay near healthy (1.5x / 1.25x of learned) */
        int hot = (L->vib_ref > 0 && vib > 1.5f * L->vib_ref) || (L->cur_ref > 0 && cur > 1.25f * L->cur_ref);
        ai_severe = ai && hot;
        if (ai_severe && c->duty < 0.30f && t - c->t_last_step >= L->step_s) stop |= 64;
    }

    if (stop) {                                          /* safety interlock */
        c->state = ST_STOPPED; c->duty = 0.0f; c->t_stop = t; c->reason |= stop;
        if (c->t_warn < 0) c->t_warn = t;
        return c->duty;
    }
    /* what may reduce speed: vibration / current / rpm (mechanical) or the AI */
    int derate_cause = (warn & (MSG_VIB | MSG_CUR | MSG_RPM)) || ai_severe;
    if (warn) {
        c->ok_streak = 0;
        if (c->state == ST_NORMAL) { c->state = ST_WARNING; c->t_last_step = t; }
        if (c->t_warn < 0) c->t_warn = t;
        c->reason |= warn;
        if (derate_cause && t - c->t_last_step >= L->step_s) {
            if (c->duty > 0.30f) { c->duty -= 0.25f; c->state = ST_DERATED; c->t_last_step = t; }
            else if (c->mode == CTRL_THRESHOLD) { c->state = ST_STOPPED; c->duty = 0.0f; c->t_stop = t; }
        }
    } else if (c->state == ST_WARNING) {
        if (++c->ok_streak > 20) c->state = ST_NORMAL;   /* warning clears; derating is latched */
    }
    return c->duty;
}
