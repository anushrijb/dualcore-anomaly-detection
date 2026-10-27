/* ---------------------------------------------------------------------------
 * edge_dualcore.c - ESP32 dual-core edge node, emulated with two pinned threads
 *
 *  Core 0 task "acq" : every 1/fs, burst-read the MPU6050 (14 bytes over I2C,
 *                      ~385 us at 400 kHz) and push the sample into a lock-free
 *                      ring buffer (FreeRTOS queue equivalent).
 *  Core 1 task "dsp" : pop samples, build 256-sample windows with 50 % overlap,
 *                      extract 31 features, learn on-device for the first
 *                      --learn-s seconds, then infer (similarity, Normal/Anomaly).
 *
 *  --mode single runs BOTH stages in one thread on one core (the baseline a
 *  single-core MCU would have): sensor reads wait while a window is processed.
 *
 *  Timing model
 *   --pace rt   : real time. The sensor produces sample k at t0 + k/fs whether
 *                 or not it is read; the reader's lateness is recorded. A sample
 *                 later than the MPU6050 FIFO depth (85 samples) is counted lost.
 *   --pace fast : no waiting, no I2C delay; used to stream long scenarios quickly.
 *   --i2c-us    : bus time of one burst read (busy, the core cannot do other work)
 *   --proc-us   : emulated processing time per window (busy-wait up to this, to
 *                 model a slower MCU or a heavier model); 0 = host speed only.
 *
 *  Output: one CSV line per window on stdout (or --out), and a JSON summary
 *  (--stats file) with jitter, loss, latency and per-core utilisation.
 * ------------------------------------------------------------------------- */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "mpu6050_sim.h"
#include "dsp_features.h"
#include "detector.h"
#include "ringbuf.h"

/* ------------------------------------------------------------- options --- */
static motor_cfg_t M;
static double opt_seconds = 60, opt_i2c_us = -1, opt_proc_us = 0, opt_learn_s = 480;
static double opt_center = 1.6, opt_sens = 3.0;
static int opt_dual = 1, opt_rt = 1, opt_kind = DET_MAHA, opt_quiet = 0;
static const char *opt_out = NULL, *opt_stats = NULL;

/* ---------------------------------------------------------- shared state -- */
static ringbuf_t RB;
static uint64_t n_samples, t0_ns;
static volatile int producer_done = 0;

typedef struct {                 /* per window decision */
    uint32_t w;
    float t_end_s, sim, ax_rms, temp, proc_us, e2e_us;
    uint8_t learning, anomaly;
} decision_t;
static decision_t *DEC;
static uint32_t n_dec = 0;

/* statistics */
static double acq_busy_ns = 0, dsp_busy_ns = 0, late_max_ns = 0, late_sum_ns = 0;
static uint64_t late_n = 0, lost_fifo = 0, lost_ring = 0, late_over_period = 0;
static double *late_hist;        /* per-sample lateness (us) for percentiles */
static uint32_t ring_high = 0;

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static inline void spin_until(uint64_t t) { while (now_ns() < t) { } }

static void pin_to(int cpu)
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu % (int)sysconf(_SC_NPROCESSORS_ONLN), &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

/* --------------------------------------------- Core 0: sensor acquisition -- */
static void acquire_one(uint64_t k, rb_item_t *it)
{
    uint64_t target = t0_ns + (uint64_t)(k * 1e9 / M.fs);
    if (opt_rt) spin_until(target);                 /* sample period (hardware timer) */
    uint64_t start = now_ns();
    mpu6050_read(&M, k, it->raw);                    /* I2C burst read 0x3B..0x48 */
    if (opt_rt && opt_i2c_us > 0) spin_until(start + (uint64_t)(opt_i2c_us * 1000));
    uint64_t end = now_ns();
    it->t_target_ns = opt_rt ? target : start;
    acq_busy_ns += end - start;
    if (opt_rt) {
        double late = (double)(start > target ? start - target : 0);
        late_hist[k] = late / 1000.0;
        late_sum_ns += late; late_n++;
        if (late > late_max_ns) late_max_ns = late;
        if (late > 1e9 / M.fs) late_over_period++;
        if (late > MPU_FIFO_SAMPLES * 1e9 / M.fs) lost_fifo++;
    }
}

static void *task_acq(void *arg)
{
    (void)arg;
    pin_to(0);
    rb_item_t it;
    for (uint64_t k = 0; k < n_samples; k++) {
        acquire_one(k, &it);
        if (opt_rt) { if (!rb_push(&RB, &it)) lost_ring++; }
        else while (!rb_push(&RB, &it)) { }          /* fast mode: back-pressure */
        uint32_t c = rb_count(&RB);
        if (c > ring_high) ring_high = c;
    }
    producer_done = 1;
    return NULL;
}

/* ---------------------------------------- Core 1: features + detection ----- */
static detector_t DET;
static float WINBUF[WIN * MPU_REGS];
static uint64_t win_fill = 0, since_hop = 0, wcount = 0;
static uint64_t last_target = 0;

static void process_window(void)
{
    uint64_t t_start = now_ns();
    float x[WIN * MPU_REGS], f[N_FEAT];
    memcpy(x, WINBUF, sizeof x);
    features_window(x, M.fs, f);
    decision_t *d = &DEC[n_dec];
    d->w = (uint32_t)wcount;
    d->t_end_s = (float)((wcount * HOP + WIN) / M.fs);
    d->ax_rms = f[0];
    d->temp = f[30];
    if (d->t_end_s <= opt_learn_s) {                     /* on-device learning */
        det_learn(&DET, f);
        d->learning = 1; d->sim = -1; d->anomaly = 0;
    } else {
        if (!DET.ready) det_finalize(&DET);
        uint8_t s = det_similarity(&DET, f);
        d->learning = 0; d->sim = s; d->anomaly = s < 90;
    }
    uint64_t t_done = now_ns();
    if (opt_proc_us > 0) {                               /* emulate slower MCU / heavier model */
        uint64_t want = t_start + (uint64_t)(opt_proc_us * 1000);
        if (t_done < want) { spin_until(want); t_done = want; }
    }
    d->proc_us = (t_done - t_start) / 1000.0f;
    d->e2e_us = opt_rt ? (t_done - last_target) / 1000.0f : d->proc_us;
    dsp_busy_ns += t_done - t_start;
    n_dec++;
    wcount++;
}

static void consume(const rb_item_t *it)
{
    /* shift-register window with 50 % overlap */
    if (win_fill == WIN) {
        memmove(WINBUF, WINBUF + HOP * MPU_REGS, (WIN - HOP) * MPU_REGS * sizeof(float));
        win_fill = WIN - HOP;
    }
    mpu6050_to_phys(it->raw, &WINBUF[win_fill * MPU_REGS]);
    win_fill++;
    since_hop++;
    last_target = it->t_target_ns;
    if (win_fill == WIN && (wcount == 0 ? 1 : since_hop >= HOP)) {
        since_hop = 0;
        process_window();
    }
}

static void *task_dsp(void *arg)
{
    (void)arg;
    pin_to(1);
    rb_item_t it;
    for (;;) {
        if (rb_pop(&RB, &it)) consume(&it);
        else if (producer_done && rb_count(&RB) == 0) break;
        else if (opt_rt) {                 /* queue empty: block briefly, like xQueueReceive */
            struct timespec ts = {0, 20000};
            nanosleep(&ts, NULL);
        }
    }
    return NULL;
}

/* ------------------------------------------- single-core baseline ---------- */
static void *task_single(void *arg)
{
    (void)arg;
    pin_to(0);
    rb_item_t it;
    for (uint64_t k = 0; k < n_samples; k++) {
        acquire_one(k, &it);
        consume(&it);                                    /* processing blocks reads */
    }
    return NULL;
}

/* ------------------------------------------------------------- main -------- */
static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static int arg(const char *a, const char *name, const char **val)
{
    size_t n = strlen(name);
    if (strncmp(a, name, n) == 0 && a[n] == '=') { *val = a + n + 1; return 1; }
    return 0;
}

int main(int argc, char **argv)
{
    motor_default(&M, 1);
    const char *v;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (arg(a, "--seconds", &v)) opt_seconds = atof(v);
        else if (arg(a, "--mode", &v)) opt_dual = strcmp(v, "single") != 0;
        else if (arg(a, "--pace", &v)) opt_rt = strcmp(v, "fast") != 0;
        else if (arg(a, "--i2c-us", &v)) opt_i2c_us = atof(v);
        else if (arg(a, "--proc-us", &v)) opt_proc_us = atof(v);
        else if (arg(a, "--learn-s", &v)) opt_learn_s = atof(v);
        else if (arg(a, "--detector", &v)) opt_kind = strcmp(v, "zsm") == 0 ? DET_ZSM : DET_MAHA;
        else if (arg(a, "--center", &v)) opt_center = atof(v);
        else if (arg(a, "--sens", &v)) opt_sens = atof(v);
        else if (arg(a, "--out", &v)) opt_out = v;
        else if (arg(a, "--stats", &v)) opt_stats = v;
        else if (strcmp(a, "--quiet") == 0) opt_quiet = 1;
        /* motor configuration */
        else if (arg(a, "--seed", &v)) M.seed = strtoull(v, NULL, 10);
        else if (arg(a, "--fs", &v)) M.fs = atof(v);
        else if (arg(a, "--rpm", &v)) M.rpm = atof(v);
        else if (arg(a, "--speed-drift", &v)) M.speed_drift = atof(v);
        else if (arg(a, "--load-amp", &v)) M.load_amp = atof(v);
        else if (arg(a, "--load-period-s", &v)) M.load_period_s = atof(v);
        else if (arg(a, "--a1", &v)) M.a1 = atof(v);
        else if (arg(a, "--case-temp-c", &v)) M.case_temp_c = atof(v);
        else if (arg(a, "--ambient-amp", &v)) M.ambient_amp = atof(v);
        else if (arg(a, "--ambient-period-s", &v)) M.ambient_period_s = atof(v);
        else if (arg(a, "--fault", &v)) M.fault = atoi(v);
        else if (arg(a, "--onset-s", &v)) M.onset_s = atof(v);
        else if (arg(a, "--ramp-s", &v)) M.ramp_s = atof(v);
        else if (arg(a, "--tau-s", &v)) M.tau_s = atof(v);
        else { fprintf(stderr, "unknown option %s\n", a); return 2; }
    }
    if (opt_i2c_us < 0) opt_i2c_us = mpu6050_i2c_read_us();
    fft_init();
    det_init(&DET, opt_kind, N_FEAT, (float)opt_center, (float)opt_sens);
    n_samples = (uint64_t)(opt_seconds * M.fs);
    DEC = calloc(n_samples / HOP + 2, sizeof *DEC);
    late_hist = calloc(n_samples + 1, sizeof *late_hist);

    t0_ns = now_ns() + 20000000ull;                       /* start in 20 ms */
    uint64_t wall0 = now_ns();
    pthread_t a, b;
    if (opt_dual) {
        pthread_create(&b, NULL, task_dsp, NULL);
        pthread_create(&a, NULL, task_acq, NULL);
        pthread_join(a, NULL);
        pthread_join(b, NULL);
    } else {
        pthread_create(&a, NULL, task_single, NULL);
        pthread_join(a, NULL);
    }
    double wall = (now_ns() - (opt_rt ? t0_ns : wall0)) / 1e9;

    /* decisions */
    FILE *fo = opt_out ? fopen(opt_out, "w") : (opt_quiet ? NULL : stdout);
    if (fo) {
        fprintf(fo, "window,t_end_s,learning,similarity,anomaly,ax_rms_g,temp_c,proc_us,e2e_us\n");
        for (uint32_t i = 0; i < n_dec; i++) {
            decision_t *d = &DEC[i];
            fprintf(fo, "%u,%.3f,%u,%.0f,%u,%.5f,%.3f,%.2f,%.2f\n", d->w, d->t_end_s, d->learning,
                    d->sim, d->anomaly, d->ax_rms, d->temp, d->proc_us, d->e2e_us);
        }
        if (fo != stdout) fclose(fo);
    }

    /* statistics */
    double p99 = 0, p50 = 0, e2e_p99 = 0, e2e_max = 0, proc_max = 0, proc_mean = 0;
    if (opt_rt && late_n) {
        qsort(late_hist, late_n, sizeof(double), cmp_d);
        p50 = late_hist[late_n / 2];
        p99 = late_hist[(size_t)(late_n * 0.99)];
    }
    uint32_t ninf = 0;
    double *e2e = calloc(n_dec + 1, sizeof(double));
    for (uint32_t i = 0; i < n_dec; i++) {
        proc_mean += DEC[i].proc_us;
        if (DEC[i].proc_us > proc_max) proc_max = DEC[i].proc_us;
        e2e[ninf++] = DEC[i].e2e_us;
    }
    if (n_dec) proc_mean /= n_dec;
    if (ninf) {
        qsort(e2e, ninf, sizeof(double), cmp_d);
        e2e_p99 = e2e[(size_t)(ninf * 0.99)];
        e2e_max = e2e[ninf - 1];
    }
    FILE *fs = opt_stats ? fopen(opt_stats, "w") : stderr;
    fprintf(fs, "{\"mode\":\"%s\",\"pace\":\"%s\",\"fs_hz\":%.1f,\"seconds\":%.2f,\"samples\":%llu,"
                "\"windows\":%u,\"i2c_us\":%.1f,\"proc_us_target\":%.1f,"
                "\"lateness_us_mean\":%.2f,\"lateness_us_p50\":%.2f,\"lateness_us_p99\":%.2f,\"lateness_us_max\":%.2f,"
                "\"samples_late_over_period\":%llu,\"samples_lost_fifo\":%llu,\"samples_lost_ring\":%llu,"
                "\"ring_high_water\":%u,\"proc_us_mean\":%.2f,\"proc_us_max\":%.2f,"
                "\"e2e_latency_us_p99\":%.2f,\"e2e_latency_us_max\":%.2f,"
                "\"core0_util\":%.4f,\"core1_util\":%.4f,\"wall_s\":%.3f,\"model_bytes\":%u}\n",
            opt_dual ? "dual" : "single", opt_rt ? "rt" : "fast", M.fs, opt_seconds,
            (unsigned long long)n_samples, n_dec, opt_i2c_us, opt_proc_us,
            late_n ? late_sum_ns / late_n / 1000.0 : 0.0, p50, p99, late_max_ns / 1000.0,
            (unsigned long long)late_over_period, (unsigned long long)lost_fifo,
            (unsigned long long)lost_ring, ring_high, proc_mean, proc_max, e2e_p99, e2e_max,
            opt_dual ? acq_busy_ns / 1e9 / wall : (acq_busy_ns + dsp_busy_ns) / 1e9 / wall,
            opt_dual ? dsp_busy_ns / 1e9 / wall : 0.0, wall, det_model_bytes(&DET));
    if (fs != stderr) fclose(fs);
    return 0;
}
