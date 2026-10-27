/* ---------------------------------------------------------------------------
 * edge_ms.c - sample-project edge node (ESP32 + MPU6050 + Hall + DHT22 +
 *             ACS712 + INMP441 + L298N), two cores, with ZERO-LOSS accounting.
 *
 *  Core 0 : every 1 ms read all sensors (MPU6050 burst over I2C ~385 us, ACS712
 *           ADC, Hall edge from the ISR, 8 audio samples from I2S DMA, DHT22
 *           every 2 s), stamp the tick with a sequence number, push to queue.
 *  Core 1 : check sequence continuity, build windows, 18 features, multi-input
 *           Z-score model (on-device learning), staged speed control (L298N
 *           duty goes back to Core 0's motor), packetise for Wi-Fi.
 *  Wi-Fi  : packet loss + outages; optional store-and-forward with ACKs.
 *  Fog    : checks window ids and sample coverage end to end.
 *
 *  Where a sample can be lost, and how each loss is counted:
 *   L1 sensor FIFO overflow  : Core 0 later than 85 samples (MPU6050 FIFO)
 *   L2 queue full            : Core 1 not keeping up
 *   L3 sequence gaps         : seen by Core 1 (must equal L1 + L2)
 *   L4 uplink                : packets never delivered
 *   L5 fog coverage          : samples whose window never reached the fog
 * ------------------------------------------------------------------------- */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "ms_core.h"
#include "dsp_features.h"
#include "detector.h"

#define Q_CAP 1024                       /* Core 0 -> Core 1 queue (ticks)      */

/* ---------------- options ---------------- */
static ms_cfg_t CFG;
static double o_seconds = 60, o_i2c_us = 385, o_learn_s = 300, o_center = 3.0, o_sens = 3.0;
static double o_loss = 0.02, o_proc_us = 0;
static int o_rt = 1, o_ctrl = CTRL_AI, o_sf = 1, o_sf_cap = 4096, o_dual = 1;
static char o_outages[256] = "";
static const char *o_stats = NULL, *o_out = NULL;

/* ---------------- shared ---------------- */
typedef struct {
    _Alignas(64) _Atomic uint32_t head;
    _Alignas(64) _Atomic uint32_t tail;
    ms_tick_t buf[Q_CAP];
} tickq_t;
static tickq_t Q;
static _Atomic float g_duty = 1.0f;
static _Atomic int producer_done = 0;
static uint64_t n_ticks, t0_ns;

static uint64_t L1_fifo = 0, L2_queue = 0, late_n = 0;
static double late_max = 0, late_sum = 0;
static double *late_hist;

static inline uint64_t now_ns(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static inline void spin_until(uint64_t t) { while (now_ns() < t) { } }
static void pin(int cpu)
{
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu % (int)sysconf(_SC_NPROCESSORS_ONLN), &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}
static int q_push(const ms_tick_t *t)
{
    uint32_t h = atomic_load_explicit(&Q.head, memory_order_relaxed);
    if (h - atomic_load_explicit(&Q.tail, memory_order_acquire) >= Q_CAP) return 0;
    Q.buf[h % Q_CAP] = *t;
    atomic_store_explicit(&Q.head, h + 1, memory_order_release);
    return 1;
}
static int q_pop(ms_tick_t *t)
{
    uint32_t tl = atomic_load_explicit(&Q.tail, memory_order_relaxed);
    if (tl == atomic_load_explicit(&Q.head, memory_order_acquire)) return 0;
    *t = Q.buf[tl % Q_CAP];
    atomic_store_explicit(&Q.tail, tl + 1, memory_order_release);
    return 1;
}

/* ---------------- Core 0 ---------------- */
static ms_plant_t PLANT;
static void acquire(uint64_t k)
{
    ms_tick_t tk;
    uint64_t target = t0_ns + k * 1000000ull;
    if (o_rt) spin_until(target);
    uint64_t start = now_ns();
    ms_plant_set_duty(&PLANT, atomic_load(&g_duty));     /* L298N PWM from Core 1 */
    ms_plant_step(&PLANT, &tk);
    if (o_rt) spin_until(start + (uint64_t)(o_i2c_us * 1000));
    if (o_rt) {
        double late = start > target ? (start - target) / 1000.0 : 0.0;   /* us */
        late_hist[late_n++] = late; late_sum += late; if (late > late_max) late_max = late;
        if (late > 85 * 1000.0) { L1_fifo++; return; }   /* MPU6050 FIFO overwrote it */
    }
    if (o_rt) { if (!q_push(&tk)) L2_queue++; }
    else while (!q_push(&tk)) { }
}
static void *task_core0(void *a) { (void)a; pin(0); for (uint64_t k = 0; k < n_ticks; k++) acquire(k); producer_done = 1; return NULL; }

/* ---------------- Wi-Fi + fog ---------------- */
typedef struct { uint32_t win, seq_last, n_new; uint8_t sim, state; float duty; } pkt_t;   /* n_new = samples Core 1 received since the previous packet */
static pkt_t *SF; static int sf_head = 0, sf_n = 0;        /* store-and-forward ring */
static uint64_t pkt_made = 0, pkt_attempts = 0, L4_uplink = 0, fog_rx = 0;
static uint8_t *fog_seen; static uint64_t fog_samples = 0, fog_last_seq = 0;
static uint64_t rng = 88172645463325252ull;
static double rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 11) * (1.0 / 9007199254740992.0); }
static double OUT_S[16], OUT_D[16]; static int n_out = 0;
static int link_up(double t) { for (int i = 0; i < n_out; i++) if (t >= OUT_S[i] && t < OUT_S[i] + OUT_D[i]) return 0; return 1; }
static void fog_receive(const pkt_t *p)
{
    fog_rx++;
    if (fog_seen[p->win]) return;                           /* duplicate (retransmission) */
    fog_seen[p->win] = 1;
    fog_samples += p->n_new;                                /* samples this packet accounts for */
    if (p->seq_last > fog_last_seq) fog_last_seq = p->seq_last;
}
static void uplink(const pkt_t *p, double t)
{
    pkt_made++;
    if (!o_sf) {                                            /* send now, 3 tries */
        for (int r = 0; r < 3; r++) { pkt_attempts++; if (link_up(t) && rnd() > o_loss) { fog_receive(p); return; } }
        L4_uplink++;
        return;
    }
    if (sf_n == o_sf_cap) { sf_head = (sf_head + 1) % o_sf_cap; sf_n--; L4_uplink++; }   /* buffer full: oldest lost */
    SF[(sf_head + sf_n) % o_sf_cap] = *p; sf_n++;
    while (sf_n && link_up(t)) {                           /* flush in order until a send fails */
        pkt_attempts++;
        if (rnd() < o_loss) break;                         /* no ACK: keep, retry next window */
        fog_receive(&SF[sf_head]); sf_head = (sf_head + 1) % o_sf_cap; sf_n--;
    }
}

/* ---------------- Core 1 ---------------- */
static ms_windower_t WD; static ms_window_t W; static detector_t DET; static ms_ctrl_t CT;
static uint64_t windows = 0, rx_at_last_pkt = 0, dev_last_seq = 0; static double proc_max_us = 0, proc_sum_us = 0;
static FILE *FO;
static void core1_tick(const ms_tick_t *tk)
{
    float duty = atomic_load(&g_duty);
    if (!ms_windower_push(&WD, tk, duty, &W)) return;
    uint64_t a = now_ns();
    float f[MS_NF], raw[6];
    double t = (tk->seq + 1) / (double)MS_FS;
    ms_features(&W, CFG.rpm_nom, f);
    ms_raw_values(&W, f, raw);
    int sim = -1;
    if (t <= o_learn_s) { det_learn(&DET, f); ms_ctrl_learn(&CT, raw); }
    else { if (!DET.ready) det_finalize(&DET); sim = det_similarity(&DET, f); }
    if (o_ctrl != CTRL_NONE && t > o_learn_s) atomic_store(&g_duty, ms_ctrl_step(&CT, (float)t, raw, sim));
    if (o_proc_us > 0) spin_until(a + (uint64_t)(o_proc_us * 1000));
    double us = (now_ns() - a) / 1000.0;
    proc_sum_us += us; if (us > proc_max_us) proc_max_us = us;
    pkt_t p = {(uint32_t)windows, tk->seq, (uint32_t)(WD.received - rx_at_last_pkt), (uint8_t)(sim < 0 ? 255 : sim), (uint8_t)CT.state, atomic_load(&g_duty)};
    rx_at_last_pkt = WD.received;
    dev_last_seq = tk->seq;
    uplink(&p, t);
    if (FO) fprintf(FO, "%llu,%.3f,%d,%d,%.2f,%.4f,%.2f,%.2f,%.1f,%.3f\n", (unsigned long long)windows, t, sim,
                    CT.state, atomic_load(&g_duty), raw[0], raw[2], raw[4], raw[1] * CFG.rpm_nom * W.duty, raw[5]);
    windows++;
}
static void *task_core1(void *a)
{
    (void)a; pin(1);
    ms_tick_t tk;
    for (;;) {
        if (q_pop(&tk)) core1_tick(&tk);
        else if (producer_done) { if (!q_pop(&tk)) break; core1_tick(&tk); }
        else if (o_rt) { struct timespec ts = {0, 20000}; nanosleep(&ts, NULL); }
    }
    return NULL;
}
static void *task_single(void *a)
{
    (void)a; pin(0);
    ms_tick_t tk;
    for (uint64_t k = 0; k < n_ticks; k++) { acquire(k); while (q_pop(&tk)) core1_tick(&tk); }
    return NULL;
}

static int cmpd(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return a < b ? -1 : a > b; }
static int opt(const char *a, const char *n, const char **v) { size_t l = strlen(n); if (!strncmp(a, n, l) && a[l] == '=') { *v = a + l + 1; return 1; } return 0; }

int main(int argc, char **argv)
{
    ms_cfg_default(&CFG, 1);
    const char *v;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (opt(a, "--seconds", &v)) o_seconds = atof(v);
        else if (opt(a, "--pace", &v)) o_rt = strcmp(v, "fast") != 0;
        else if (opt(a, "--mode", &v)) o_dual = strcmp(v, "single") != 0;
        else if (opt(a, "--ctrl", &v)) o_ctrl = !strcmp(v, "none") ? CTRL_NONE : (!strcmp(v, "threshold") ? CTRL_THRESHOLD : CTRL_AI);
        else if (opt(a, "--i2c-us", &v)) o_i2c_us = atof(v);
        else if (opt(a, "--proc-us", &v)) o_proc_us = atof(v);
        else if (opt(a, "--learn-s", &v)) o_learn_s = atof(v);
        else if (opt(a, "--center", &v)) o_center = atof(v);
        else if (opt(a, "--sens", &v)) o_sens = atof(v);
        else if (opt(a, "--wifi-loss", &v)) o_loss = atof(v);
        else if (opt(a, "--outages", &v)) snprintf(o_outages, sizeof o_outages, "%s", v);
        else if (opt(a, "--store-forward", &v)) o_sf = atoi(v);
        else if (opt(a, "--sf-cap", &v)) o_sf_cap = atoi(v);
        else if (opt(a, "--stats", &v)) o_stats = v;
        else if (opt(a, "--out", &v)) o_out = v;
        else if (opt(a, "--seed", &v)) CFG.seed = strtoull(v, NULL, 10);
        else if (opt(a, "--rpm-nom", &v)) CFG.rpm_nom = atof(v);
        else if (opt(a, "--a1", &v)) CFG.a1 = atof(v);
        else if (opt(a, "--ambient-c", &v)) CFG.ambient_c = atof(v);
        else if (opt(a, "--rh-pct", &v)) CFG.rh_pct = atof(v);
        else if (opt(a, "--drift-c", &v)) CFG.drift_c = atof(v);
        else if (opt(a, "--drift-rh", &v)) CFG.drift_rh = atof(v);
        else if (opt(a, "--drift-s", &v)) CFG.drift_s = atof(v);
        else if (opt(a, "--load-amp", &v)) CFG.load_amp = atof(v);
        else if (opt(a, "--load-period-s", &v)) CFG.load_period_s = atof(v);
        else if (opt(a, "--fault", &v)) CFG.fault = atoi(v);
        else if (opt(a, "--onset-s", &v)) CFG.onset_s = atof(v);
        else if (opt(a, "--ramp-s", &v)) CFG.ramp_s = atof(v);
        else { fprintf(stderr, "unknown option %s\n", a); return 2; }
    }
    for (char *p = strtok(o_outages, ","); p && n_out < 16; p = strtok(NULL, ",")) {
        if (sscanf(p, "%lf:%lf", &OUT_S[n_out], &OUT_D[n_out]) == 2) n_out++;
    }
    fft_init();
    ms_plant_init(&PLANT, &CFG);
    ms_ctrl_init(&CT, o_ctrl);
    det_init(&DET, DET_ZSM, MS_NF, (float)o_center, (float)o_sens);
    ms_detector_setup(&DET);
    n_ticks = (uint64_t)(o_seconds * MS_FS);
    late_hist = calloc(n_ticks + 1, sizeof(double));
    SF = calloc(o_sf_cap, sizeof(pkt_t));
    fog_seen = calloc(n_ticks / MS_HOP + 4, 1);
    FO = o_out ? fopen(o_out, "w") : NULL;
    if (FO) fprintf(FO, "window,t_s,similarity,state,duty,vib_rms_g,t_motor_c,rh_pct,rpm,current_a\n");

    t0_ns = now_ns() + 20000000ull;
    pthread_t a, b;
    if (o_dual) { pthread_create(&b, NULL, task_core1, NULL); pthread_create(&a, NULL, task_core0, NULL); pthread_join(a, NULL); pthread_join(b, NULL); }
    else { pthread_create(&a, NULL, task_single, NULL); pthread_join(a, NULL); }
    /* end of run: let the radio flush what is still buffered once the link is up */
    double t_end = o_seconds + 3600.0;
    while (o_sf && sf_n && link_up(t_end)) { pkt_attempts++; if (rnd() < o_loss) continue; fog_receive(&SF[sf_head]); sf_head = (sf_head + 1) % o_sf_cap; sf_n--; }
    if (FO) fclose(FO);

    /* trailing samples lost after the last one Core 1 saw are gaps too */
    if (WD.next_seq < n_ticks) WD.gaps += n_ticks - WD.next_seq;
    uint64_t fog_windows = 0, missing = 0;
    for (uint64_t w = 0; w < windows; w++) { fog_windows += fog_seen[w]; missing += !fog_seen[w]; }
    /* end-to-end check at the fog: samples accounted for by delivered packets
       vs the sequence numbers the device produced up to the last window */
    uint64_t last_seq = 0, covered = fog_samples;
    if (windows) last_seq = dev_last_seq;    /* newest sequence number the device packetised */
    double p99 = 0;
    if (late_n) { qsort(late_hist, late_n, sizeof(double), cmpd); p99 = late_hist[(size_t)(late_n * 0.99)]; }
    uint64_t lost_total = L1_fifo + L2_queue;
    FILE *fs = o_stats ? fopen(o_stats, "w") : stderr;
    fprintf(fs, "{\"mode\":\"%s\",\"pace\":\"%s\",\"seconds\":%.1f,\"samples_produced\":%llu,"
                "\"L1_sensor_fifo_lost\":%llu,\"L2_queue_lost\":%llu,\"L3_seq_gaps_seen_core1\":%llu,"
                "\"samples_received_core1\":%llu,\"windows\":%llu,\"packets\":%llu,\"send_attempts\":%llu,"
                "\"L4_uplink_lost_packets\":%llu,\"fog_windows\":%llu,\"fog_missing_windows\":%llu,"
                "\"samples_expected_at_fog\":%llu,\"L5_fog_samples_covered\":%llu,"
                "\"loss_rate_upper95_if_zero\":%.3g,\"lateness_us_p99\":%.2f,\"lateness_us_max\":%.2f,"
                "\"proc_us_mean\":%.2f,\"proc_us_max\":%.2f,\"wifi_loss\":%.3f,\"outages\":%d,\"store_forward\":%d,"
                "\"t_warn\":%.2f,\"t_stop\":%.2f,\"final_duty\":%.2f}\n",
            o_dual ? "dual" : "single", o_rt ? "rt" : "fast", o_seconds, (unsigned long long)n_ticks,
            (unsigned long long)L1_fifo, (unsigned long long)L2_queue, (unsigned long long)WD.gaps,
            (unsigned long long)WD.received, (unsigned long long)windows, (unsigned long long)pkt_made,
            (unsigned long long)pkt_attempts, (unsigned long long)L4_uplink, (unsigned long long)fog_windows,
            (unsigned long long)missing, (unsigned long long)(windows ? last_seq + 1 : 0), (unsigned long long)covered,
            lost_total == 0 ? 3.0 / n_ticks : (double)lost_total / n_ticks, p99, late_max,
            windows ? proc_sum_us / windows : 0, proc_max_us, o_loss, n_out, o_sf,
            CT.t_warn, CT.t_stop, atomic_load(&g_duty));
    if (fs != stderr) fclose(fs);
    return 0;
}
