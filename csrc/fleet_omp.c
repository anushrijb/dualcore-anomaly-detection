/* ---------------------------------------------------------------------------
 * fleet_omp.c - fog / cloud tier: OpenMP processing of a huge fleet dataset
 *
 * Dataset : M motors x W windows, each window = 256 samples x 7 MPU6050
 *           registers (int16).  Default 64 x 512 = 32 768 windows = 117 MB.
 *           Motors 0..M/2-1 are healthy, M/2..M-1 develop faults 1..5 (onset
 *           900 s), so anomalous windows are concentrated in the 2nd half of
 *           the index range - exactly what makes static scheduling unbalanced.
 *
 * Stages (each an OpenMP loop; schedule chosen at run time):
 *   S1 gen    : regenerate the raw sensor archive              (uniform, compute bound)
 *   S2 feat   : 31 features per window, FFT                    (uniform)
 *   S3 train  : per-motor Mahalanobis models (Welford + Cholesky + inverse),
 *               fault-class models by parallel array reduction (TLP + DLP)
 *   S4 score  : Mahalanobis distance of every window, blocks of 16 windows,
 *               triangular matrix x matrix product - scalar vs SIMD   (DLP)
 *   S5 deep   : ONLY flagged windows: re-read 1024 contiguous samples, envelope
 *               spectrum (3 x 1024-pt FFT), classify fault type     (IRREGULAR)
 *   ILP       : dot product with 1/2/4/8 independent accumulators (cache resident)
 *
 * --sweep writes a CSV of every timing; a JSON summary carries the dataset,
 * diagnosis accuracy and the serial-vs-parallel verification.
 * ------------------------------------------------------------------------- */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpu6050_sim.h"
#include "dsp_features.h"
#include "detector.h"

#define D N_FEAT
#define DC (N_FEAT + 1)          /* class-model dimension: deviation + envelope */
#define BLK 16                   /* windows per SIMD block in S4 */
#define DEEP_N (2 * WIN)        /* 512 contiguous samples for the envelope spectrum */
#define S4_PASSES 50            /* S4 is short: time 50 passes, report per pass */

static int M = 64, W = 512, REPS = 3;
static float CENTER = 1.6f, SENS = 3.0f, SPAN_S = 1800.0f;
static motor_cfg_t *CFG;
static int16_t *RAW;             /* [N][WIN][7] */
static float *X;                 /* [N][D] features */
static float *DIST;              /* [N] distance to own-motor model */
static unsigned char *FLAG;      /* [N] similarity < 90 */
static float *ENV;               /* [N] envelope BPFO ratio (deep stage) */
static signed char *PRED;        /* [N] predicted class (deep stage) */
static size_t N;

typedef struct { float mean[D], scale[D], P[D * D]; } model_t;   /* P = inverse(L) */
static model_t *MOD;
typedef struct { double mean[DC]; double P[DC * DC]; int n; } cmodel_t;
static cmodel_t CLS[FAULT_N];

static inline size_t win_start(size_t i) { return (size_t)((i % W) * (SPAN_S * 1000.0 / W)); }
static inline int motor_of(size_t i) { return (int)(i / W); }
static inline float t_of(size_t i) { return (float)(win_start(i) + WIN) / 1000.0f; }

/* ------------------------------------------------------------ dataset ----- */
static void make_fleet(void)
{
    CFG = calloc(M, sizeof *CFG);
    for (int m = 0; m < M; m++) {
        motor_cfg_t *c = &CFG[m];
        motor_default(c, 10000 + m);
        unsigned h = (unsigned)(m * 2654435761u);
        c->rpm = 1420.0f + (h % 80);
        c->load_amp = 0.15f + 0.2f * ((h >> 8) % 100) / 100.0f;
        c->a1 = 0.04f + 0.02f * ((h >> 16) % 100) / 100.0f;
        c->case_temp_c = 38.0f + 6.0f * ((h >> 4) % 100) / 100.0f;
        c->fault = m < M / 2 ? FAULT_NONE : 1 + (m - M / 2) % 5;
        c->onset_s = 900.0f;
        c->ramp_s = 600.0f;
    }
}

/* -------------------------------------------------------------- stages ---- */
static void s1_gen(void)
{
    #pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < N; i++) {
        const motor_cfg_t *c = &CFG[motor_of(i)];
        size_t n0 = win_start(i);
        int16_t *r = &RAW[i * WIN * MPU_REGS];
        for (int k = 0; k < WIN; k++) mpu6050_read(c, n0 + k, &r[k * MPU_REGS]);
    }
}

static void s2_feat(void)
{
    #pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < N; i++)
        features_window_raw(&RAW[i * WIN * MPU_REGS], 1000.0f, &X[i * D]);
}

static void invert_lower(const float *L, float *P, int n)
{
    memset(P, 0, sizeof(float) * n * n);
    for (int c = 0; c < n; c++) {                       /* solve L p = e_c */
        P[c * n + c] = 1.0f / L[c * n + c];
        for (int i = c + 1; i < n; i++) {
            float s = 0.0f;
            for (int k = c; k < i; k++) s += L[i * n + k] * P[k * n + c];
            P[i * n + c] = -s / L[i * n + i];
        }
    }
}

static void s3_train_motors(void)
{
    #pragma omp parallel for schedule(runtime)
    for (int m = 0; m < M; m++) {
        detector_t det;
        det_init(&det, DET_MAHA, D, CENTER, SENS);
        for (int w = 0; w < W; w++) {
            size_t i = (size_t)m * W + w;
            if (t_of(i) <= 480.0f) det_learn(&det, &X[i * D]);   /* healthy commissioning period */
        }
        det_finalize(&det);
        memcpy(MOD[m].mean, det.mean, sizeof(float) * D);
        memcpy(MOD[m].scale, det.scale, sizeof(float) * D);
        invert_lower(det.L, MOD[m].P, D);
    }
}

/* S4: scalar and SIMD versions of the same blocked kernel ------------------ */
#define S4_BODY(PRAGMA)                                                              \
    for (size_t b0 = 0; b0 < N; b0 += BLK) {                                         \
        const model_t *mo = &MOD[motor_of(b0)];                                      \
        float Z[D][BLK], Y[BLK], S2[BLK], MX[BLK];                                   \
        int nb = (int)((N - b0) < BLK ? (N - b0) : BLK);                             \
        for (int j = 0; j < D; j++) {                                                \
            PRAGMA                                                                   \
            for (int b = 0; b < BLK; b++)                                            \
                Z[j][b] = b < nb ? (X[(b0 + b) * D + j] - mo->mean[j]) / mo->scale[j] : 0.0f; \
        }                                                                            \
        PRAGMA                                                                       \
        for (int b = 0; b < BLK; b++) { S2[b] = 0.0f; MX[b] = 0.0f; }                \
        for (int r = 0; r < D; r++) {                                                \
            PRAGMA                                                                   \
            for (int b = 0; b < BLK; b++) Y[b] = 0.0f;                               \
            for (int k = 0; k <= r; k++) {                                           \
                float p = mo->P[r * D + k];                                          \
                PRAGMA                                                               \
                for (int b = 0; b < BLK; b++) Y[b] += p * Z[k][b];                   \
            }                                                                        \
            PRAGMA                                                                   \
            for (int b = 0; b < BLK; b++) {                                          \
                S2[b] += Y[b] * Y[b];                                                \
                float a = fabsf(Y[b]); MX[b] = a > MX[b] ? a : MX[b];                \
            }                                                                        \
        }                                                                            \
        for (int b = 0; b < nb; b++) {                                               \
            float rms = sqrtf(S2[b] / D), pk = MX[b] / 3.0f;                         \
            float d = rms > pk ? rms : pk;                                           \
            DIST[b0 + b] = d;                                                        \
            float s = 100.0f / (1.0f + expf(fminf(60.0f, SENS * (d - CENTER))));     \
            FLAG[b0 + b] = floorf(s + 0.5f) < 90.0f;                                 \
        }                                                                            \
    }

__attribute__((optimize("no-tree-vectorize")))
static void s4_score_scalar(void)
{
    #pragma omp parallel for schedule(runtime)
    S4_BODY()
}

static void s4_score_simd(void)
{
    #pragma omp parallel for schedule(runtime)
    S4_BODY(_Pragma("omp simd"))
}

static void s4_simd_passes(void) { for (int p = 0; p < S4_PASSES; p++) s4_score_simd(); }
static void s4_scalar_passes(void) { for (int p = 0; p < S4_PASSES; p++) s4_score_scalar(); }

/* flag threshold calibrated on TRAINING motors' healthy windows (480-900 s):
   centre chosen so that 1 % of them fall under 90 % similarity */
static int cmpf(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static void calibrate_center(void)
{
    float *v = malloc(N * sizeof(float));
    size_t n = 0;
    for (size_t i = 0; i < N; i++) {
        int m = motor_of(i);
        float t = t_of(i);
        if (!(m & 1) && t > 480.0f && motor_severity(&CFG[m], t) == 0.0f) v[n++] = DIST[i];
    }
    qsort(v, n, sizeof(float), cmpf);
    float q99 = n ? v[(size_t)(0.99 * (n - 1))] : 1.0f;
    CENTER = q99 + logf(9.0f) / SENS;
    free(v);
}

/* S5: deep analysis of flagged windows only (irregular cost) --------------- */
static void deviation(size_t i, double *v)
{
    const model_t *mo = &MOD[motor_of(i)];
    for (int j = 0; j < D; j++) v[j] = (X[i * D + j] - mo->mean[j]) / mo->scale[j];
}

static void s5_deep(void)
{
    #pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < N; i++) {
        PRED[i] = -1;
        ENV[i] = 0.0f;
        if (!FLAG[i]) continue;                           /* only flagged windows */
        const motor_cfg_t *c = &CFG[motor_of(i)];
        float ax[DEEP_N];
        int16_t r[MPU_REGS];
        size_t n0 = win_start(i);
        for (int k = 0; k < DEEP_N; k++) {                /* re-read archive: 1024 samples */
            mpu6050_read(c, n0 + k, r);
            ax[k] = r[0] / MPU_ACCEL_LSB_PER_G;
        }
        ENV[i] = envelope_bpfo_ratio(ax, DEEP_N, 1000.0f, c->rpm / 60.0f);
        int cls_ready = CLS[1].n > 0;
        if (cls_ready) {                                  /* nearest fault class */
            double v[DC], best = 1e300;
            deviation(i, v);
            v[D] = log10(ENV[i] + 1e-6);
            for (int k = 1; k < FAULT_N; k++) {
                if (CLS[k].n < 2) continue;                   /* class unseen in training */
                double y2 = 0.0, z[DC];
                for (int j = 0; j < DC; j++) z[j] = v[j] - CLS[k].mean[j];
                for (int a = 0; a < DC; a++) {
                    double y = 0.0;
                    for (int b = 0; b <= a; b++) y += CLS[k].P[a * DC + b] * z[b];
                    y2 += y * y;
                }
                if (y2 < best) { best = y2; PRED[i] = (signed char)k; }
            }
        }
    }
}

/* class models: parallel reduction of means and co-moments over labelled
 * flagged windows of TRAINING motors (even index).  Two passes, double. */
static void s3_train_classes(void)
{
    double sum[FAULT_N][DC] = {{0}}, cov[FAULT_N][DC * DC];
    int cnt[FAULT_N] = {0};
    memset(cov, 0, sizeof cov);
    #pragma omp parallel for schedule(runtime) reduction(+:sum[:FAULT_N][:DC], cnt[:FAULT_N])
    for (size_t i = 0; i < N; i++) {
        int m = motor_of(i), k = CFG[m].fault;
        if (k == 0 || (m & 1) || !FLAG[i] || motor_severity(&CFG[m], t_of(i)) < 0.3f) continue;
        double v[DC];
        deviation(i, v);
        v[D] = log10(ENV[i] + 1e-6);
        for (int j = 0; j < DC; j++) sum[k][j] += v[j];
        cnt[k]++;
    }
    for (int k = 1; k < FAULT_N; k++)
        for (int j = 0; j < DC; j++) CLS[k].mean[j] = cnt[k] ? sum[k][j] / cnt[k] : 0.0;
    #pragma omp parallel for schedule(runtime) reduction(+:cov[:FAULT_N][:DC * DC])
    for (size_t i = 0; i < N; i++) {
        int m = motor_of(i), k = CFG[m].fault;
        if (k == 0 || (m & 1) || !FLAG[i] || motor_severity(&CFG[m], t_of(i)) < 0.3f) continue;
        double v[DC];
        deviation(i, v);
        v[D] = log10(ENV[i] + 1e-6);
        for (int a = 0; a < DC; a++) v[a] -= CLS[k].mean[a];
        for (int a = 0; a < DC; a++) {
            #pragma omp simd
            for (int b = 0; b < DC; b++) cov[k][a * DC + b] += v[a] * v[b];   /* DLP: outer product row */
        }
    }
    for (int k = 1; k < FAULT_N; k++) {
        CLS[k].n = cnt[k];
        if (cnt[k] < 2) continue;
        /* shrink, Cholesky, invert (double, D=32: negligible serial part) */
        double A[DC * DC], L[DC * DC] = {0};
        for (int a = 0; a < DC; a++)
            for (int b = 0; b < DC; b++) {
                double c = cov[k][a * DC + b] / (cnt[k] - 1);
                A[a * DC + b] = a == b ? c + 0.05 : 0.7 * c;
            }
        for (int j = 0; j < DC; j++) {
            double s = A[j * DC + j];
            for (int q = 0; q < j; q++) s -= L[j * DC + q] * L[j * DC + q];
            L[j * DC + j] = sqrt(s > 1e-12 ? s : 1e-12);
            for (int i2 = j + 1; i2 < DC; i2++) {
                double t = A[i2 * DC + j];
                for (int q = 0; q < j; q++) t -= L[i2 * DC + q] * L[j * DC + q];
                L[i2 * DC + j] = t / L[j * DC + j];
            }
        }
        memset(CLS[k].P, 0, sizeof CLS[k].P);
        for (int c = 0; c < DC; c++) {
            CLS[k].P[c * DC + c] = 1.0 / L[c * DC + c];
            for (int i2 = c + 1; i2 < DC; i2++) {
                double s = 0.0;
                for (int q = c; q < i2; q++) s += L[i2 * DC + q] * CLS[k].P[q * DC + c];
                CLS[k].P[i2 * DC + c] = -s / L[i2 * DC + i2];
            }
        }
    }
}

/* ILP micro-benchmark: same dot product, 1/2/4/8 independent accumulators ---- */
#define ILP_N 4096
#define ILP_REPS 20000
static float VA[ILP_N], VB[ILP_N];

#define DOT_K(K)                                                                  \
__attribute__((optimize("no-tree-vectorize"), noinline))                          \
static float dot_acc##K(void) {                                                   \
    float acc[K] = {0}, total = 0.0f;                                             \
    for (int r = 0; r < ILP_REPS; r++) {                                          \
        for (int i = 0; i < ILP_N; i += K)                                        \
            for (int k = 0; k < K; k++) acc[k] += VA[i + k] * VB[i + k];          \
        __asm__ volatile("" ::: "memory");                                        \
    }                                                                             \
    for (int k = 0; k < K; k++) total += acc[k];                                  \
    return total;                                                                 \
}
DOT_K(1) DOT_K(2) DOT_K(4) DOT_K(8)

__attribute__((noinline)) static float dot_simd(void)
{
    float total = 0.0f;
    for (int r = 0; r < ILP_REPS; r++) {
        float acc = 0.0f;
        #pragma omp simd reduction(+:acc)
        for (int i = 0; i < ILP_N; i++) acc += VA[i] * VB[i];
        total += acc;
        __asm__ volatile("" ::: "memory");
    }
    return total;
}

/* --------------------------------------------------------------- harness -- */
static FILE *CSV;
static const char *SCHED_NAME[] = {"", "static", "dynamic", "guided", "auto"};

static double timeit(void (*fn)(void), int reps)
{
    double best[16], t;
    if (reps > 16) reps = 16;
    for (int r = 0; r < reps; r++) {
        t = omp_get_wtime();
        fn();
        best[r] = omp_get_wtime() - t;
    }
    for (int a = 0; a < reps; a++)                     /* median */
        for (int b = a + 1; b < reps; b++)
            if (best[b] < best[a]) { t = best[a]; best[a] = best[b]; best[b] = t; }
    return best[reps / 2];
}

static void record(const char *exp, const char *stage, int thr, omp_sched_t k, int chunk,
                   const char *variant, double sec, double items, double flops)
{
    fprintf(CSV, "%s,%s,%d,%s,%d,%s,%.6f,%.0f,%.1f,%.4f\n", exp, stage, thr, SCHED_NAME[k], chunk,
            variant, sec, items, items / sec, flops / sec / 1e9);
    fflush(CSV);
}

static void set(int thr, omp_sched_t k, int chunk)
{
    omp_set_num_threads(thr);
    omp_set_schedule(k, chunk);
}

static size_t count_flag(void)
{
    size_t c = 0;
    for (size_t i = 0; i < N; i++) c += FLAG[i];
    return c;
}

int main(int argc, char **argv)
{
    const char *out = "fleet_timings.csv", *summary = "fleet_summary.json";
    char tlist[128] = "";
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--motors=", 9)) M = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--windows=", 10)) W = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--reps=", 7)) REPS = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--center=", 9)) CENTER = atof(argv[i] + 9);
        else if (!strncmp(argv[i], "--sens=", 7)) SENS = atof(argv[i] + 7);
        else if (!strncmp(argv[i], "--threads=", 10)) snprintf(tlist, sizeof tlist, "%s", argv[i] + 10);
        else if (!strncmp(argv[i], "--out=", 6)) out = argv[i] + 6;
        else if (!strncmp(argv[i], "--summary=", 10)) summary = argv[i] + 10;
    }
    int maxT = omp_get_num_procs();
    int TL[16], nT = 0;
    if (tlist[0]) {
        for (char *p = strtok(tlist, ","); p && nT < 16; p = strtok(NULL, ",")) TL[nT++] = atoi(p);
    } else {
        for (int t = 1; t < maxT; t *= 2) TL[nT++] = t;
        TL[nT++] = maxT;
    }
    maxT = TL[nT - 1];
    N = (size_t)M * W;
    fft_init();
    make_fleet();
    RAW = malloc(N * WIN * MPU_REGS * sizeof(int16_t));
    X = malloc(N * D * sizeof(float));
    DIST = malloc(N * sizeof(float));
    FLAG = malloc(N);
    ENV = malloc(N * sizeof(float));
    PRED = malloc(N);
    MOD = calloc(M, sizeof *MOD);
    if (!RAW || !X) { fprintf(stderr, "out of memory for %zu windows\n", N); return 1; }
    for (int i = 0; i < ILP_N; i++) { VA[i] = 1.0f + (i % 7) * 1e-3f; VB[i] = 1.0f - (i % 5) * 1e-3f; }

    CSV = fopen(out, "w");
    fprintf(CSV, "experiment,stage,threads,schedule,chunk,variant,seconds,items,items_per_s,gflops\n");
    fprintf(stderr, "fleet: %d motors x %d windows = %zu windows, raw %.1f MB, threads %d..%d\n",
            M, W, N, N * WIN * MPU_REGS * 2 / 1e6, TL[0], maxT);

    /* ---- reference run (1 thread) for verification ---------------------- */
    set(1, omp_sched_static, 0);
    s1_gen(); s2_feat(); s3_train_motors(); s4_score_simd();
    calibrate_center(); s4_score_simd();
    s5_deep(); s3_train_classes(); s5_deep();
    float *Xref = malloc(N * D * sizeof(float)), *Eref = malloc(N * sizeof(float));
    memcpy(Xref, X, N * D * sizeof(float));
    memcpy(Eref, ENV, N * sizeof(float));
    double cls_ref = 0;
    for (int k = 1; k < FAULT_N; k++) for (int j = 0; j < DC * DC; j++) cls_ref += fabs(CLS[k].P[j]);
    size_t nflag = count_flag();
    double flops_s4 = (double)N * (D * (D + 1) + 8.0 * D);

    /* ---- A. thread scaling of every stage ------------------------------- */
    for (int ti = 0; ti < nT; ti++) {
        int T = TL[ti];
        set(T, omp_sched_static, 0);
        record("scaling", "S1_gen", T, omp_sched_static, 0, "", timeit(s1_gen, REPS), N, 0);
        record("scaling", "S2_feat", T, omp_sched_static, 0, "", timeit(s2_feat, REPS), N, 0);
        record("scaling", "S3_train", T, omp_sched_static, 0, "", timeit(s3_train_motors, REPS), M, 0);
        record("scaling", "S4_score_simd", T, omp_sched_static, 0, "simd", timeit(s4_simd_passes, REPS) / S4_PASSES, N, flops_s4);
        set(T, omp_sched_dynamic, 16);
        record("scaling", "S5_deep", T, omp_sched_dynamic, 16, "", timeit(s5_deep, REPS), nflag, 0);
        set(T, omp_sched_static, 0);
        record("scaling", "S5_deep_static", T, omp_sched_static, 0, "", timeit(s5_deep, REPS), nflag, 0);
    }

    /* ---- B. schedule x chunk size at max threads ------------------------- */
    omp_sched_t kinds[3] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
    int chunks[6] = {0, 1, 4, 16, 64, 512};
    for (int a = 0; a < 3; a++)
        for (int c = 0; c < 6; c++) {
            set(maxT, kinds[a], chunks[c]);
            record("schedule", "S5_deep", maxT, kinds[a], chunks[c], "", timeit(s5_deep, REPS), nflag, 0);
            record("schedule", "S2_feat", maxT, kinds[a], chunks[c], "", timeit(s2_feat, REPS), N, 0);
        }

    /* ---- C. DLP: scalar vs SIMD scoring ---------------------------------- */
    int dlpT[2] = {1, maxT};
    for (int q = 0; q < 2; q++) {
        set(dlpT[q], omp_sched_static, 0);
        record("dlp", "S4_score", dlpT[q], omp_sched_static, 0, "scalar", timeit(s4_scalar_passes, REPS) / S4_PASSES, N, flops_s4);
        record("dlp", "S4_score", dlpT[q], omp_sched_static, 0, "simd", timeit(s4_simd_passes, REPS) / S4_PASSES, N, flops_s4);
    }

    /* ---- D. ILP: independent accumulators -------------------------------- */
    double fl = 2.0 * ILP_N * ILP_REPS, t;
    float chk = 0;
    t = omp_get_wtime(); chk += dot_acc1(); record("ilp", "dot", 1, omp_sched_static, 0, "acc1", omp_get_wtime() - t, ILP_N * (double)ILP_REPS, fl);
    t = omp_get_wtime(); chk += dot_acc2(); record("ilp", "dot", 1, omp_sched_static, 0, "acc2", omp_get_wtime() - t, ILP_N * (double)ILP_REPS, fl);
    t = omp_get_wtime(); chk += dot_acc4(); record("ilp", "dot", 1, omp_sched_static, 0, "acc4", omp_get_wtime() - t, ILP_N * (double)ILP_REPS, fl);
    t = omp_get_wtime(); chk += dot_acc8(); record("ilp", "dot", 1, omp_sched_static, 0, "acc8", omp_get_wtime() - t, ILP_N * (double)ILP_REPS, fl);
    t = omp_get_wtime(); chk += dot_simd(); record("ilp", "dot", 1, omp_sched_static, 0, "simd", omp_get_wtime() - t, ILP_N * (double)ILP_REPS, fl);

    /* ---- E. verification: parallel == serial ------------------------------ */
    set(maxT, omp_sched_dynamic, 16);
    s1_gen(); s2_feat(); s3_train_motors(); s4_score_simd(); s5_deep(); s3_train_classes(); s5_deep();
    double dx = 0, de = 0, cls_par = 0;
    for (size_t i = 0; i < N * D; i++) dx = fmax(dx, fabs(X[i] - Xref[i]));
    for (size_t i = 0; i < N; i++) de = fmax(de, fabs(ENV[i] - Eref[i]));
    for (int k = 1; k < FAULT_N; k++) for (int j = 0; j < DC * DC; j++) cls_par += fabs(CLS[k].P[j]);

    /* ---- diagnosis accuracy on TEST motors (odd index) -------------------- */
    int conf[FAULT_N][FAULT_N] = {{0}};
    size_t fa_normal = 0, normal_win = 0, det_fault = 0, fault_win = 0;
    for (size_t i = 0; i < N; i++) {
        int m = motor_of(i), k = CFG[m].fault;
        float sev = motor_severity(&CFG[m], t_of(i));
        if (t_of(i) <= 480.0f) continue;                       /* learning period */
        if (k == 0 || sev == 0.0f) { normal_win++; fa_normal += FLAG[i]; }
        if (k > 0 && sev >= 0.3f) {
            fault_win++; det_fault += FLAG[i];
            if ((m & 1) && FLAG[i] && PRED[i] > 0) conf[k][(int)PRED[i]]++;
        }
    }
    int correct = 0, total = 0;
    for (int a = 1; a < FAULT_N; a++) for (int b = 1; b < FAULT_N; b++) { total += conf[a][b]; if (a == b) correct += conf[a][b]; }

    FILE *js = fopen(summary, "w");
    fprintf(js, "{\"motors\":%d,\"windows_per_motor\":%d,\"windows\":%zu,\"raw_mb\":%.1f,\"feature_matrix_mb\":%.2f,"
                "\"max_threads\":%d,\"flagged_windows\":%zu,\"flagged_fraction\":%.4f,"
                "\"false_alarm_rate_healthy\":%.5f,\"detection_rate_faulty_sev03\":%.5f,"
                "\"diagnosis_accuracy_test_motors\":%.4f,\"diagnosis_n\":%d,\"confusion\":[",
            M, W, N, N * WIN * MPU_REGS * 2 / 1e6, N * D * 4 / 1e6, maxT, nflag, (double)nflag / N,
            normal_win ? (double)fa_normal / normal_win : 0, fault_win ? (double)det_fault / fault_win : 0,
            total ? (double)correct / total : 0, total);
    for (int a = 1; a < FAULT_N; a++) {
        fprintf(js, "%s[", a > 1 ? "," : "");
        for (int b = 1; b < FAULT_N; b++) fprintf(js, "%s%d", b > 1 ? "," : "", conf[a][b]);
        fprintf(js, "]");
    }
    fprintf(js, "],\"verify_max_abs_diff_features\":%.3g,\"verify_max_abs_diff_envelope\":%.3g,"
                "\"verify_rel_diff_class_models\":%.3g,\"flag_center\":%.4f,\"checksum\":%.3f}\n",
            dx, de, fabs(cls_par - cls_ref) / (cls_ref + 1e-30), CENTER, chk);
    fclose(js);
    fclose(CSV);
    fprintf(stderr, "flagged %zu (%.1f%%), diagnosis acc %.3f (n=%d), verify dX=%g dEnv=%g\n",
            nflag, 100.0 * nflag / N, total ? (double)correct / total : 0, total, dx, de);
    return 0;
}
