/* ---------------------------------------------------------------------------
 * detector.h - one-class anomaly detectors with on-device learning
 *
 *   DET_ZSM  : diagonal Z-score model (the base paper's NanoEdge ZSM family)
 *   DET_MAHA : full-covariance Mahalanobis model (proposed) - learns the
 *              correlation between features with an online (Welford)
 *              covariance update, then a shrinkage-regularised Cholesky.
 *
 * Distance d = max(RMS of the standardised/whitened deviations, largest one / 3).
 * Both map d to a similarity 0..100 %:
 *      sim = 100 / (1 + exp(sens * (d - center)))
 * and report "anomaly" when sim < threshold (90 %), as in the base paper.
 * ------------------------------------------------------------------------- */
#ifndef DETECTOR_H
#define DETECTOR_H
#include <stdint.h>

#define DET_MAX_DIM 32

enum det_kind { DET_ZSM = 0, DET_MAHA = 1 };

typedef struct {
    int      kind, dim;
    uint32_t n;                                  /* learned windows           */
    float    center, sens, shrink, sigma_min;
    float    mean[DET_MAX_DIM];
    float    m2[DET_MAX_DIM * DET_MAX_DIM];      /* Welford co-moment (diag only for ZSM) */
    float    L[DET_MAX_DIM * DET_MAX_DIM];       /* Cholesky factor of shrunk covariance  */
    float    scale[DET_MAX_DIM];                 /* per-feature std used to standardise   */
    int      ready;
} detector_t;

void    det_init(detector_t *d, int kind, int dim, float center, float sens);
void    det_learn(detector_t *d, const float *x);
int     det_finalize(detector_t *d);            /* build model after learning; 0 = ok */
float   det_distance(const detector_t *d, const float *x);
uint8_t det_similarity(const detector_t *d, const float *x);
uint32_t det_model_bytes(const detector_t *d);  /* RAM the learned model needs */
/* self-optimising baseline (ZSM): exponentially-weighted update of mean and
   spread with a confident-normal window; call only when the window is normal */
void    det_adapt(detector_t *d, const float *x, float alpha);

#endif
