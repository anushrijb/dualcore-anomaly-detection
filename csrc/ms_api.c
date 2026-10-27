/* Closed-loop multi-sensor sessions for the Python experiments (single thread,
 * fast): plant -> windower -> features -> ZSM -> controller -> L298N duty. */
#include <string.h>
#include "ms_core.h"
#include "dsp_features.h"
#include "detector.h"

int ms_api_nf(void) { return MS_NF; }
int ms_api_cfg_sizeof(void) { return (int)sizeof(ms_cfg_t); }
void ms_api_cfg_default(ms_cfg_t *c, unsigned long long seed) { ms_cfg_default(c, seed); }
const char *ms_api_feature_name(int i) { return MS_FEATURE_NAMES[i]; }
int ms_api_feature_group(int i) { return MS_FEATURE_GROUP[i]; }

/* per-window outputs (each array has room for max_win entries):
   t, duty, state, sim (-1 learning), sev, rpm, raw[6], feat[18]
   mode: CTRL_NONE (-1) open loop at 100 %, CTRL_THRESHOLD, CTRL_AI
   adapt_alpha > 0: self-optimising baseline */
int ms_api_run(const ms_cfg_t *cfg, int mode, float seconds, float learn_s, float center, float sens,
               float adapt_alpha, int max_win, float *t_o, float *duty_o, int *state_o, int *sim_o,
               float *sev_o, float *rpm_o, float *raw_o, float *feat_o, float *ctrl_o)
{
    static ms_plant_t P;
    static ms_windower_t WD;
    static ms_window_t W;
    ms_ctrl_t C;
    detector_t D;
    fft_init();
    ms_plant_init(&P, cfg);
    memset(&WD, 0, sizeof WD);
    ms_ctrl_init(&C, mode);
    det_init(&D, DET_ZSM, MS_NF, center, sens);
    ms_detector_setup(&D);
    uint64_t n = (uint64_t)(seconds * MS_FS);
    int nw = 0;
    float duty = 1.0f;
    for (uint64_t i = 0; i < n && nw < max_win; i++) {
        ms_tick_t tk;
        ms_plant_step(&P, &tk);
        if (!ms_windower_push(&WD, &tk, duty, &W)) continue;
        float f[MS_NF], raw[6];
        float t = (float)((i + 1) / (double)MS_FS);
        ms_features(&W, cfg->rpm_nom, f);
        ms_raw_values(&W, f, raw);
        int sim = -1;
        if (t <= learn_s) { det_learn(&D, f); ms_ctrl_learn(&C, raw); }
        else {
            if (!D.ready) det_finalize(&D);
            sim = det_similarity(&D, f);
            if (adapt_alpha > 0 && sim >= 95) det_adapt(&D, f, adapt_alpha);
        }
        if (mode != CTRL_NONE && t > learn_s) {
            duty = ms_ctrl_step(&C, t, raw, sim);
            ms_plant_set_duty(&P, duty);
        }
        t_o[nw] = t; duty_o[nw] = duty; state_o[nw] = C.state; sim_o[nw] = sim;
        sev_o[nw] = (float)P.sev; rpm_o[nw] = raw[1] * (W.duty > 0.05f ? W.duty : 0.05f) * cfg->rpm_nom;
        memcpy(&raw_o[nw * 6], raw, sizeof raw);
        memcpy(&feat_o[nw * MS_NF], f, sizeof f);
        nw++;
    }
    ctrl_o[0] = C.t_warn; ctrl_o[1] = C.t_stop; ctrl_o[2] = (float)C.reason;
    ctrl_o[3] = (float)WD.gaps; ctrl_o[4] = (float)WD.received;
    return nw;
}
