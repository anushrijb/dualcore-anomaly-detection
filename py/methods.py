"""The three detectors compared in Objective 4.

  base_temp_zsm : base paper - DS18B20 temperature, 16-sample windows (stride 4),
                  30 learning windows, diagonal Z-score (NanoEdge ZSM family)
  vib_zsm       : MPU6050 vibration features + diagonal Z-score  (ablation)
  vib_maha      : MPU6050 vibration features + Mahalanobis       (proposed)

Each returns, per session, the decision times, a distance and the fault
severity at that time. The distance -> similarity map is the same for all:
    sim = round(100 / (1 + exp(sens * (d - center)))),  anomaly if sim < 90
"""
import numpy as np

from config import (BASE_WINDOW, BASE_STRIDE, BASE_LEARN, BASE_LEARN_S, SIM_THRESHOLD,
                    DEBOUNCE, LABEL_ANOMALY_SEVERITY, ONSET_S, STATIC_ALARM_C, N_FEAT)
import cbind

METHODS = {
    "base_temp_zsm": "Base paper: temperature + ZSM",
    "vib_zsm": "Vibration + ZSM (ablation)",
    "vib_maha": "Proposed: vibration + Mahalanobis",
}
SENS = {"base_temp_zsm": 1.5, "vib_zsm": 3.0, "vib_maha": 3.0}


def base_temp_zsm(s, sigma_min=0.25):
    T = s.temp
    learn = np.stack([T[k * BASE_WINDOW:(k + 1) * BASE_WINDOW] for k in range(BASE_LEARN)])
    mu, sd = learn.mean(0), np.maximum(learn.std(0, ddof=1), sigma_min)
    starts = np.arange(BASE_LEARN_S, len(T) - BASE_WINDOW + 1, BASE_STRIDE)
    W = np.stack([T[a:a + BASE_WINDOW] for a in starts])
    d = np.sqrt((((W - mu) / sd) ** 2).mean(1))
    t = starts + BASE_WINDOW
    sev = cbind.severity(s.cfg, len(t), float(t[0]), float(BASE_STRIDE))
    return t.astype(float), d, sev


def vib(s, kind):
    det = cbind.Detector(kind, N_FEAT, 0.0, 1.0)
    det.learn(s.X[s.t_win <= BASE_LEARN_S])
    d, _ = det.score(s.X)
    keep = s.t_win > BASE_LEARN_S
    return s.t_win[keep], d[keep], s.sev_win[keep]


def distances(sessions, method):
    if method == "base_temp_zsm":
        return [base_temp_zsm(s) for s in sessions]
    return [vib(s, 0 if method == "vib_zsm" else 1) for s in sessions]


def similarity(d, center, sens):
    x = np.clip(sens * (d - center), -60, 60)
    return np.floor(100.0 / (1.0 + np.exp(x)) + 0.5)


def debounce(flags, k=DEBOUNCE):
    """alarm[i] = flags[i-k+1..i] all true."""
    f = flags.astype(int)
    c = np.convolve(f, np.ones(k, int), "full")[:len(f)]
    return c >= k


def window_metrics(dist_list, center, sens):
    tp = fp = tn = fn = 0
    for t, d, sev in dist_list:
        a = similarity(d, center, sens) < SIM_THRESHOLD
        nom, ano = sev == 0, sev >= LABEL_ANOMALY_SEVERITY
        tp += int((a & ano).sum()); fn += int((~a & ano).sum())
        fp += int((a & nom).sum()); tn += int((~a & nom).sum())
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"Balanced accuracy (%)": 100 * (rec + spec) / 2, "Recall (%)": 100 * rec,
            "Specificity (%)": 100 * spec, "Precision (%)": 100 * prec, "F1 (%)": 100 * f1,
            "MCC (%)": 100 * mcc, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def calibrate(dist_list, sens, min_spec=99.0):
    """Choose the centre (decision threshold) on CALIBRATION sessions:
    highest balanced accuracy with window specificity >= 99 %."""
    allnom = np.concatenate([d[sev == 0] for _, d, sev in dist_list])
    best = None
    for c in np.quantile(allnom, np.linspace(0.90, 0.99999, 300)) + np.log(9) / sens:
        m = window_metrics(dist_list, c, sens)
        if m["Specificity (%)"] < min_spec:
            continue
        if best is None or m["Balanced accuracy (%)"] > best[1]["Balanced accuracy (%)"]:
            best = (float(c), m)
    return best


def session_events(t, d, sev, center, sens):
    """Debounced alarms: false-alarm events while healthy, first detection after onset."""
    alarm = debounce(similarity(d, center, sens) < SIM_THRESHOLD)
    healthy = sev == 0
    rising = alarm & ~np.r_[False, alarm[:-1]]
    fa_events = int((rising & healthy & (t <= ONSET_S)).sum())
    healthy_s = float((t <= ONSET_S).sum()) * (t[1] - t[0])
    after = np.where(alarm & (t > ONSET_S))[0]
    delay = float(t[after[0]] - ONSET_S) if len(after) else None
    return fa_events, healthy_s, delay


def static_alarm_delay(s):
    idx = np.where((s.temp >= STATIC_ALARM_C) & (np.arange(len(s.temp)) > ONSET_S))[0]
    return float(idx[0] - ONSET_S) if len(idx) else None
