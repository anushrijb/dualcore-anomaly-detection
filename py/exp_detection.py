"""
Objective 1 (functionalities) + Objective 4 (detection metrics).

  * builds calibration and test recording sessions (simulated MPU6050 + DS18B20)
  * calibrates each detector's threshold on the CALIBRATION sessions
    (highest balanced accuracy with window specificity >= 99 %)
  * reports on separate TEST sessions:
      window metrics (balanced accuracy, recall, specificity, precision, F1, MCC),
      false-alarm events per hour, detection delay after fault onset,
      and the static 50 degC alarm for reference
"""
import json

import numpy as np
import pandas as pd

from config import RESULTS, FIGS, ONSET_S, FAULT_NAMES, STATIC_ALARM_C, FS_HZ, SIM_THRESHOLD
import cbind
import methods
import sessions
from style import plt, save, SERIES, INK2, ANOMALY, NEUTRAL


def run(verbose=True):
    cal, test = sessions.build("calib"), sessions.build("test")
    params, rows, delays = {}, [], []
    dist_test = {}
    for m, label in methods.METHODS.items():
        c, cal_metrics = methods.calibrate(methods.distances(cal, m), methods.SENS[m])
        params[m] = {"center": c, "sens": methods.SENS[m],
                     "calib_balanced_accuracy": cal_metrics["Balanced accuracy (%)"]}
        dl = methods.distances(test, m)
        dist_test[m] = dl
        met = methods.window_metrics(dl, c, methods.SENS[m])
        fa, healthy_s = 0, 0.0
        for s, (t, d, sev) in zip(test, dl):
            f, h, delay = methods.session_events(t, d, sev, c, methods.SENS[m])
            fa += f; healthy_s += h
            if s.fault:
                delays.append({"session": s.cfg.seed, "fault": FAULT_NAMES[s.fault], "method": m,
                               "delay_s": delay})
        met["False alarms per hour"] = fa / (healthy_s / 3600.0)
        rows.append({"method": label, "key": m, **met})
    for s in test:
        if s.fault:
            delays.append({"session": s.cfg.seed, "fault": FAULT_NAMES[s.fault], "method": "static_50C",
                           "delay_s": methods.static_alarm_delay(s)})
    met = pd.DataFrame(rows)
    dly = pd.DataFrame(delays)
    met.to_csv(RESULTS / "o4_detection_metrics.csv", index=False)
    dly.to_csv(RESULTS / "o4_detection_delays.csv", index=False)

    summ = (dly.groupby(["fault", "method"])["delay_s"]
              .agg(median="median", min="min", max="max",
                   missed=lambda x: int(x.isna().sum()), n="size").reset_index())
    summ.to_csv(RESULTS / "o4_detection_delay_summary.csv", index=False)
    overall = dly.groupby("method")["delay_s"].agg(median="median",
                                                    missed=lambda x: int(x.isna().sum()), n="size")
    (RESULTS / "o4_detection_params.json").write_text(json.dumps(
        {"params": params, "test_sessions": len(test), "calib_sessions": len(cal),
         "overall_delay": overall.reset_index().to_dict("records")}, indent=2, default=float))

    fig_delay(summ)
    fig_timeline(test, params)
    fig_spectra()
    if verbose:
        print(met[["method", "Balanced accuracy (%)", "Recall (%)", "Specificity (%)", "F1 (%)",
                   "MCC (%)", "False alarms per hour"]].round(2).to_string(index=False))
        print(summ.round(1).to_string(index=False))
    return met, dly, params


def fig_delay(summ):
    faults = FAULT_NAMES[1:]
    order = ["static_50C", "base_temp_zsm", "vib_zsm", "vib_maha"]
    names = {"static_50C": "Static 50 \u00b0C alarm", "base_temp_zsm": "Base paper (temperature ZSM)",
             "vib_zsm": "Vibration ZSM (ablation)", "vib_maha": "Proposed (vibration Mahalanobis)"}
    cols = {"static_50C": "#d8d6cf", "base_temp_zsm": NEUTRAL, "vib_zsm": "#9cc2ee", "vib_maha": SERIES[0]}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.2
    for j, m in enumerate(order):
        for i, f in enumerate(faults):
            r = summ[(summ.fault == f) & (summ.method == m)]
            x = i + (j - 1.5) * w
            if r.empty or r["missed"].iloc[0] == r["n"].iloc[0]:
                ax.text(x, 0.15, "\u00d7", ha="center", va="bottom", color=INK2, fontsize=11)
                continue
            med, lo, hi = r["median"].iloc[0] / 60, r["min"].iloc[0] / 60, r["max"].iloc[0] / 60
            ax.bar(x, med, w * 0.9, color=cols[m], edgecolor="white")
            ax.plot([x, x], [lo, hi], color=INK2, lw=1)
            if r["missed"].iloc[0]:
                ax.text(x, hi + 0.15, f"{int(r['missed'].iloc[0])}/{int(r['n'].iloc[0])}\nmissed",
                        ha="center", fontsize=7, color=INK2)
    ax.set_xticks(range(len(faults)), faults)
    ax.set_ylabel("Detection delay after onset (min)")
    ax.set_ylim(0, 11)
    ax.set_title("Vibration catches mechanical faults 4-7 min before temperature; cooling faults tie")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols[m], label=names[m]) for m in order], ncol=2, fontsize=8, loc="upper left")
    ax.text(len(faults) - 0.5, 10.6, "\u00d7 = no alarm within 10 min \u00b7 bar = median, line = range of 5 motors",
            ha="right", fontsize=7.5, color=INK2)
    save(fig, FIGS / "o4_detection_delay.png")


def fig_timeline(test, params):
    s = next(x for x in test if x.fault == 3)              # a bearing-fault session
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
    t1 = np.arange(len(s.temp))
    a0.plot(t1 / 60, s.temp, color=SERIES[1], lw=1.5, label="Case temperature (DS18B20)")
    a0.axhline(STATIC_ALARM_C, color=INK2, ls="--", lw=1)
    a0.text(24.8, STATIC_ALARM_C - 0.7, "static alarm 50 °C", fontsize=8, color=INK2, ha="right")
    a0.set_ylabel("°C")
    a0.set_ylim(min(s.temp.min() - 1, 36), 52)
    for m, col in [("base_temp_zsm", "#9a9890"), ("vib_maha", SERIES[0])]:
        t, d, sev = methods.distances([s], m)[0]
        sim = methods.similarity(d, params[m]["center"], params[m]["sens"])
        a1.plot(t / 60, sim, color=col, lw=1.2 if m == "vib_maha" else 1.6,
                label=methods.METHODS[m])
        _, _, delay = methods.session_events(t, d, sev, params[m]["center"], params[m]["sens"])
        if delay is not None:
            for ax in (a0, a1):
                ax.axvline((ONSET_S + delay) / 60, color=col, lw=1.3, ls=":")
    for ax in (a0, a1):
        ax.axvspan(0, 8, color="#eda100", alpha=0.12, lw=0)
        ax.axvline(ONSET_S / 60, color=ANOMALY, lw=1)
    a0.text(0.2, a0.get_ylim()[1] - 1, "on-device learning", fontsize=8, color=INK2, va="top")
    a0.text(ONSET_S / 60 + 0.2, a0.get_ylim()[1] - 1, "bearing fault starts", fontsize=8, color=ANOMALY, va="top")
    a1.axhline(SIM_THRESHOLD, color=INK2, ls="--", lw=1)
    a1.set_ylabel("Similarity (%)"); a1.set_ylim(-3, 105); a1.set_xlabel("Time (min)")
    a0.legend(loc="lower left", fontsize=8); a1.legend(loc="lower left", fontsize=8)
    a0.set_title("Bearing fault: vibration alarm (dotted blue) long before temperature (dotted grey)")
    save(fig, FIGS / "o1_bearing_timeline.png")


def fig_spectra():
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True, sharey=True)
    n = 4096
    for k, ax in enumerate(axes.flat):
        c = cbind.motor(4242, fault=k, onset_s=0.0, ramp_s=1.0)
        raw = cbind.raw_samples(c, int(300 * FS_HZ), n)
        ax_g = raw[:, 0] / 16384.0
        spec = np.abs(np.fft.rfft((ax_g - ax_g.mean()) * np.hanning(n))) / n
        f = np.fft.rfftfreq(n, 1 / FS_HZ)
        ax.semilogy(f, spec + 1e-7, color=SERIES[0] if k else NEUTRAL, lw=0.8)
        ax.set_title(FAULT_NAMES[k], fontsize=10)
        ax.set_ylim(1e-6, 1e-1)
    for ax in axes[1]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("|X(f)| radial accel (g)")
    fig.suptitle("Each fault leaves its own vibration signature (MPU6050 radial axis, full severity)",
                 fontweight="bold")
    save(fig, FIGS / "o1_fault_spectra.png")


if __name__ == "__main__":
    run()
