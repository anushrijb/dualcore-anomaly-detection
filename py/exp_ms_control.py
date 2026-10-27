"""
Review remark 6: the sample project - DC motor + L298N, ESP32, MPU6050, Hall, DHT22.

Closed loop: the controller sets the L298N duty and the motor really slows down,
which also slows speed-dependent wear. Two controllers:
  threshold : the professor's rules - vibration over threshold -> warning and
              step the speed down 25 % every 10 s, then stop; Hall RPM off by
              > 10 % -> warning; DHT temperature / humidity over limits ->
              warning, then stop; current over limit -> stop.
  ai        : the same hard limits as a safety net, PLUS the multi-input
              Z-score model: similarity < 90 -> warning; < 50 -> step the
              speed down; < 10 at 25 % speed -> stop.
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import RESULTS, FIGS
import cbind
from style import plt, save, SERIES, INK2, NEUTRAL, ANOMALY

SESSION_S, LEARN_S, ONSET_S, RAMP_S = 1800.0, 300.0, 600.0, 900.0
STATE = ["normal", "warning", "derated", "stopped"]


def _cfg(seed, fault):
    rng = np.random.default_rng(seed)
    return cbind.ms_cfg(seed, fault=fault, onset_s=ONSET_S, ramp_s=RAMP_S,
                        rpm_nom=float(rng.uniform(2800, 3200)), a1=float(rng.uniform(0.04, 0.06)),
                        ambient_c=float(rng.uniform(24, 32)), rh_pct=float(rng.uniform(45, 65)),
                        load_amp=float(rng.uniform(0.1, 0.3)), load_period_s=float(rng.uniform(180, 300)))


def _job(a):
    seed, fault, mode, center = a
    r = cbind.ms_run(_cfg(seed, fault), mode, SESSION_S, LEARN_S, center, 3.0)
    after = r["t"] > ONSET_S
    return {"seed": seed, "fault": cbind.MS_FAULTS[fault], "mode": "AI + thresholds" if mode == 1 else "thresholds only",
            "warn_after_onset_s": (r["t_warn"] - ONSET_S) if r["t_warn"] >= 0 else None,
            "warned_before_onset": bool(0 <= r["t_warn"] <= ONSET_S),
            "stop_after_onset_s": (r["t_stop"] - ONSET_S) if r["t_stop"] >= 0 else None,
            "final_severity": float(r["sev"][-1]), "mean_speed_after_onset_pct": 100 * float(r["duty"][after].mean()),
            "reason": r["reason"], "_r": r if seed % 100 == 0 or fault == 2 else None}


def run(center, verbose=True):
    jobs = []
    for fault in range(0, 8):
        for k in range(3):
            seed = 9000 + 100 * fault + k
            for mode in (0, 1):
                jobs.append((seed, fault, mode, center))
    with Pool() as p:
        res = p.map(_job, jobs)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_r"} for r in res])
    df.to_csv(RESULTS / "r6_control_runs.csv", index=False)
    g = df.groupby(["fault", "mode"]).agg(
        warning_min=("warn_after_onset_s", lambda x: np.nanmedian(np.array(x, float)) / 60),
        stopped=("stop_after_onset_s", lambda x: int(pd.notna(x).sum())),
        final_severity=("final_severity", "median"),
        speed_pct=("mean_speed_after_onset_pct", "median"),
        false_warn=("warned_before_onset", "sum")).reset_index()
    g.to_csv(RESULTS / "r6_control_summary.csv", index=False)
    ex = {r["mode"]: r["_r"] for r in res if r["_r"] is not None and r["fault"] == "bearing" and r["seed"] == 9200}
    fig_timeline(ex)
    fig_severity(g)
    if verbose:
        print(g.round(2).to_string(index=False))
    return df, g


def fig_timeline(ex):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True)
    for mode, col in [("thresholds only", NEUTRAL), ("AI + thresholds", SERIES[0])]:
        r = ex[mode]
        m = r["t"] / 60
        axes[0].plot(m, 100 * r["duty"], color=col, lw=2, label=mode)
        axes[1].plot(m, r["raw"][:, 0], color=col, lw=0.8)
        axes[2].plot(m, r["sev"], color=col, lw=2)
    axes[1].axhline(0.075, color=INK2, ls="--", lw=1); axes[1].text(0.3, 0.078, "vibration warning 0.075 g", fontsize=8, color=INK2)
    axes[1].axhline(0.15, color=ANOMALY, ls="--", lw=1); axes[1].text(0.3, 0.153, "vibration stop 0.15 g", fontsize=8, color=ANOMALY)
    for a in axes:
        a.axvline(ONSET_S / 60, color=ANOMALY, lw=1)
    axes[0].set_ylabel("Motor speed (% duty)"); axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_ylabel("Radial vibration RMS (g)")
    axes[2].set_ylabel("Bearing damage (severity)"); axes[2].set_xlabel("Time (min); bearing fault starts at 10 min")
    axes[0].set_title("Bearing fault: AI steps the speed down earlier, so the bearing wears more slowly")
    save(fig, FIGS / "r6_control_timeline.png")


def fig_severity(g):
    faults = ["imbalance", "bearing", "misalignment", "looseness", "overload"]   # speed-dependent wear
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(faults))
    for k, (mode, col) in enumerate([("thresholds only", NEUTRAL), ("AI + thresholds", SERIES[0])]):
        v = [g[(g.fault == f) & (g["mode"] == mode)].final_severity.iloc[0] for f in faults]
        b = ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=col, label=mode)
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, faults); ax.set_ylabel("Damage at end of run (severity 0-1)")
    ax.set_ylim(0, 1.1); ax.legend(fontsize=8)
    ax.set_title("Mechanical faults: wear at 30 min with each controller (lower is better)")
    save(fig, FIGS / "r6_control_severity.png")


if __name__ == "__main__":
    p = json.loads((RESULTS / "r1_params.json").read_text())["params"]["all 6 sensors (Z-score)"]
    run(p["center"])
