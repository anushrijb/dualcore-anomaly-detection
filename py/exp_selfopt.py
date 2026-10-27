"""
Review remark 3: AI and GenAI for predictive maintenance -> self-optimising system.

1. Self-optimising baseline: the healthy reference follows slow, harmless
   changes (ambient temperature +10 C, humidity +20 % over 2 h) by updating the
   Z-score model with windows it is confident are normal (similarity >= 95 %),
   so it does not raise false alarms - but stops updating as soon as a fault
   pulls similarity down, so faults are still caught.
2. Self-optimising operation: the controller derates speed according to how
   abnormal the motor is (see exp_ms_control).
3. GenAI maintenance report: for each alarm, the system assembles the evidence
   (which sensors deviate, by how much, fault-type signature match, actions
   taken) and writes a work order. Offline it uses a template; with
   ANTHROPIC_API_KEY set it asks an LLM to write the report from the same
   evidence (prompt saved in results/genai_prompts.jsonl).
"""
import json
import os
import urllib.request

import numpy as np
import pandas as pd

from config import RESULTS, FIGS
import cbind
from style import plt, save, SERIES, NEUTRAL, ANOMALY, INK2

LEARN_S = 300.0
DRIFT = dict(drift_c=10.0, drift_rh=20.0, drift_s=7200.0)


def sustained_first(t, sim, t0, win=469):
    a = (sim >= 0) & (sim < 90)
    run = np.convolve(a.astype(int), np.ones(3, int), "full")[:len(a)] >= 3
    frac = np.convolve(a.astype(float), np.ones(win) / win, "full")[win - 1:win - 1 + len(a)]
    idx = np.where(run & (t > t0) & (frac >= 0.5))[0]
    return float(t[idx[0]] - t0) if len(idx) else None


def drift_experiment(center, alpha=0.128 / 600):   # time constant 10 min
    rows, ex = [], {}
    for seed in (31, 32, 33):
        for fault in (0, 2):
            for a in (0.0, alpha):
                cfg = cbind.ms_cfg(seed, fault=fault, onset_s=3600.0 if fault else 1e9, ramp_s=900.0, **DRIFT)
                r = cbind.ms_run(cfg, -1, 7200.0, LEARN_S, center, 3.0, adapt=a)
                t, sim = r["t"], r["sim"]
                healthy = (t > LEARN_S) & (r["sev"] == 0)
                an = (sim >= 0) & (sim < 90)
                alarm = np.convolve(an.astype(int), np.ones(3, int), "full")[:len(an)] >= 3
                rise = alarm & ~np.r_[False, alarm[:-1]]
                hours = healthy.sum() * 0.128 / 3600
                rows.append({"seed": seed, "fault": cbind.MS_FAULTS[fault],
                             "baseline": "self-optimising" if a else "static",
                             "false_alarms_per_hour": float((rise & healthy).sum() / hours),
                             "anomalous_healthy_windows_pct": float(100 * an[healthy].mean()),
                             "detection_delay_s": sustained_first(t, sim, 3600.0) if fault else None})
                if seed == 31:
                    ex[(fault, a > 0)] = r
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "r3_selfopt_drift.csv", index=False)
    fig_drift(ex)
    return df


def fig_drift(ex):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    r = ex[(0, False)]
    axes[0].plot(r["t"] / 60, r["X"][:, 10], color=SERIES[1], lw=1.5, label="ambient temperature (DHT22, °C)")
    ax2 = axes[0].twinx()
    ax2.plot(r["t"] / 60, r["X"][:, 11], color=SERIES[6], lw=1.2, label="humidity (%)")
    ax2.set_ylabel("RH (%)")
    axes[0].set_ylabel("°C"); axes[0].legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
    for adapt, col, lab in [(False, NEUTRAL, "static baseline"), (True, SERIES[0], "self-optimising baseline")]:
        rr = ex[(0, adapt)]
        m = rr["sim"] >= 0
        axes[1].plot(rr["t"][m] / 60, rr["sim"][m], color=col, lw=1, label=lab)
    axes[1].axhline(90, color=INK2, ls="--", lw=1)
    axes[1].set_ylabel("Similarity (%)"); axes[1].set_xlabel("Time (min), healthy motor")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[0].set_title("Healthy ambient drift: the static model raises false alarms, the self-optimising one adapts")
    save(fig, FIGS / "r3_selfopt_drift.png")


# ------------------------------------------------------------------ GenAI --
GROUP_NAME = {1: "vibration (MPU6050)", 2: "temperature (MPU6050 die / DHT22)", 4: "humidity (DHT22)",
              8: "speed (Hall sensor)", 16: "motor current (ACS712)", 32: "sound (INMP441)"}
ADVICE = {
    "imbalance": "Check rotor/coupling for mass imbalance or build-up; perform field balancing.",
    "bearing": "Inspect the drive-end bearing (outer race) for wear and lubrication; plan replacement.",
    "misalignment": "Check shaft alignment and coupling; re-align with dial gauge or laser.",
    "looseness": "Check mounting bolts, base and coupling for mechanical looseness; re-torque.",
    "overload": "Check the driven load for jamming or excess friction; verify current rating.",
    "cooling": "Check the cooling fan, vents and ambient airflow; clean dust from the housing.",
    "humidity": "Check the enclosure for water ingress or condensation; dry and reseal.",
}


def evidence(r, names, groups, sig):
    learn = r["t"] <= LEARN_S
    mu, sd = r["X"][learn].mean(0), r["X"][learn].std(0) + 1e-6
    i_warn = int(np.searchsorted(r["t"], r["t_warn"])) if r["t_warn"] >= 0 else None
    if i_warn is None:
        return None
    # evidence: 30-90 s after the warning (or up to the stop), when the pattern is clear
    i_end = int(np.searchsorted(r["t"], r["t_stop"])) if r["t_stop"] >= 0 else len(r["t"])
    a, b = i_warn + 234, min(i_warn + 703, i_end)
    seg = slice(a, b) if b > a + 5 else slice(i_warn, max(i_end, i_warn + 1))
    z = ((r["X"][seg] - mu) / sd).mean(0)
    top = np.argsort(-np.abs(z))[:4]
    zn = z / (np.linalg.norm(z) + 1e-9)
    scores = sig @ zn
    best = int(np.argmax(scores)) + 1
    steps = []
    for k in range(1, len(r["duty"])):
        if r["duty"][k] != r["duty"][k - 1]:
            steps.append({"t_min": round(float(r["t"][k]) / 60, 2), "speed_pct": int(round(100 * r["duty"][k]))})
    gset = sorted({GROUP_NAME[int(groups[i])] for i in top})
    return {"warning_time_min": round(r["t_warn"] / 60, 2),
            "similarity_at_warning": int(r["sim"][i_warn]),
            "top_deviations": [{"feature": names[i], "z": round(float(z[i]), 1)} for i in top],
            "sensors_involved": gset,
            "likely_fault": cbind.MS_FAULTS[best], "signature_match": round(float(scores[best - 1]), 2),
            "actions": steps, "stopped_at_min": round(r["t_stop"] / 60, 2) if r["t_stop"] >= 0 else None}


def template_report(motor, ev):
    lines = [f"### Work order - {motor}",
             f"- **Alarm**: similarity fell to {ev['similarity_at_warning']} % at {ev['warning_time_min']} min.",
             f"- **Evidence**: " + "; ".join(f"{d['feature']} {d['z']:+.1f} sigma" for d in ev["top_deviations"]),
             f"- **Sensors involved**: {', '.join(ev['sensors_involved'])}.",
             f"- **Likely cause**: {ev['likely_fault']} (signature match {ev['signature_match']:.2f}).",
             "- **Automatic actions**: " + (", ".join(f"{a['speed_pct']} % at {a['t_min']} min" for a in ev["actions"])
                                           if ev["actions"] else "none") +
             (f"; motor stopped at {ev['stopped_at_min']} min." if ev["stopped_at_min"] else "."),
             f"- **Recommended**: {ADVICE[ev['likely_fault']]}"]
    return "\n".join(lines)


def llm_report(motor, ev):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    prompt = ("You are a maintenance planner. Write a short work order (max 120 words) for the technician "
              f"for motor {motor} from this evidence produced by an edge anomaly detector. Do not invent "
              "measurements that are not in the evidence.\n" + json.dumps(ev))
    body = json.dumps({"model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as f:
            return json.loads(f.read())["content"][0]["text"]
    except Exception as e:                                # offline / no access: fall back
        return f"(LLM call failed: {e})"


def genai_reports(center):
    names, groups = cbind.ms_feature_names(), cbind.ms_feature_groups()
    sig = np.load(RESULTS / "r1_signatures.npy")
    out, prompts, ok = ["# Generated maintenance reports", "",
                        "One AI-controlled run per fault type. Evidence is computed on the device; the text is "
                        "written from it (template offline, LLM when ANTHROPIC_API_KEY is set).", ""], [], 0
    for f in range(1, 8):
        cfg = cbind.ms_cfg(9500 + f, fault=f, onset_s=600.0, ramp_s=900.0)
        r = cbind.ms_run(cfg, 1, 1800.0, LEARN_S, center, 3.0)
        ev = evidence(r, names, groups, sig)
        motor = f"MOTOR-{f:02d}"
        if ev is None:
            out += [f"### {motor}: no alarm", ""]
            continue
        ok += ev["likely_fault"] == cbind.MS_FAULTS[f]
        prompts.append({"motor": motor, "true_fault": cbind.MS_FAULTS[f], "evidence": ev})
        out += [template_report(motor, ev), ""]
        llm = llm_report(motor, ev)
        if llm:
            out += ["*LLM version:*", "", llm, ""]
    (RESULTS / "r3_maintenance_reports.md").write_text("\n".join(out))
    with open(RESULTS / "genai_prompts.jsonl", "w") as fh:
        for p in prompts:
            fh.write(json.dumps(p) + "\n")
    return {"reports": len(prompts), "cause_correct": ok}


def run(center, verbose=True):
    df = drift_experiment(center)
    g = genai_reports(center)
    summ = df.groupby(["fault", "baseline"]).agg(
        fa_per_h=("false_alarms_per_hour", "mean"), anomalous_pct=("anomalous_healthy_windows_pct", "mean"),
        delay_s=("detection_delay_s", lambda x: np.nanmedian(np.array(x, float)) if x.notna().any() else np.nan)).reset_index()
    summ.to_csv(RESULTS / "r3_selfopt_summary.csv", index=False)
    if verbose:
        print(summ.round(2).to_string(index=False))
        print("GenAI:", g)
    return summ, g


if __name__ == "__main__":
    p = json.loads((RESULTS / "r1_params.json").read_text())["params"]["all 6 sensors (Z-score)"]
    run(p["center"])
