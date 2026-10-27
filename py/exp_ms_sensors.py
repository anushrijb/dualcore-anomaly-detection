"""
Review remark 1: one-class Z-score model with MORE THAN ONE INPUT
(temperature, vibration, humidity, RPM, sound, current).

Sample-project motor (ESP32 + MPU6050 + Hall + DHT22 + ACS712 + INMP441),
open loop at 100 % speed. 7 fault types. For every sensor set we learn a
Z-score model on the first 300 s (healthy), choose its threshold on
calibration sessions, and report on separate test sessions:
    detection delay per fault, missed faults, balanced accuracy, false alarms.
Also trains a nearest-signature fault classifier (used by the GenAI report).
"""
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from config import RESULTS, FIGS, SEED, SIM_THRESHOLD, DEBOUNCE
import cbind
import methods
from style import plt, save, INK2

SESSION_S, LEARN_S, ONSET_S, RAMP_S = 1200.0, 300.0, 600.0, 900.0
LABEL_SEV = 0.2
GROUPS = cbind.MS_GROUPS
SETS = {  # name -> group mask
    "temperature": GROUPS["temperature"], "humidity": GROUPS["humidity"], "rpm": GROUPS["rpm"],
    "current": GROUPS["current"], "sound": GROUPS["sound"], "vibration": GROUPS["vibration"],
    "all 6 sensors (Z-score)": 63,
}


def _cfg(seed, fault, rng):
    return cbind.ms_cfg(seed, fault=fault, onset_s=ONSET_S, ramp_s=RAMP_S,
                        rpm_nom=float(rng.uniform(2800, 3200)), a1=float(rng.uniform(0.04, 0.06)),
                        ambient_c=float(rng.uniform(24, 32)), rh_pct=float(rng.uniform(45, 65)),
                        load_amp=float(rng.uniform(0.1, 0.3)), load_period_s=float(rng.uniform(180, 300)))


def _run(args):
    seed, fault = args
    rng = np.random.default_rng(seed)
    r = cbind.ms_run(_cfg(seed, fault, rng), -1, SESSION_S, LEARN_S)
    return {"seed": seed, "fault": fault, "t": r["t"], "sev": r["sev"], "X": r["X"]}


def build(kind):
    per, normals, base = (2, 3, 6000) if kind == "calib" else (4, 5, 8000)
    jobs, s = [], base
    for _ in range(normals):
        jobs.append((s, 0)); s += 1
    for f in range(1, 8):
        for _ in range(per):
            jobs.append((s, f)); s += 1
    with Pool() as p:
        return p.map(_run, jobs)


def zsm_dist(sess, cols, kind=0):
    out = []
    for s in sess:
        X = s["X"][:, cols]
        learn = s["t"] <= LEARN_S
        fl = {j: cbind.MS_FLOORS[c] for j, c in enumerate(cols) if c in cbind.MS_FLOORS}
        det = cbind.Detector(kind, len(cols), 0.0, 1.0, floors=fl if kind == 0 else None)
        det.learn(X[learn])
        d, _ = det.score(X)
        keep = ~learn
        out.append((s["t"][keep], d[keep], s["sev"][keep]))
    return out


def run(verbose=True):
    groups = cbind.ms_feature_groups()
    names = cbind.ms_feature_names()
    cal, test = build("calib"), build("test")
    sets = dict(SETS)
    sets["all 6 sensors (Mahalanobis)"] = -63
    rows, delays, params = [], [], {}
    for name, mask in sets.items():
        kind = 1 if mask < 0 else 0
        cols = np.where(groups & abs(mask))[0]
        c, _ = methods.calibrate(zsm_dist(cal, cols, kind), 3.0, min_spec=99.9)
        params[name] = {"center": c, "cols": cols.tolist(), "kind": kind}
        dl = zsm_dist(test, cols, kind)
        tp = fp = tn = fn = fa = 0
        healthy_s = 0.0
        for s, (t, d, sev) in zip(test, dl):
            a = methods.similarity(d, c, 3.0) < SIM_THRESHOLD
            nom, ano = sev == 0, sev >= LABEL_SEV
            tp += int((a & ano).sum()); fn += int((~a & ano).sum())
            fp += int((a & nom).sum()); tn += int((~a & nom).sum())
            alarm = methods.debounce(a, DEBOUNCE)
            rise = alarm & ~np.r_[False, alarm[:-1]]
            fa += int((rise & (t <= ONSET_S)).sum())
            healthy_s += float((t <= ONSET_S).sum()) * 0.128
            if s["fault"]:
                # sustained detection: debounced alarm AND >= 50 % anomalous windows in the next 60 s
                frac = np.convolve(a.astype(float), np.ones(469) / 469, "full")[468:468 + len(a)]
                idx = np.where(alarm & (t > ONSET_S) & (frac >= 0.5))[0]
                delays.append({"set": name, "fault": cbind.MS_FAULTS[s["fault"]],
                               "delay_s": float(t[idx[0]] - ONSET_S) if len(idx) else None})
        rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        rows.append({"sensor set": name, "features": len(cols), "balanced accuracy (%)": 100 * (rec + spec) / 2,
                     "recall (%)": 100 * rec, "specificity (%)": 100 * spec,
                     "false alarms per hour": fa / (healthy_s / 3600)})
    met = pd.DataFrame(rows)
    dly = pd.DataFrame(delays)
    summ = dly.groupby(["set", "fault"])["delay_s"].agg(
        median="median", missed=lambda x: int(x.isna().sum()), n="size").reset_index()
    det_rate = dly.groupby("set")["delay_s"].apply(lambda x: 100 * x.notna().mean()).rename("faults detected (%)")
    met = met.merge(det_rate, left_on="sensor set", right_index=True)
    met.to_csv(RESULTS / "r1_sensor_sets.csv", index=False)
    summ.to_csv(RESULTS / "r1_sensor_delays.csv", index=False)

    diag = diagnosis(cal, test, params["all 6 sensors (Z-score)"])
    (RESULTS / "r1_params.json").write_text(json.dumps({"params": params, "feature_names": names,
                                                         "diagnosis": diag}, indent=2, default=float))
    fig_heatmap(summ)
    if verbose:
        print(met.round(2).to_string(index=False))
        print("diagnosis accuracy:", diag["accuracy"])
    return met, summ, params, diag


def zvec(s, cols):
    """mean z-score pattern of a session's clearly faulty windows vs its own healthy learning"""
    X = s["X"][:, cols]
    learn = s["t"] <= LEARN_S
    mu, sd = X[learn].mean(0), X[learn].std(0) + 1e-6
    return (X - mu) / sd


def diagnosis(cal, test, p):
    """Nearest-signature fault classifier: centroid of normalised z-patterns per fault type
    (trained on calibration motors, tested on test motors, windows with severity >= 0.3)."""
    cols = np.array(p["cols"])
    def pats(sess):
        P, y = [], []
        for s in sess:
            if not s["fault"]:
                continue
            Z = zvec(s, cols)[s["sev"] >= 0.3]
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
            P.append(Z); y += [s["fault"]] * len(Z)
        return np.vstack(P), np.array(y)
    Pc, yc = pats(cal)
    cent = {f: Pc[yc == f].mean(0) for f in range(1, 8)}
    Pt, yt = pats(test)
    C = np.stack([cent[f] for f in range(1, 8)])
    pred = 1 + np.argmax(Pt @ C.T, axis=1)
    cm = pd.crosstab(pd.Series(yt, name="true"), pd.Series(pred, name="pred"))
    np.save(RESULTS / "r1_signatures.npy", C)
    return {"accuracy": float((pred == yt).mean()), "n": int(len(yt)),
            "per_fault": {cbind.MS_FAULTS[f]: float((pred[yt == f] == f).mean()) for f in range(1, 8)}}


def fig_heatmap(summ):
    sets = list(SETS) + ["all 6 sensors (Mahalanobis)"]
    faults = cbind.MS_FAULTS[1:]
    M = np.full((len(faults), len(sets)), np.nan)
    txt = [["" for _ in sets] for _ in faults]
    for i, f in enumerate(faults):
        for j, s in enumerate(sets):
            r = summ[(summ.set == s) & (summ.fault == f)]
            if r.empty:
                continue
            med, miss, n = r["median"].iloc[0], int(r["missed"].iloc[0]), int(r["n"].iloc[0])
            if miss == n:
                txt[i][j] = "missed"
            else:
                M[i, j] = med / 60
                txt[i][j] = f"{med / 60:.1f}" + (f"\n{miss}/{n} missed" if miss else "")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(np.nan_to_num(M, nan=12), cmap="Blues_r", vmin=0, vmax=12, aspect="auto")
    for i in range(len(faults)):
        for j in range(len(sets)):
            v = M[i, j]
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(v) and v < 4) else "#0b0b0b")
    ax.set_xticks(range(len(sets)), [s.replace(" (", "\n(") for s in sets], fontsize=8)
    ax.set_yticks(range(len(faults)), faults)
    ax.grid(False)
    ax.set_title("Each fault shows up on different sensors; only the multi-input model catches all seven "
                 "(median minutes to alarm)", fontsize=10)
    save(fig, FIGS / "r1_sensor_heatmap.png")


if __name__ == "__main__":
    run()
