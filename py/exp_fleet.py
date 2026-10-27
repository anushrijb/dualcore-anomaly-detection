"""
Objective 3 (fog/cloud): OpenMP on the huge fleet dataset.

Runs build/fleet_omp once (it sweeps internally) and turns its CSV into
  * thread scaling: speedup, efficiency, Karp-Flatt serial fraction per stage
  * schedule (static / dynamic / guided) x chunk size for the irregular S5
    stage and the uniform S2 stage
  * DLP: scalar vs SIMD scoring kernel (GFLOP/s)
  * ILP: 1 / 2 / 4 / 8 independent accumulators (and SIMD) on a dot product
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

from config import BUILD, RESULTS, FIGS, FLEET_MOTORS, FLEET_WINDOWS
from style import plt, save, SERIES, INK2, NEUTRAL

STAGE_NAMES = {"S1_gen": "S1 generate raw archive", "S2_feat": "S2 features (FFT)",
               "S4_score_simd": "S4 Mahalanobis scoring", "S5_deep": "S5 deep analysis (dynamic,16)",
               "S5_deep_static": "S5 deep analysis (static)", "S3_train": "S3 per-motor training"}


def run(verbose=True, threads=None, reps=3):
    csv, js = RESULTS / "o3_fleet_timings.csv", RESULTS / "o3_fleet_summary.json"
    cmd = [str(BUILD / "fleet_omp"), f"--motors={FLEET_MOTORS}", f"--windows={FLEET_WINDOWS}",
           f"--reps={reps}", f"--out={csv}", f"--summary={js}"]
    if threads:
        cmd.append("--threads=" + ",".join(map(str, threads)))
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if verbose:
        print(" ", r.stderr.strip().replace("\n", "\n  "))
    df = pd.read_csv(csv)
    summary = json.loads(js.read_text())

    sc = df[df.experiment == "scaling"].copy()
    base = sc[sc.threads == sc.threads.min()].set_index("stage")["seconds"]
    sc["speedup"] = sc.apply(lambda r: base[r.stage] / r.seconds, axis=1)
    sc["efficiency"] = sc.speedup / sc.threads
    sc["karp_flatt"] = sc.apply(lambda r: (1 / r.speedup - 1 / r.threads) / (1 - 1 / r.threads)
                                if r.threads > 1 else np.nan, axis=1)
    sc.to_csv(RESULTS / "o3_fleet_scaling.csv", index=False)

    fig_scaling(sc)
    fig_schedule(df[df.experiment == "schedule"])
    fig_dlp_ilp(df)
    if verbose:
        print(sc[["stage", "threads", "seconds", "speedup", "efficiency", "karp_flatt"]].round(3).to_string(index=False))
    return df, sc, summary


def fig_scaling(sc):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    T = sorted(sc.threads.unique())
    ax.plot(T, T, color=INK2, ls="--", lw=1, label="ideal (linear)")
    for (stage, col) in [("S1_gen", SERIES[2]), ("S2_feat", SERIES[0]), ("S4_score_simd", SERIES[6]),
                         ("S5_deep", SERIES[1]), ("S5_deep_static", NEUTRAL)]:
        r = sc[sc.stage == stage].sort_values("threads")
        ax.plot(r.threads, r.speedup, "o-", color=col, label=STAGE_NAMES[stage])
    ax.set_xlabel("OpenMP threads"); ax.set_ylabel("Speedup over 1 thread")
    ax.set_xticks(T)
    ax.set_title(f"Thread scaling on {max(T)} hardware threads")
    ax.legend(fontsize=8)
    save(fig, FIGS / "o3_fleet_scaling.png")


def fig_schedule(sch):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    chunks = sorted(sch.chunk.unique())
    labels = ["default" if c == 0 else str(c) for c in chunks]
    for ax, stage, title in [(axes[0], "S5_deep", "Irregular work (S5): dynamic/guided beat static"),
                             (axes[1], "S2_feat", "Uniform work (S2): schedule hardly matters")]:
        for kind, col in [("static", NEUTRAL), ("dynamic", SERIES[0]), ("guided", SERIES[2])]:
            r = sch[(sch.stage == stage) & (sch.schedule == kind)].set_index("chunk").reindex(chunks)
            ax.plot(range(len(chunks)), r.seconds, "o-", color=col, label=kind)
        ax.set_xticks(range(len(chunks)), labels)
        ax.set_xlabel("Chunk size (iterations)"); ax.set_ylabel("Time (s)")
        ax.set_ylim(0, None)
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    T = int(sch.threads.iloc[0])
    fig.suptitle(f"OpenMP schedule and chunk size at {T} threads", fontweight="bold")
    save(fig, FIGS / "o3_fleet_schedule.png")


def fig_dlp_ilp(df):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    d = df[df.experiment == "dlp"]
    Ts = sorted(d.threads.unique())
    x = np.arange(len(Ts))
    for k, (var, col) in enumerate([("scalar", NEUTRAL), ("simd", SERIES[0])]):
        v = [d[(d.threads == t) & (d.variant == var)].gflops.iloc[0] for t in Ts]
        bars = a.bar(x + (k - 0.5) * 0.36, v, 0.34, color=col, label=var.upper() if var == "simd" else "scalar")
        a.bar_label(bars, fmt="%.1f", fontsize=8)
    a.set_xticks(x, [f"{t} thread{'s' if t > 1 else ''}" for t in Ts])
    a.set_ylabel("GFLOP/s"); a.set_title("DLP: SIMD scoring kernel (S4)")
    a.legend(fontsize=8)
    i = df[df.experiment == "ilp"]
    order = ["acc1", "acc2", "acc4", "acc8", "simd"]
    v = [i[i.variant == o].gflops.iloc[0] for o in order]
    bars = b.bar(range(5), v, 0.6, color=[NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL, SERIES[0]])
    b.bar_label(bars, fmt="%.1f", fontsize=8)
    b.set_xticks(range(5), ["1 acc", "2 acc", "4 acc", "8 acc", "SIMD"])
    b.set_ylabel("GFLOP/s (1 thread)"); b.set_title("ILP: independent accumulators hide FP-add latency")
    save(fig, FIGS / "o3_dlp_ilp.png")


if __name__ == "__main__":
    run()
