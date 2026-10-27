"""
Objective 3 (edge, thread-level parallelism): dual-core pipeline vs single core.

Real-time runs of build/edge_dualcore at 1 kHz with the computed I2C read time
(385 us per 14-byte burst at 400 kHz). The processing time per window is swept
by --proc-us (busy work) to model heavier models or a slower MCU; 0 = host
speed. For each run we record sample lateness (jitter), samples lost from the
MPU6050 FIFO (85 samples deep), end-to-end decision latency and core load.
"""
import json
import subprocess

import pandas as pd

from config import BUILD, RESULTS, FIGS
from style import plt, save, SERIES, INK2, NEUTRAL

PROC_US = [0, 1000, 10000, 40000, 70000, 90000, 110000, 140000]
SECONDS = 6


def run_one(mode, proc_us, seconds=SECONDS, fs=1000):
    out = RESULTS / f"_rt_{mode}_{proc_us}.json"
    subprocess.run([str(BUILD / "edge_dualcore"), f"--mode={mode}", f"--seconds={seconds}",
                    f"--proc-us={proc_us}", f"--fs={fs}", "--learn-s=1", "--quiet",
                    f"--stats={out}"], check=True)
    d = json.loads(out.read_text())
    out.unlink()
    return d


def run(verbose=True):
    rows = []
    for p in PROC_US:
        for mode in ("single", "dual"):
            d = run_one(mode, p)
            d["proc_us_setting"] = p
            rows.append(d)
            if verbose:
                print(f"  {mode:6s} proc={p/1000:6.1f} ms  late p99={d['lateness_us_p99']/1000:7.2f} ms "
                      f"lost={d['samples_lost_fifo'] + d['samples_lost_ring']:5d}  "
                      f"core0={d['core0_util']:.2f} core1={d['core1_util']:.2f}")
    df = pd.DataFrame(rows)
    df["lost_pct"] = 100 * (df.samples_lost_fifo + df.samples_lost_ring) / df.samples
    df["late_over_period_pct"] = 100 * df.samples_late_over_period / df.samples
    df.to_csv(RESULTS / "o3_edge_dualcore.csv", index=False)
    fig(df)
    return df


def fig(df):
    f, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for mode, col, lab in [("single", NEUTRAL, "Single core (acquire + process)"),
                           ("dual", SERIES[0], "Dual core (Core 0 acquire, Core 1 process)")]:
        r = df[df["mode"] == mode].sort_values("proc_us_setting")
        a.plot(r.proc_us_setting / 1000, r.lateness_us_p99 / 1000, "o-", color=col, label=lab)
        b.plot(r.proc_us_setting / 1000, r.e2e_latency_us_p99 / 1000, "o-", color=col, label=lab)
    a.set_yscale("log")
    a.axhline(85, color=INK2, ls="--", lw=1)
    a.text(1, 100, "MPU6050 FIFO full (85 ms at 1 kHz)", fontsize=8, color=INK2)
    a.set_xlabel("Processing time per window (ms)"); a.set_ylabel("Sample read lateness, p99 (ms, log)")
    a.set_title("Sampling stays on time only with a dedicated core")
    b.set_yscale("log")
    b.axvline(128, color=INK2, ls=":", lw=1)
    b.text(126, 2000, "new window every 128 ms", fontsize=8, color=INK2, ha="right")
    b.set_xlabel("Processing time per window (ms)"); b.set_ylabel("Decision latency, p99 (ms, log)")
    b.set_title("Single core falls behind at ~79 ms, dual core at 128 ms")
    a.legend(fontsize=8, loc="upper left")
    save(f, FIGS / "o3_edge_dualcore.png")


if __name__ == "__main__":
    run()
