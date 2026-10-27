"""
Review remark 2: zero loss of samples - and how to measure it.

Method: every 1 ms tick carries a sequence number from Core 0. Losses are
counted separately at each place a sample can disappear:
    L1 sensor FIFO overflow (Core 0 > 85 samples late), L2 Core 0 -> Core 1 queue full,
    L3 gaps in the sequence seen by Core 1 (must equal L1 + L2),
    L4 Wi-Fi packets never delivered, L5 samples whose window never reached the fog.
Zero observed losses in N samples bounds the true loss rate below 3/N with
95 % confidence (the "rule of three").

Runs (build/edge_ms):
  A  real time, dual core, closed loop, healthy motor          -> timing losses
  B  real time, heavy model (90 ms/window): single vs dual core -> where losses appear
  C  accelerated 6 h soak with 2 % packet loss + Wi-Fi outages: store-and-forward on/off
"""
import json
import subprocess

import pandas as pd

from config import BUILD, RESULTS

RT_SECONDS = 240
SOAK_S = 21600
OUTAGES = "3600:60,9000:300,16000:120"


def edge(args):
    out = RESULTS / "_zl.json"
    subprocess.run([str(BUILD / "edge_ms"), *args, f"--stats={out}"], check=True)
    d = json.loads(out.read_text())
    out.unlink()
    return d


def run(center, verbose=True, rt_seconds=RT_SECONDS, soak_s=SOAK_S):
    runs = []
    a = edge([f"--seconds={rt_seconds}", "--learn-s=120", f"--center={center}", "--ctrl=ai"])
    runs.append({"run": f"A: real time {rt_seconds} s, dual core, closed loop", **a})
    for mode in ("single", "dual"):
        d = edge(["--seconds=30", "--learn-s=10", "--ctrl=none", "--proc-us=90000", f"--mode={mode}"])
        runs.append({"run": f"B: real time 30 s, 90 ms model, {mode} core", **d})
    for sf in (0, 1):
        d = edge(["--pace=fast", f"--seconds={soak_s}", "--learn-s=300", "--ctrl=none", "--wifi-loss=0.02",
                  f"--outages={OUTAGES}", f"--store-forward={sf}"])
        runs.append({"run": f"C: {soak_s / 3600:.0f} h soak, 2 % loss + 3 outages, store-and-forward {'on' if sf else 'off'}", **d})
    df = pd.DataFrame(runs)
    df["fog_coverage_pct"] = 100 * df.L5_fog_samples_covered / df.samples_expected_at_fog.clip(lower=1)
    df["L3_equals_L1_plus_L2"] = df.L3_seq_gaps_seen_core1 == df.L1_sensor_fifo_lost + df.L2_queue_lost
    df.to_csv(RESULTS / "r2_zero_loss.csv", index=False)
    if verbose:
        cols = ["run", "samples_produced", "L1_sensor_fifo_lost", "L2_queue_lost", "L3_seq_gaps_seen_core1",
                "L4_uplink_lost_packets", "fog_missing_windows", "fog_coverage_pct", "loss_rate_upper95_if_zero",
                "lateness_us_p99", "L3_equals_L1_plus_L2"]
        print(df[cols].to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    p = json.loads((RESULTS / "r1_params.json").read_text())["params"]["all 6 sensors (Z-score)"]
    run(p["center"])
