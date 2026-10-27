"""Appends the Review-1 remark results to results/RESULTS.md."""
import json

import pandas as pd

from config import RESULTS


def write():
    L = ["", "## Review 1 remarks", ""]
    m = pd.read_csv(RESULTS / "r1_sensor_sets.csv")
    L += ["### Multi-input one-class Z-score model (sensor ablation, 7 faults, test motors)", "",
          m.round(2).to_markdown(index=False), ""]
    s = pd.read_csv(RESULTS / "r1_sensor_delays.csv")
    s["median_min"] = (s["median"] / 60).round(1)
    piv = s.pivot(index="fault", columns="set", values="median_min")
    L += ["Median minutes to a sustained alarm (blank = never):", "", piv.to_markdown(), ""]
    d = json.loads((RESULTS / "r1_params.json").read_text())["diagnosis"]
    L += [f"Fault-type diagnosis (signature match): {100 * d['accuracy']:.1f} % of {d['n']} windows.", ""]
    c = pd.read_csv(RESULTS / "r6_control_summary.csv")
    L += ["### Sample project: closed-loop control (3 motors per fault)", "", c.round(2).to_markdown(index=False), ""]
    z = pd.read_csv(RESULTS / "r2_zero_loss.csv")
    cols = ["run", "samples_produced", "L1_sensor_fifo_lost", "L2_queue_lost", "L3_seq_gaps_seen_core1",
            "L4_uplink_lost_packets", "fog_missing_windows", "fog_coverage_pct", "loss_rate_upper95_if_zero",
            "lateness_us_p99"]
    L += ["### Zero sample loss", "", z[cols].to_markdown(index=False), ""]
    so = pd.read_csv(RESULTS / "r3_selfopt_summary.csv")
    L += ["### Self-optimising baseline (healthy drift +10 C, +20 %RH over 2 h)", "", so.round(2).to_markdown(index=False),
          "", "Generated maintenance reports: results/r3_maintenance_reports.md", ""]
    with open(RESULTS / "RESULTS.md", "a") as f:
        f.write("\n".join(L) + "\n")
