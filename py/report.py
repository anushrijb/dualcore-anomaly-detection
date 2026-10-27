"""Writes results/RESULTS.md from the experiment outputs."""
import json

import pandas as pd

from config import RESULTS


def _md(df, index=False):
    return df.to_markdown(index=index)


def write(det, arch, rt, fleet, events):
    met, dly, params = det
    df_fleet, sc, summ = fleet
    L = ["# Results - Dual-Core Real-Time Anomaly Detection for IoT Sensors", "",
         "Produced by `python run_all.py`. Data are simulated (register-level MPU6050 on a motor "
         "model); the edge pipeline and the OpenMP kernels are real C code measured on this machine.", ""]

    L += ["## Objective 1 + 4 - detection (test sessions: 6 healthy + 25 faulty motors)", "",
          "Thresholds chosen on separate calibration sessions (best balanced accuracy with window "
          "specificity >= 99 %). Labels: healthy = before onset; anomaly = fault severity >= 0.2.", "",
          _md(met.drop(columns=["key", "TP", "FP", "TN", "FN"]).round(2)), ""]
    s = pd.read_csv(RESULTS / "o4_detection_delay_summary.csv")
    s["median_min"] = (s["median"] / 60).round(2)
    piv = s.pivot(index="fault", columns="method", values="median_min")
    miss = s.pivot(index="fault", columns="method", values="missed")
    out = piv.copy().astype(object)
    for c in out.columns:
        out[c] = [("-" if pd.isna(v) else f"{v:.1f}") + (f" ({int(m)} missed)" if m else "")
                  for v, m in zip(piv[c], miss[c])]
    L += ["Median detection delay after fault onset (minutes; 5 motors per fault):", "", _md(out, index=True), ""]

    L += ["## Objective 2 - architecture numbers", ""]
    if arch.get("esp32_code"):
        c = arch["esp32_code"]
        L += [f"- ESP32 code size of features + detector ({c['compiler']} -Os): "
              f"**{c['flash_bytes']} B flash**, {c['static_ram_bytes']} B static RAM"]
    L += [f"- Edge RAM budget: **{arch['ram_total_bytes'] / 1024:.1f} KB** = "
          f"{arch['ram_share_of_esp32_sram_pct']:.1f} % of the ESP32's 520 KB SRAM",
          f"- Learned model: {arch['learned_model_bytes']} B",
          f"- I2C burst read: {arch['i2c_read_us']:.0f} us -> {arch['i2c_core0_load_at_1khz_pct']:.1f} % of Core 0 at 1 kHz",
          f"- Uplink reduced {arch['uplink_reduction_x']:.0f}x by deciding at the edge", ""]
    rates = pd.DataFrame(arch["data_rates_bytes_per_s"], columns=["Link", "Bytes per second"]).round(1)
    L += [_md(rates), ""]

    L += ["## Objective 3a - dual-core vs single-core edge (1 kHz, real time)", ""]
    r = rt[["mode", "proc_us_setting", "lateness_us_p99", "lost_pct", "e2e_latency_us_p99",
            "core0_util", "core1_util"]].copy()
    r.columns = ["mode", "processing per window (us)", "sample lateness p99 (us)", "samples lost (%)",
                 "decision latency p99 (us)", "core 0 load", "core 1 load"]
    L += [_md(r.round(2)), ""]

    L += ["## Objective 3b - OpenMP fog/cloud", "",
          f"Dataset: {summ['motors']} motors x {summ['windows_per_motor']} windows = {summ['windows']} windows, "
          f"{summ['raw_mb']} MB raw; {summ['flagged_windows']} flagged ({100 * summ['flagged_fraction']:.1f} %).",
          f"Fault-type diagnosis on held-out motors: {100 * summ['diagnosis_accuracy_test_motors']:.1f} % "
          f"(n = {summ['diagnosis_n']}). Parallel vs serial results: max |diff| features "
          f"{summ['verify_max_abs_diff_features']}, envelope {summ['verify_max_abs_diff_envelope']}, "
          f"class models (relative) {summ['verify_rel_diff_class_models']}.", "",
          _md(sc[["stage", "threads", "seconds", "speedup", "efficiency", "karp_flatt"]].round(3)), ""]
    sch = df_fleet[df_fleet.experiment == "schedule"]
    p = sch.pivot_table(index=["stage", "schedule"], columns="chunk", values="seconds").round(3)
    p.columns = ["default" if c == 0 else str(c) for c in p.columns]
    L += ["Schedule x chunk size (seconds):", "", _md(p, index=True), ""]
    dl = df_fleet[df_fleet.experiment.isin(["dlp", "ilp"])][["experiment", "threads", "variant", "seconds", "gflops"]]
    L += ["DLP and ILP:", "", _md(dl.round(4)), ""]
    L += ["## UI", "", "Fog alarms in the dashboard snapshot:", ""]
    L += [f"- {e['device']}: alarm at {e['t'] / 60:.2f} min (similarity {e['sim']} %)" for e in events]
    (RESULTS / "RESULTS.md").write_text("\n".join(L) + "\n")
