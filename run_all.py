"""
Runs every experiment of the proposed work and writes results/RESULTS.md.

    python run_all.py              (about 6-8 minutes on a 2-core machine, faster on more cores)
    python run_all.py --quick      (smaller fleet, shorter real-time runs)
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [ROOT, os.path.join(ROOT, "py")]

QUICK = "--quick" in sys.argv
if QUICK:
    os.environ.setdefault("FLEET_MOTORS", "32")
    os.environ.setdefault("FLEET_WINDOWS", "256")

import cbind
import exp_architecture
import exp_detection
import exp_edge_rt
import exp_fleet
import report
import report_remarks
import exp_ms_sensors
import exp_ms_control
import exp_zeroloss
import exp_selfopt


def step(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78, flush=True)


def main():
    t0 = time.time()
    step("Build C code: libdualcore.so, edge_dualcore (pthreads), fleet_omp (OpenMP)")
    cbind.build_all(verbose=True)

    step("Objective 1 + 4: detection on test sessions vs the base paper")
    det = exp_detection.run()

    step("Objective 2: ESP32 footprint, RAM budget, I/O data rates")
    arch = exp_architecture.run(verbose=False)

    step("Objective 3a: dual-core vs single-core edge pipeline (real time)")
    if QUICK:
        exp_edge_rt.PROC_US[:] = [0, 40000, 90000, 140000]
        exp_edge_rt.SECONDS = 4
    rt = exp_edge_rt.run()

    step("Objective 3b: OpenMP fog/cloud on the fleet dataset (TLP, schedules, DLP, ILP)")
    fleet = exp_fleet.run(reps=1 if QUICK else 3)

    step("UI: fog gateway + cloud DB + dashboard snapshot")
    import importlib.util
    spec = importlib.util.spec_from_file_location("fog_server", os.path.join(ROOT, "ui", "fog_server.py"))
    fog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fog)
    events = fog.build_snapshot()
    print("  alarms:", events)

    report.write(det, arch, rt, fleet, events)

    step("Review 1, remark 1: multi-input one-class Z-score model (6 sensors)")
    _, _, params, _ = exp_ms_sensors.run()
    center = params["all 6 sensors (Z-score)"]["center"]
    step("Review 1, sample project: warning -> step speed down -> stop (closed loop)")
    exp_ms_control.run(center)
    step("Review 1, remark 2: zero sample loss (sequence numbers, every hop)")
    if QUICK:
        exp_zeroloss.run(center, rt_seconds=60, soak_s=3600)
    else:
        exp_zeroloss.run(center)
    step("Review 1, remark 3: self-optimising baseline + GenAI maintenance reports")
    exp_selfopt.run(center)
    report_remarks.write()
    print(f"\nDone in {time.time() - t0:.0f}s. See results/RESULTS.md, results/figures/, "
          f"results/dashboard_snapshot.html")


if __name__ == "__main__":
    main()
