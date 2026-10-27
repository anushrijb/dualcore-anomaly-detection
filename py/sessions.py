"""Recording sessions shared by the detection experiments.

A session = one motor monitored for 1500 s:
    0-480 s    healthy: on-device learning (all methods)
    480-900 s  healthy monitoring
    900 s      fault onset; severity ramps 0 -> 1 over 600 s (none for normal)

For each session we keep
    X        vibration feature windows (C feature extractor, OpenMP)   [n, 31]
    t_win    end time of each window (s)
    sev_win  fault severity at each window end
    temp     DS18B20 reading at 1 Hz (base paper path)                  [1500]
"""
import numpy as np

from config import (SESSION_S, ONSET_S, RAMP_S, FS_HZ, WIN, HOP, SEED)
import cbind


def random_motor(rng, seed, fault):
    return cbind.motor(seed, fault=fault, onset_s=ONSET_S, ramp_s=RAMP_S,
                       rpm=float(rng.uniform(1420, 1500)), load_amp=float(rng.uniform(0.15, 0.35)),
                       a1=float(rng.uniform(0.04, 0.06)), case_temp_c=float(rng.uniform(38, 44)),
                       ambient_amp=float(rng.uniform(0.2, 0.8)),
                       load_period_s=float(rng.uniform(180, 300)))


class Session:
    def __init__(self, cfg, label):
        self.cfg, self.label, self.fault = cfg, label, cfg.fault
        n = int((SESSION_S * FS_HZ - WIN) // HOP + 1)
        self.X = cbind.session_features(cfg, n)
        self.t_win = (np.arange(n) * HOP + WIN) / FS_HZ
        self.sev_win = cbind.severity(cfg, n, WIN / FS_HZ, HOP / FS_HZ)
        rng = np.random.default_rng(cfg.seed + 99)
        true_t = cbind.case_temp(cfg, SESSION_S, 0.0, 1.0)
        self.temp = np.round((true_t + rng.normal(0, 0.03, SESSION_S)) / 0.0625) * 0.0625


def build(kind="test"):
    """kind = 'calib' (choose detector parameters) or 'test' (report)."""
    per_fault, normals, base = (2, 3, 5000) if kind == "calib" else (5, 6, 7000)
    rng = np.random.default_rng(SEED + (1 if kind == "calib" else 2))
    out, seed = [], base
    for _ in range(normals):
        out.append(Session(random_motor(rng, seed, 0), "normal")); seed += 1
    for fault in range(1, 6):
        for _ in range(per_fault):
            out.append(Session(random_motor(rng, seed, fault), f"fault{fault}")); seed += 1
    return out
