"""The streaming dual-core edge program (ring buffer, Core 0 / Core 1 threads)
must make exactly the same decisions as the batch evaluation used in the
experiments: both run the same C features + detector."""
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "py")]
import numpy as np
import pandas as pd

import cbind
from config import BUILD, SESSION_S

if __name__ == "__main__":
    worst = 0
    for fault in (0, 3, 5):
        cfg = cbind.motor(31337 + fault, fault=fault, onset_s=900.0, ramp_s=600.0)
        out = subprocess.run([str(BUILD / "edge_dualcore"), "--pace=fast", f"--seconds={SESSION_S}",
                              "--learn-s=480", "--center=2.8", "--sens=3", *cfg.as_args()],
                             capture_output=True, text=True, check=True).stdout
        edge = pd.read_csv(io.StringIO(out))
        n = len(edge)
        X = cbind.session_features(cfg, n)
        t = (np.arange(n) * 128 + 256) / 1000.0
        det = cbind.Detector(1, 31, 2.8, 3.0)
        det.learn(X[t <= 480])
        _, sim = det.score(X)
        m = edge.learning.values == 0
        diff = np.abs(edge.similarity.values[m] - sim[m]).max()
        worst = max(worst, diff)
        print(f"fault {fault}: {m.sum()} decisions, max |sim_edge - sim_batch| = {diff:.0f}, "
              f"alarms edge={int(edge.anomaly.sum())} batch={int((sim[m] < 90).sum())}")
    print("PASS" if worst == 0 else "FAIL")
    sys.exit(0 if worst == 0 else 1)
