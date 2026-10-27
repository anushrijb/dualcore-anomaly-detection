"""Global configuration for the proposed work:
   Dual-Core Real-Time Anomaly Detection for IoT Sensors (ESP32 + MPU6050)."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
for _d in (BUILD, RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- sensing --
FS_HZ = 1000.0             # MPU6050 accel output rate (max 1 kHz)
WIN = 256                  # samples per analysis window (0.256 s)
HOP = 128                  # 50 % overlap -> a decision every 0.128 s
N_FEAT = 31

# -------------------------------------------------------------- learning --
LEARN_WINDOWS = 256        # on-device learning: 256 windows = 32.8 s of healthy motor
SIM_THRESHOLD = 90         # same 90 % similarity threshold as the base paper
DEBOUNCE = 3               # alarm after 3 consecutive anomalous decisions (all methods)

# ------------------------------------------------------- session timeline --
# 0-480 s  : healthy, on-device learning (base paper needs 30 x 16 s = 480 s)
# 480-900 s: healthy monitoring
# 900 s    : fault onset, severity ramps 0 -> 1 over 600 s
SESSION_S = 1500
ONSET_S = 900
RAMP_S = 600
BASE_LEARN_S = 480
LABEL_ANOMALY_SEVERITY = 0.2   # windows with severity >= 0.2 are labelled "anomaly"

# ------------------------------------------------- base paper (temperature) --
BASE_WINDOW = 16
BASE_STRIDE = 4
BASE_LEARN = 30
STATIC_ALARM_C = 50.0

# ---------------------------------------------------------------- edge RT --
I2C_READ_US = 385.0        # 17 bytes x 9 bits at 400 kHz + overhead (computed in C too)
RING_CAPACITY = 4096

# ------------------------------------------------------------- fog/cloud --
FLEET_MOTORS = int(os.environ.get("FLEET_MOTORS", 64))
FLEET_WINDOWS = int(os.environ.get("FLEET_WINDOWS", 512))  # per motor -> 32768 windows (~117 MB raw)

FAULT_NAMES = ["normal", "imbalance", "misalignment", "bearing", "looseness", "cooling"]
SEED = 2026
