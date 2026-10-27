"""Builds the C code and exposes it to Python through ctypes.

The SAME C files (mpu6050_sim.c, features.c, detector.c) are used by
  - the dual-core edge program  (build/edge_dualcore)
  - the OpenMP fog/cloud program (build/fleet_omp)
  - the Python experiments       (build/libdualcore.so, this module)
"""
import ctypes as C
import shutil
import subprocess

import numpy as np

from config import CSRC, BUILD

CORE = ["mpu6050_sim.c", "features.c", "detector.c"]
CFLAGS = ["-O3", "-march=native", "-Wall", "-Wno-unused-result"]


def _run(cmd):
    r = subprocess.run(cmd, cwd=CSRC, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(" ".join(map(str, cmd)) + "\n" + r.stderr)
    return r.stdout + r.stderr


def build_all(verbose=False):
    BUILD.mkdir(exist_ok=True)
    out = {}
    out["lib"] = _run(["gcc", *CFLAGS, "-fopenmp", "-fPIC", "-shared", *CORE, "sim_api.c", "ms_core.c", "ms_api.c",
                       "-o", str(BUILD / "libdualcore.so"), "-lm"])
    out["edge_ms"] = _run(["gcc", *CFLAGS, "-pthread", *CORE, "ms_core.c", "edge_ms.c",
                           "-o", str(BUILD / "edge_ms"), "-lm"])
    out["edge"] = _run(["gcc", *CFLAGS, "-pthread", *CORE, "edge_dualcore.c",
                        "-o", str(BUILD / "edge_dualcore"), "-lm"])
    out["fleet"] = _run(["gcc", *CFLAGS, "-fopenmp", *CORE, "fleet_omp.c",
                         "-o", str(BUILD / "fleet_omp"), "-lm"])
    if verbose:
        for k, v in out.items():
            if v.strip():
                print(k, v)
    return out


class MotorCfg(C.Structure):
    _fields_ = [("seed", C.c_uint64), ("fs", C.c_float), ("rpm", C.c_float),
                ("speed_drift", C.c_float), ("load_amp", C.c_float), ("load_period_s", C.c_float),
                ("a1", C.c_float), ("case_temp_c", C.c_float), ("ambient_amp", C.c_float),
                ("ambient_period_s", C.c_float), ("fault", C.c_int), ("onset_s", C.c_float),
                ("ramp_s", C.c_float), ("tau_s", C.c_float)]

    def as_args(self):
        return [f"--{k.replace('_', '-')}={getattr(self, k)}" for k, _ in self._fields_]


_lib = None


def lib():
    global _lib
    if _lib is None:
        so = BUILD / "libdualcore.so"
        if not so.exists():
            build_all()
        L = C.CDLL(str(so))
        fp, u8p = C.POINTER(C.c_float), C.POINTER(C.c_uint8)
        L.api_motor_default.argtypes = [C.POINTER(MotorCfg), C.c_ulonglong]
        L.api_session_features.argtypes = [C.POINTER(MotorCfg), C.c_int, fp]
        L.api_raw_samples.argtypes = [C.POINTER(MotorCfg), C.c_longlong, C.c_int, C.POINTER(C.c_short)]
        L.api_case_temp.argtypes = [C.POINTER(MotorCfg), C.c_int, C.c_double, C.c_double, fp]
        L.api_severity.argtypes = [C.POINTER(MotorCfg), C.c_int, C.c_double, C.c_double, fp]
        L.api_det_init.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_float, C.c_float]
        L.api_det_learn_batch.argtypes = [C.c_void_p, fp, C.c_int]
        L.api_det_finalize.argtypes = [C.c_void_p]
        L.api_det_score_batch.argtypes = [C.c_void_p, fp, C.c_int, fp, u8p]
        L.api_det_model_bytes.argtypes = [C.c_void_p]
        L.api_det_model_bytes.restype = C.c_uint
        L.api_i2c_us.restype = C.c_float
        L.api_det_set_floor.argtypes = [C.c_void_p, C.c_int, C.c_float]
        assert L.api_cfg_sizeof() == C.sizeof(MotorCfg), "MotorCfg layout mismatch"
        _lib = L
    return _lib


def _fp(a):
    return a.ctypes.data_as(C.POINTER(C.c_float))


def motor(seed, **kw):
    c = MotorCfg()
    lib().api_motor_default(C.byref(c), seed)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def session_features(cfg, n_windows):
    X = np.empty((n_windows, lib().api_n_feat()), np.float32)
    lib().api_session_features(C.byref(cfg), n_windows, _fp(X))
    return X


def raw_samples(cfg, n0, n):
    out = np.empty((n, 7), np.int16)
    lib().api_raw_samples(C.byref(cfg), n0, n, out.ctypes.data_as(C.POINTER(C.c_short)))
    return out


def case_temp(cfg, n, t0, dt):
    out = np.empty(n, np.float32)
    lib().api_case_temp(C.byref(cfg), n, t0, dt, _fp(out))
    return out


def severity(cfg, n, t0, dt):
    out = np.empty(n, np.float32)
    lib().api_severity(C.byref(cfg), n, t0, dt, _fp(out))
    return out


class Detector:
    """Wraps the C detector used on the ESP32 (kind 0 = ZSM, 1 = Mahalanobis)."""

    def __init__(self, kind, dim, center, sens, floors=None):
        self.buf = C.create_string_buffer(lib().api_det_sizeof())
        lib().api_det_init(self.buf, kind, dim, center, sens)
        for i, v in (floors or {}).items():          # per-feature minimum spread
            lib().api_det_set_floor(self.buf, int(i), float(v))

    def learn(self, X):
        X = np.ascontiguousarray(X, np.float32)
        lib().api_det_learn_batch(self.buf, _fp(X), len(X))
        return lib().api_det_finalize(self.buf)

    def score(self, X):
        X = np.ascontiguousarray(X, np.float32)
        d = np.empty(len(X), np.float32)
        s = np.empty(len(X), np.uint8)
        lib().api_det_score_batch(self.buf, _fp(X), len(X), _fp(d), s.ctypes.data_as(C.POINTER(C.c_uint8)))
        return d, s

    def model_bytes(self):
        return lib().api_det_model_bytes(self.buf)


def have(tool):
    return shutil.which(tool) is not None


# ------------------------------------------------ multi-sensor sample project --
class MsCfg(C.Structure):
    _fields_ = [("seed", C.c_uint64), ("rpm_nom", C.c_float), ("a1", C.c_float),
                ("ambient_c", C.c_float), ("rh_pct", C.c_float), ("drift_c", C.c_float),
                ("drift_rh", C.c_float), ("drift_s", C.c_float), ("load_amp", C.c_float),
                ("load_period_s", C.c_float), ("fault", C.c_int), ("onset_s", C.c_float),
                ("ramp_s", C.c_float)]

    def as_args(self):
        return [f"--{k.replace('_', '-')}={getattr(self, k)}" for k, _ in self._fields_]


MS_FAULTS = ["normal", "imbalance", "bearing", "misalignment", "looseness", "overload", "cooling", "humidity"]
MS_FLOORS = {9: 0.5, 10: 0.5, 11: 2.0}     # same as ms_detector_setup() in C
MS_GROUPS = {"vibration": 1, "temperature": 2, "humidity": 4, "rpm": 8, "current": 16, "sound": 32}


def ms_cfg(seed, **kw):
    c = MsCfg()
    lib().ms_api_cfg_default(C.byref(c), C.c_ulonglong(seed))
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def ms_feature_names():
    L = lib()
    L.ms_api_feature_name.restype = C.c_char_p
    return [L.ms_api_feature_name(i).decode() for i in range(L.ms_api_nf())]


def ms_feature_groups():
    return np.array([lib().ms_api_feature_group(i) for i in range(lib().ms_api_nf())])


def ms_run(cfg, mode=-1, seconds=1800.0, learn_s=300.0, center=3.0, sens=3.0, adapt=0.0):
    """mode: -1 open loop (100 % speed), 0 threshold rules, 1 AI + threshold rules."""
    L = lib()
    nf = L.ms_api_nf()
    mw = int(seconds * 1000 / 128) + 2
    f32 = lambda n: np.zeros(n, np.float32)
    t, duty, sev, rpm = f32(mw), f32(mw), f32(mw), f32(mw)
    state, sim = np.zeros(mw, np.int32), np.zeros(mw, np.int32)
    raw, feat, ctrl = f32(mw * 6), f32(mw * nf), f32(8)
    ip = C.POINTER(C.c_int)
    L.ms_api_run.restype = C.c_int
    n = L.ms_api_run(C.byref(cfg), mode, C.c_float(seconds), C.c_float(learn_s), C.c_float(center),
                     C.c_float(sens), C.c_float(adapt), mw, _fp(t), _fp(duty), state.ctypes.data_as(ip),
                     sim.ctypes.data_as(ip), _fp(sev), _fp(rpm), _fp(raw), _fp(feat), _fp(ctrl))
    return {"t": t[:n], "duty": duty[:n], "state": state[:n], "sim": sim[:n], "sev": sev[:n],
            "rpm": rpm[:n], "raw": raw[:n * 6].reshape(n, 6), "X": feat[:n * nf].reshape(n, nf),
            "t_warn": float(ctrl[0]), "t_stop": float(ctrl[1]), "reason": int(ctrl[2]),
            "gaps": int(ctrl[3]), "received": int(ctrl[4])}
