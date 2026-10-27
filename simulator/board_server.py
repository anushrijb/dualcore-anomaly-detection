"""
Simulator engine: builds the dual-core edge C code (csrc/) and runs it on two
threads pinned to two CPU cores; board_tk.py is the window that uses it.

Uplink: every decision packet is also sent to the fog server
(python ui/fog_server.py, http://localhost:8060), exactly like the ESP32's
Wi-Fi task: a packet stays in a local buffer until the fog acknowledges it
(store-and-forward), so the dashboard shows the simulator board live as SIM-01.

"""
import argparse
import ctypes as C
import json
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
SRC = HERE / "csrc"
LIB = HERE / "build" / "libboard.so"
C_FILES = [SRC / f for f in ("board.c", "edge_twin.c", "ms_core.c", "features.c", "detector.c", "mpu6050_sim.c")]

FAULTS = {1: "imbalance", 2: "bearing", 3: "misalignment", 4: "looseness", 5: "overload"}
STATES = ["NORMAL", "WARNING", "DERATED", "STOPPED"]
REASONS = {1: "vibration", 2: "temperature", 4: "humidity", 8: "RPM", 16: "current", 64: "AI model"}


class Packet(C.Structure):
    _fields_ = [("win", C.c_uint32), ("seq_last", C.c_uint32), ("n_new", C.c_uint32), ("sim", C.c_int),
                ("state", C.c_int), ("duty", C.c_float), ("vib", C.c_float), ("t_motor", C.c_float),
                ("t_amb", C.c_float), ("rh", C.c_float), ("rpm", C.c_float), ("current", C.c_float),
                ("reason", C.c_uint32), ("gaps", C.c_uint64)]


def build():
    LIB.parent.mkdir(exist_ok=True)
    if LIB.exists() and all(LIB.stat().st_mtime > f.stat().st_mtime for f in C_FILES):
        return
    print("building the virtual board (C code)...", flush=True)
    subprocess.run(["gcc", "-O2", "-fPIC", "-shared", "-pthread", "-DFFT_MAX=2048", f"-I{SRC}",
                    *map(str, C_FILES), "-o", str(LIB), "-lm"], check=True)


class Board:
    def __init__(self, learn_s, speed, fog_url="http://localhost:8060/api/ingest", device="SIM-01"):
        build()
        self.L = C.CDLL(str(LIB))
        self.L.board_set_inputs.argtypes = [C.c_int, C.c_float, C.c_float, C.c_float, C.c_float]
        self.L.board_start.argtypes = [C.c_float, C.c_float]
        self.L.board_packets.argtypes = [C.POINTER(Packet), C.c_int]
        self.L.board_counters.argtypes = [C.POINTER(C.c_uint64)]
        assert self.L.board_sizeof_packet() == C.sizeof(Packet), "packet layout mismatch"
        self.learn_s, self.speed = learn_s, speed
        self.lock = threading.Lock()
        self.inputs = {"fault_type": 1, "sev": 0.0, "motor_c": 40.0, "amb_c": 25.0, "rh": 50.0}
        self.fog_url, self.device = fog_url, device
        self.uplink, self.sent, self.fog_ok = deque(maxlen=20000), 0, False
        self.start()
        threading.Thread(target=self.poll, daemon=True).start()
        if fog_url:
            threading.Thread(target=self.forward, daemon=True).start()

    def start(self):
        with self.lock:
            self.packets, self.events, self.last_state, self.last_learning = [], [], None, None
            if hasattr(self, "uplink"):
                self.uplink.clear()
            self.L.board_stop()
            self.apply_inputs()
            self.L.board_start(self.learn_s, self.speed)
            self.t_start = time.time()

    def apply_inputs(self):
        i = self.inputs
        self.L.board_set_inputs(int(i["fault_type"]), float(i["sev"]), float(i["motor_c"]),
                                float(i["amb_c"]), float(i["rh"]))

    def set_inputs(self, d):
        with self.lock:
            for k in self.inputs:
                if k in d:
                    self.inputs[k] = float(d[k])
            self.apply_inputs()

    def event(self, t, text):
        self.events.append({"t": round(t, 1), "text": text})

    def poll(self):
        buf = (Packet * 256)()
        while True:
            time.sleep(0.05)
            with self.lock:
                n = self.L.board_packets(buf, 256)
                for k in range(n):
                    p = buf[k]
                    d = {f: getattr(p, f) for f, _ in Packet._fields_}
                    d["t"] = (p.seq_last + 1) / 1000.0
                    learning = p.sim < 0
                    if learning != self.last_learning:
                        self.event(d["t"], "learning the healthy motor" if learning else
                                   "learning finished - monitoring with the Z-score model")
                        self.last_learning = learning
                    if self.last_state is not None and p.state != self.last_state:
                        why = ", ".join(v for b, v in REASONS.items() if p.reason & b) or "-"
                        self.event(d["t"], f"{STATES[self.last_state]} -> {STATES[p.state]} "
                                           f"(speed {100 * p.duty:.0f} %, triggered by: {why})")
                    self.last_state = p.state
                    self.packets.append(d)
                    if self.fog_url:
                        self.uplink.append(d)
                if len(self.packets) > 30000:
                    del self.packets[:10000]

    def forward(self):
        """store-and-forward uplink to the fog server: packets go in batches (up to 64 per
        request, about 4 requests a second) and are dropped only after the ACK (HTTP 200)"""
        while True:
            time.sleep(0.25)
            if not self.uplink:
                continue
            batch = list(self.uplink)[:64]
            body = json.dumps([{"device": self.device, "win": d["win"], "seq_last": d["seq_last"],
                                "n_new": d["n_new"], "sim": d["sim"], "state": d["state"],
                                "duty": round(d["duty"], 3), "vib": round(d["vib"], 5),
                                "t_motor": round(d["t_motor"], 2), "t_amb": round(d["t_amb"], 2),
                                "rh": round(d["rh"], 1), "rpm": round(d["rpm"]), "current": round(d["current"], 3),
                                "gaps": d["gaps"]} for d in batch]).encode()
            try:
                req = urllib.request.Request(self.fog_url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=3) as r:
                    ok = r.status == 200
            except Exception:
                ok = False
            if ok:
                for d in batch:                       # ACK: remove exactly the packets that were sent
                    if self.uplink and self.uplink[0] is d:
                        self.uplink.popleft()
                self.sent += len(batch)
                self.fog_ok = True
            else:
                self.fog_ok = False
                time.sleep(1.0)                       # fog not running yet: keep the packets, retry

    def state(self, since):
        cnt = (C.c_uint64 * 5)()
        self.L.board_counters(cnt)
        with self.lock:
            new = [p for p in self.packets[-3000:] if p["win"] >= since]
            last = self.packets[-1] if self.packets else None
            gaps = last["gaps"] if last else 0
            return {"packets": new, "last": last, "inputs": self.inputs, "events": self.events[-12:],
                    "learn_s": self.learn_s, "speed": self.speed,
                    "counters": {"samples": int(cnt[0]), "lost_queue": int(cnt[1]), "lost_late": int(cnt[2]),
                                 "packets_dropped": int(cnt[3]), "queued": int(cnt[4]), "gaps": int(gaps),
                                 "windows": (last["win"] + 1) if last else 0,
                                 "fog_sent": self.sent, "fog_pending": len(self.uplink),
                                 "fog_ok": self.fog_ok, "fog_on": bool(self.fog_url)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed (1 = real time)")
    ap.add_argument("--learn", type=float, default=60.0, help="on-device learning time (s)")
    a = ap.parse_args()
    board = Board(a.learn, a.speed)
    page = (HERE / "board.html").read_bytes() if (HERE / "board.html").exists() else b"run board_tk.py"

    class H(BaseHTTPRequestHandler):
        def send(self, body, ct="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self.send(page, "text/html; charset=utf-8")
            if u.path == "/api/state":
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                return self.send(json.dumps(board.state(since)).encode())
            self.send_error(404)

        def do_POST(self):
            u = urlparse(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
            if u.path == "/api/inputs":
                board.set_inputs(json.loads(body or b"{}"))
            elif u.path == "/api/restart":
                board.L.board_restart()
            elif u.path == "/api/relearn":
                board.start()
            else:
                return self.send_error(404)
            self.send(b'{"ok":true}')

        def log_message(self, *x):
            pass

    print(f"Virtual ESP32 board running ({a.speed:g}x real time, learning {a.learn:g} s).")
    print(f"Open http://localhost:{a.port} in your browser. Ctrl+C to stop.", flush=True)
    try:
        ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
    except KeyboardInterrupt:
        board.L.board_stop()


if __name__ == "__main__":
    sys.exit(main())
