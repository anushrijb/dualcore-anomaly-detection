"""
Fog gateway + cloud store + dashboard (UI).

Each edge node is the real C pipeline (build/edge_dualcore: Core 0 acquisition
thread + Core 1 detection thread) run on a simulated motor. The fog gateway
collects their decisions, raises debounced alarms, and stores everything in a
cloud database (SQLite); the dashboard reads it.

    python ui/fog_server.py              # live replay, 30x real time, http://localhost:8060
    python ui/fog_server.py --speed 60   # faster
    python ui/fog_server.py --snapshot   # only write results/dashboard_snapshot.html
"""
import argparse
import io
import json
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "py")]
import pandas as pd

import numpy as np

import sessions
from config import BUILD, RESULTS, SESSION_S, ONSET_S, RAMP_S, DEBOUNCE

FLEET = [  # device, site, note, fault, seed
    ("PUMP-01", "Line A", "healthy", 0, 8101),
    ("PUMP-02", "Line A", "bearing wear from 15 min", 3, 8102),
    ("FAN-03", "Line B", "misalignment from 15 min", 2, 8103),
    ("FAN-04", "Line B", "cooling failure from 15 min", 5, 8104),
]
HTML = (ROOT / "ui" / "dashboard.html").read_text()


def edge_params():
    p = RESULTS / "o4_detection_params.json"
    if p.exists():
        d = json.loads(p.read_text())["params"]["vib_maha"]
        return d["center"], d["sens"]
    return 2.8, 3.0


def run_edge_nodes():
    center, sens = edge_params()
    out = {}
    for dev, site, note, fault, seed in FLEET:
        cfg = sessions.random_motor(np.random.default_rng(seed), seed, fault)
        txt = subprocess.run([str(BUILD / "edge_dualcore"), "--pace=fast", f"--seconds={SESSION_S}",
                              "--learn-s=480", f"--center={center}", f"--sens={sens}", *cfg.as_args()],
                             capture_output=True, text=True, check=True).stdout
        df = pd.read_csv(io.StringIO(txt)).iloc[::4]         # fog keeps one decision per 0.5 s
        out[dev] = (site, note, df)
    return out


class Cloud:
    def __init__(self, path):
        if path.exists():
            path.unlink()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.executescript("""
            CREATE TABLE devices(device TEXT PRIMARY KEY, site TEXT, note TEXT);
            CREATE TABLE telemetry(device TEXT, t REAL, mode TEXT, sim INTEGER, anom INTEGER, vib REAL, temp REAL,
                                   duty REAL, state INTEGER, t_amb REAL, rh REAL);
            CREATE TABLE events(device TEXT, t REAL, sim INTEGER);""")
        self.streak, self.ok, self.alarm = {}, {}, {}

    def register(self, dev, site, note):
        with self.lock:
            self.db.execute("INSERT INTO devices VALUES(?,?,?)", (dev, site, note))
            self.db.commit()

    def ingest(self, dev, r, commit=True):
        learn = int(r["learning"]) == 1
        anom = int(r["anomaly"])
        with self.lock:
            self.db.execute("INSERT INTO telemetry VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (dev, float(r["t_end_s"]), "learn" if learn else "infer",
                             None if learn else int(r["similarity"]), anom,
                             float(r["ax_rms_g"]) * 1000, float(r["temp_c"]),
                             r.get("duty"), r.get("state"), r.get("t_amb"), r.get("rh")))
            s = self.streak.get(dev, 0) + 1 if anom else 0
            ok = 0 if anom else self.ok.get(dev, 0) + 1
            if s >= DEBOUNCE and not self.alarm.get(dev):      # fog-level debounced, latched alarm
                self.alarm[dev] = True
                self.db.execute("INSERT INTO events VALUES(?,?,?)", (dev, float(r["t_end_s"]), int(r["similarity"])))
            if ok >= 40:                                        # 20 s of normal decisions clears it
                self.alarm[dev] = False
            self.streak[dev], self.ok[dev] = s, ok
            if commit:
                self.db.commit()

    def reset(self, dev):
        """the node restarted (window numbers start again at 0): begin a new session"""
        with self.lock:
            self.db.execute("DELETE FROM telemetry WHERE device=?", (dev,))
            self.db.execute("DELETE FROM events WHERE device=?", (dev,))
            self.db.commit()
        self.streak[dev], self.ok[dev], self.alarm[dev] = 0, 0, False

    def fleet(self):
        q = """SELECT d.device, d.site, d.note,
                 (SELECT CASE WHEN mode='learn' THEN 'LEARNING' WHEN anom=1 THEN 'ANOMALY' ELSE 'NORMAL' END
                    FROM telemetry x WHERE x.device=d.device ORDER BY t DESC LIMIT 1),
                 (SELECT vib FROM telemetry x WHERE x.device=d.device ORDER BY t DESC LIMIT 1),
                 (SELECT temp FROM telemetry x WHERE x.device=d.device ORDER BY t DESC LIMIT 1),
                 (SELECT sim FROM telemetry x WHERE x.device=d.device ORDER BY t DESC LIMIT 1),
                 (SELECT duty FROM telemetry x WHERE x.device=d.device ORDER BY t DESC LIMIT 1),
                 (SELECT state FROM telemetry x WHERE x.device=d.device ORDER BY t DESC LIMIT 1),
                 (SELECT t_amb FROM telemetry x WHERE x.device=d.device ORDER BY t DESC LIMIT 1),
                 (SELECT rh FROM telemetry x WHERE x.device=d.device ORDER BY t DESC LIMIT 1)
               FROM devices d ORDER BY d.device"""
        with self.lock:
            rows = list(self.db.execute(q))
        return [dict(zip(["device", "site", "note", "latest", "vib_mg", "temp_c", "sim", "duty", "state", "t_amb", "rh"], r)) for r in rows]

    def series(self, dev):
        with self.lock:
            rows = list(self.db.execute("SELECT t, mode, sim, anom, vib, temp, duty, state, t_amb, rh FROM telemetry "
                                        "WHERE device=? ORDER BY t", (dev,)))
        return [dict(zip(["t", "mode", "sim", "anom", "vib", "temp", "duty", "state", "t_amb", "rh"], r)) for r in rows]

    def events(self):
        with self.lock:
            rows = list(self.db.execute("SELECT device, t, sim FROM events ORDER BY t"))
        return [dict(zip(["device", "t", "sim"], r)) for r in rows]


def snapshot(cloud):
    emb = {"fleet": cloud.fleet(), "series": {d["device"]: cloud.series(d["device"]) for d in cloud.fleet()},
           "events": cloud.events()}
    html = HTML.replace("<script>\nconst EMBED", "<script>window.__EMBEDDED__=" + json.dumps(emb) +
                        ";</script>\n<script>\nconst EMBED", 1)
    (RESULTS / "dashboard_snapshot.html").write_text(html)


def build_snapshot():
    cloud = Cloud(RESULTS / "cloud.db")
    for dev, (site, note, df) in run_edge_nodes().items():
        cloud.register(dev, site, note)
        for r in df.to_dict("records"):
            cloud.ingest(dev, r, commit=False)
        cloud.db.commit()
    snapshot(cloud)
    return cloud.events()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--speed", type=float, default=30.0)
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--ingest", action="store_true", help="no simulation: wait for real ESP32 nodes")
    a = ap.parse_args()
    if a.snapshot:
        print(build_snapshot())
        return
    cloud = Cloud(RESULTS / "cloud_live.db")
    nodes = {} if a.ingest else run_edge_nodes()
    for dev, (site, note, _) in nodes.items():
        cloud.register(dev, site, note)

    def replay():
        ev = sorted(((r["t_end_s"], dev, r) for dev, (_, _, df) in nodes.items() for r in df.to_dict("records")),
                    key=lambda e: e[0])
        t0 = time.time()
        for t, dev, r in ev:
            wait = t / a.speed - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
            cloud.ingest(dev, r)
        print("replay finished")

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ct):
            self.send_response(200); self.send_header("Content-Type", ct)
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                return self._send(HTML.encode(), "text/html; charset=utf-8")
            if u.path == "/api/fleet":
                return self._send(json.dumps(cloud.fleet()).encode(), "application/json")
            if u.path == "/api/series":
                return self._send(json.dumps(cloud.series(parse_qs(u.query).get("device", [""])[0])).encode(), "application/json")
            if u.path == "/api/events":
                return self._send(json.dumps(cloud.events()).encode(), "application/json")
            self.send_error(404)

        def do_POST(self):
            # edge nodes (esp32/sample_project, simulator/board_tk.py) post their decision packets:
            # one JSON packet, or a JSON list of packets (a batch). 200 = ACK: the node then
            # drops those packets from its store-and-forward buffer
            if urlparse(self.path).path != "/api/ingest":
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            packets = body if isinstance(body, list) else [body]
            for p in packets:
                ingest_packet(p)
            with cloud.lock:
                cloud.db.commit()
            self._send(b"ok", "text/plain")

        def log_message(self, *x):
            pass

    seen = {}

    def ingest_packet(p):
        dev = p["device"]
        if dev not in seen:
            cloud.register(dev, "live", "Tkinter simulator board" if dev.startswith("SIM") else "ESP32 node")
            seen[dev] = {"next_win": 0, "missing": 0, "samples": 0}
        st = seen[dev]
        if p["win"] == 0 and st["next_win"] > 0:
            print(f"{dev} restarted - new session", flush=True)
            cloud.reset(dev)
            st.update(next_win=0, missing=0, samples=0)
        if p["win"] > st["next_win"]:
            st["missing"] += p["win"] - st["next_win"]       # windows never delivered
        st["next_win"] = max(st["next_win"], p["win"] + 1)
        st["samples"] += p["n_new"]
        cloud.ingest(dev, {"learning": int(p["sim"] < 0), "anomaly": int(0 <= p["sim"] < 90),
                           "t_end_s": (p["seq_last"] + 1) / 1000.0, "similarity": p["sim"],
                           "ax_rms_g": p["vib"], "temp_c": p["t_motor"],
                           "duty": p.get("duty"), "state": p.get("state"),
                           "t_amb": p.get("t_amb"), "rh": p.get("rh")}, commit=False)
        if p["win"] % 40 == 0 or p.get("state") != st.get("state"):
            st["state"] = p.get("state")
            print(f"{dev} win {p['win']} sim {p['sim']} state {p['state']} duty {p['duty']} | "
                  f"zero-loss: missing windows {st['missing']}, samples {st['samples']} / {p['seq_last'] + 1}, "
                  f"device gaps {p['gaps']}", flush=True)
    if not a.ingest:
        threading.Thread(target=replay, daemon=True).start()
    print(f"Dashboard at http://localhost:{a.port}  ({a.speed:.0f}x real time, Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
