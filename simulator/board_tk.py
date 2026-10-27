"""
Tkinter simulator of the project (sample project: ESP32 + MPU6050 + Hall + DHT22 + L298N).

    python3 board_tk.py              real time (learning takes 60 s)
    python3 board_tk.py --speed 2    2x faster (learning takes 30 s)

Ubuntu/WSL needs Tk once:  sudo apt install -y python3-tk

The same edge C code as the ESP32 firmware runs on two threads pinned to two
CPU cores ("Core 0" sensors at 1 kHz -> queue -> "Core 1" features + Z-score
model + controller). This window is the board, the sensors you can change, and
live charts.
"""
import argparse
import math
import tkinter as tk
from tkinter import ttk

from board_server import Board, STATES

FAULT_NAMES = ["imbalance", "bearing", "misalignment", "looseness", "overload"]
BG, PANEL, INK, INK2, GRID = "#f4f3ef", "#fcfcfb", "#141413", "#55544f", "#e4e3df"
BLUE, GREEN, AMBER, RED, ORANGE, VIOLET = "#2a78d6", "#1baf7a", "#d99a06", "#e34948", "#eb6834", "#6250d6"
SPAN = 180.0                                   # seconds shown on the charts


class App:
    def __init__(self, root, board):
        self.root, self.board = root, board
        self.rows, self.next, self.k, self.angle = [], 0, 0, 0.0
        root.title("Dual-core edge AI - virtual ESP32 motor protection")
        root.configure(bg=BG)
        root.geometry("1280x800")
        st = ttk.Style()
        st.configure("TLabel", background=PANEL)
        st.configure("TFrame", background=PANEL)

        left = tk.Frame(root, bg=BG)
        left.pack(side="left", fill="y", padx=(12, 6), pady=12)
        right = tk.Frame(root, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(6, 12), pady=12)

        # ---------------- board drawing ----------------
        self.cv = tk.Canvas(left, width=380, height=300, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
        self.cv.pack()
        self.draw_board_static()

        # ---------------- controls ----------------
        ctl = tk.Frame(left, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
        ctl.pack(fill="x", pady=(10, 0))
        tk.Label(ctl, text="Sensors and faults (move these)", bg=PANEL, fg=INK,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(8, 2))
        row = tk.Frame(ctl, bg=PANEL)
        row.pack(fill="x", padx=10)
        tk.Label(row, text="Fault type", bg=PANEL, fg=INK2).pack(side="left")
        self.fault = ttk.Combobox(row, values=FAULT_NAMES, state="readonly", width=14)
        self.fault.current(0)
        self.fault.pack(side="right")
        self.fault.bind("<<ComboboxSelected>>", lambda e: self.send())
        self.sev = self.slider(ctl, "Fault severity (%)", 0, 100, 0)
        self.motor = self.slider(ctl, "Motor temperature, MPU6050 (°C)", 20, 70, 40)
        self.amb = self.slider(ctl, "Room temperature, DHT22 (°C)", 10, 50, 25)
        self.rh = self.slider(ctl, "Humidity, DHT22 (%)", 20, 100, 50)
        btns = tk.Frame(ctl, bg=PANEL)
        btns.pack(fill="x", padx=10, pady=8)
        tk.Button(btns, text="RESTART motor", bg=GREEN, fg="white", relief="flat",
                  command=self.board.L.board_restart).pack(side="left", padx=(0, 6))
        tk.Button(btns, text="Healthy sliders", relief="flat", command=self.healthy).pack(side="left", padx=(0, 6))
        tk.Button(btns, text="Relearn", relief="flat", command=self.relearn).pack(side="left")
        tk.Label(ctl, text="Limits: vibration 0.075 / 0.15 g · motor 52 / 60 °C\nroom 40 / 45 °C · humidity 80 / 90 %"
                           "   (warning / stop)", bg=PANEL, fg=INK2, justify="left",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=10, pady=(0, 8))

        ev = tk.Frame(left, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
        ev.pack(fill="both", expand=True, pady=(10, 0))
        tk.Label(ev, text="Events", bg=PANEL, fg=INK, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
        self.events = tk.Listbox(ev, height=7, bg=PANEL, fg=INK2, relief="flat", font=("Segoe UI", 9),
                                 highlightthickness=0)
        self.events.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # ---------------- status + charts ----------------
        stat = tk.Frame(right, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
        stat.pack(fill="x")
        self.vals = {}
        for i, (key, label) in enumerate([("state", "Controller"), ("sim", "AI similarity"), ("speed", "Motor speed"),
                                          ("rpm", "RPM (Hall)"), ("vib", "Vibration RMS"), ("tm", "Motor temp"),
                                          ("dht", "DHT22"), ("cur", "Current")]):
            f = tk.Frame(stat, bg=PANEL)
            f.grid(row=i // 4, column=i % 4, sticky="w", padx=12, pady=6)
            tk.Label(f, text=label, bg=PANEL, fg=INK2, font=("Segoe UI", 9)).pack(anchor="w")
            v = tk.Label(f, text="–", bg=PANEL, fg=INK, font=("Segoe UI", 14, "bold"))
            v.pack(anchor="w")
            self.vals[key] = v
        for c in range(4):
            stat.grid_columnconfigure(c, weight=1)

        self.c_sim = self.chart(right, "Edge AI similarity  (below 90 % = anomaly; purple = learning)")
        self.c_vib = self.chart(right, "Vibration RMS, MPU6050 (g)  (dashed: warning 0.075 / stop 0.15)")
        self.c_spd = self.chart(right, "Motor speed set by the controller, L298N duty (%)")

        zl = tk.Frame(right, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
        zl.pack(fill="x", pady=(8, 0))
        tk.Label(zl, text="Zero sample loss (sequence number on every sample)", bg=PANEL, fg=INK,
                 font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=6, sticky="w", padx=10, pady=(6, 0))
        self.zl = {}
        for i, (key, label) in enumerate([("samples", "Samples (1 kHz)"), ("gaps", "Gaps seen by Core 1"),
                                          ("lost_queue", "Lost: queue full"), ("lost_late", "Lost: FIFO overflow"),
                                          ("windows", "Windows"), ("queued", "In queue")]):
            f = tk.Frame(zl, bg=PANEL)
            f.grid(row=1, column=i, sticky="w", padx=10, pady=(0, 8))
            tk.Label(f, text=label, bg=PANEL, fg=INK2, font=("Segoe UI", 8)).pack(anchor="w")
            v = tk.Label(f, text="0", bg=PANEL, fg=INK, font=("Segoe UI", 12, "bold"))
            v.pack(anchor="w")
            self.zl[key] = v
        for c in range(6):
            zl.grid_columnconfigure(c, weight=1)
        self.uplink = tk.Label(zl, text="", bg=PANEL, fg=INK2, font=("Segoe UI", 9), anchor="w")
        self.uplink.grid(row=2, column=0, columnspan=6, sticky="w", padx=10, pady=(0, 8))

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.send()
        self.update()

    # ---------------------------------------------------------------- widgets
    def slider(self, parent, text, lo, hi, val):
        f = tk.Frame(parent, bg=PANEL)
        f.pack(fill="x", padx=10, pady=(4, 0))
        tk.Label(f, text=text, bg=PANEL, fg=INK2).pack(anchor="w")
        s = tk.Scale(f, from_=lo, to=hi, orient="horizontal", bg=PANEL, fg=INK, troughcolor=GRID,
                     highlightthickness=0, length=340, command=lambda v: self.send())
        s.set(val)
        s.pack(fill="x")
        return s

    def chart(self, parent, title):
        f = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
        f.pack(fill="both", expand=True, pady=(8, 0))
        tk.Label(f, text=title, bg=PANEL, fg=INK, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(4, 0))
        c = tk.Canvas(f, bg=PANEL, highlightthickness=0, height=120)
        c.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        return c

    def draw_board_static(self):
        c = self.cv
        c.create_text(12, 14, text="Board", anchor="w", font=("Segoe UI", 11, "bold"), fill=INK)
        c.create_rectangle(18, 30, 362, 104, fill="#0d6b3c", outline="")                # LCD module
        c.create_rectangle(28, 38, 352, 96, fill="#7fa316", outline="#0a0a0a", width=2)
        self.lcd = c.create_text(40, 67, text="STARTING...", anchor="w", font=("Courier New", 15, "bold"),
                                 fill="#16210a")
        c.create_rectangle(150, 120, 250, 230, fill="#1f1f1f", outline="#000")          # ESP32
        c.create_rectangle(165, 128, 235, 175, fill="#9a9a9a", outline="")
        c.create_text(200, 151, text="ESP32", fill="white", font=("Segoe UI", 10, "bold"))
        c.create_text(200, 192, text="Core 0: sensors", fill="#cfcfcf", font=("Segoe UI", 7))
        c.create_text(200, 206, text="Core 1: AI + control", fill="#cfcfcf", font=("Segoe UI", 7))
        self.leds = {}
        for i, (key, col, lab) in enumerate([("G", GREEN, "NORMAL"), ("Y", AMBER, "WARNING"), ("R", RED, "STOP")]):
            x = 34 + i * 40
            self.leds[key] = (c.create_oval(x - 9, 128, x + 9, 146, fill=GRID, outline=""), col)
            c.create_text(x, 156, text=lab, fill=INK2, font=("Segoe UI", 6))
        self.buzz = c.create_oval(20, 178, 52, 210, fill="#222", outline="")
        c.create_text(36, 218, text="BUZZER", fill=INK2, font=("Segoe UI", 7))
        c.create_oval(285, 125, 365, 205, fill="#d9d8d3", outline="#888", width=2)      # DC motor
        self.blade = [c.create_line(325, 165, 325, 130, width=5, fill=BLUE, capstyle="round") for _ in range(3)]
        c.create_oval(319, 159, 331, 171, fill="#333", outline="")
        c.create_text(325, 218, text="DC motor (L298N)", fill=INK2, font=("Segoe UI", 7))
        self.motor_lab = c.create_text(325, 232, text="", fill=INK, font=("Segoe UI", 9, "bold"))
        c.create_text(190, 262, text="MPU6050 · Hall · DHT22 · ACS712 · mic → Core 0 at 1 kHz",
                      fill=INK2, font=("Segoe UI", 8))
        self.lcd_note = c.create_text(190, 282, text="", fill=INK2, font=("Segoe UI", 8))

    # ---------------------------------------------------------------- actions
    def send(self):
        self.board.set_inputs({"fault_type": FAULT_NAMES.index(self.fault.get()) + 1 if hasattr(self, "fault") else 1,
                               "sev": self.sev.get() / 100 if hasattr(self, "sev") else 0,
                               "motor_c": self.motor.get() if hasattr(self, "motor") else 40,
                               "amb_c": self.amb.get() if hasattr(self, "amb") else 25,
                               "rh": self.rh.get() if hasattr(self, "rh") else 50})

    def healthy(self):
        self.sev.set(0); self.motor.set(40); self.amb.set(25); self.rh.set(50)
        self.send()

    def relearn(self):
        self.board.start()
        self.rows, self.next = [], 0

    def close(self):
        self.board.L.board_stop()
        self.root.destroy()

    # ---------------------------------------------------------------- update
    def lcd_text(self, p):
        if p is None:
            return "STARTING..."
        if p["sim"] < 0:
            l1 = f"LEARNING {p['t']:3.0f}/{self.board.learn_s:.0f}s"
        else:
            l1 = f"{STATES[p['state']]:<7} sim{p['sim']:3d}%"
        page = (self.k // 10) % 3
        if page == 0:
            l2 = f"spd{p['duty'] * 100:3.0f}% {p['rpm']:4.0f}rpm"
        elif page == 1:
            l2 = f"V{p['vib']:.3f}g T{p['t_motor']:.1f}C"
        else:
            l2 = f"{self.fault.get()[:9]:<9} {self.sev.get():3d}%"
        return l1[:16] + "\n" + l2[:16]

    def plot(self, c, series, ymin, ymax, hlines=(), shade=True):
        c.delete("all")
        W, H = c.winfo_width(), c.winfo_height()
        if W < 50 or not self.rows:
            return
        L, R, T, B = 44, 10, 6, 18
        tmax = self.rows[-1]["t"]
        tmin = max(0.0, tmax - SPAN)
        x1 = max(tmax, tmin + SPAN)
        X = lambda t: L + (t - tmin) / (x1 - tmin) * (W - L - R)
        Y = lambda v: T + (1 - (min(max(v, ymin), ymax) - ymin) / (ymax - ymin)) * (H - T - B)
        for i in range(5):
            v = ymin + i * (ymax - ymin) / 4
            c.create_line(L, Y(v), W - R, Y(v), fill=GRID)
            c.create_text(L - 4, Y(v), text=f"{v:.3g}", anchor="e", fill=INK2, font=("Segoe UI", 8))
        for i in range(7):
            t = tmin + i * (x1 - tmin) / 6
            c.create_text(X(t), H - 8, text=f"{t:.0f} s", fill=INK2, font=("Segoe UI", 8))
        rows = [r for r in self.rows if r["t"] >= tmin]
        learn = [r for r in rows if r["sim"] < 0]
        if shade and learn:
            c.create_rectangle(X(learn[0]["t"]), T, X(learn[-1]["t"]), H - B, fill="#e6e3f7", outline="")
        for v in hlines:
            c.create_line(L, Y(v), W - R, Y(v), fill=INK2, dash=(5, 3))
        step = max(1, len(rows) // 600)
        for key, col, filt in series:
            pts = [r for r in rows[::step] if filt(r)]
            if len(pts) > 1:
                c.create_line(*[v for r in pts for v in (X(r["t"]), Y(key(r)))], fill=col, width=2)

    def update(self):
        try:
            s = self.board.state(self.next)
        except Exception:
            self.root.after(300, self.update)
            return
        self.k += 1
        if s["packets"]:
            if s["packets"][0]["win"] < self.next - 5:
                self.rows = []
            self.rows += s["packets"]
            self.next = s["packets"][-1]["win"] + 1
            self.rows = self.rows[-6000:]
        p = s["last"]
        c = self.cv
        c.itemconfigure(self.lcd, text=self.lcd_text(p))
        st = p["state"] if p else 0
        learning = p is None or p["sim"] < 0
        warn, stop = st in (1, 2), st == 3
        on = {"G": not learning and st == 0, "Y": (self.k % 2 == 0) if learning else warn, "R": stop}
        for key, (item, col) in self.leds.items():
            c.itemconfigure(item, fill=col if on[key] else GRID)
        c.itemconfigure(self.buzz, fill=RED if (warn or stop) and self.k % 2 == 0 else "#222")
        if p:
            self.angle += p["rpm"] / 60.0 * 0.2 * 0.15        # slowed down so the rotation is visible
            for i, b in enumerate(self.blade):
                a = self.angle + i * 2 * math.pi / 3
                c.coords(b, 325, 165, 325 + 32 * math.cos(a), 165 + 32 * math.sin(a))
            c.itemconfigure(self.motor_lab, text=f"{p['duty'] * 100:.0f} % · {p['rpm']:.0f} rpm")
            v = self.vals
            v["state"].configure(text="LEARNING" if learning else STATES[st],
                                 fg=RED if stop else (AMBER if warn else GREEN))
            v["sim"].configure(text="learning" if p["sim"] < 0 else f"{p['sim']} %")
            v["speed"].configure(text=f"{p['duty'] * 100:.0f} %")
            v["rpm"].configure(text=f"{p['rpm']:.0f}")
            v["vib"].configure(text=f"{p['vib']:.3f} g")
            v["tm"].configure(text=f"{p['t_motor']:.1f} °C")
            v["dht"].configure(text=f"{p['t_amb']:.1f} °C · {p['rh']:.0f} %")
            v["cur"].configure(text=f"{p['current']:.2f} A")
        for key, lab in self.zl.items():
            val = s["counters"][key]
            lab.configure(text=f"{val:,}", fg=(RED if val else GREEN) if key in ("gaps", "lost_queue", "lost_late") else INK)
        c = s["counters"]
        if not c["fog_on"]:
            self.uplink.configure(text="Dashboard uplink: off", fg=INK2)
        elif c["fog_ok"]:
            self.uplink.configure(text=f"Dashboard uplink: connected as SIM-01 \u00b7 {c['fog_sent']:,} packets acknowledged "
                                       f"\u00b7 {c['fog_pending']} waiting \u00b7 dashboard: {self.board.fog_url.rsplit('/api', 1)[0]}", fg=GREEN)
        else:
            self.uplink.configure(text=f"Dashboard uplink: fog server not running \u00b7 {c['fog_pending']:,} packets kept "
                                       f"until it is (start: python ui/fog_server.py)", fg=AMBER)
        self.events.delete(0, "end")
        for e in reversed(s["events"]):
            self.events.insert("end", f"{e['t']:.0f} s  {e['text']}")

        self.plot(self.c_sim, [(lambda r: r["sim"], GREEN, lambda r: r["sim"] >= 0)], 0, 100, hlines=(90,))
        vmax = max([0.2] + [r["vib"] * 1.05 for r in self.rows[-1500:]])
        self.plot(self.c_vib, [(lambda r: r["vib"], BLUE, lambda r: True)], 0, vmax, hlines=(0.075, 0.15))
        self.plot(self.c_spd, [(lambda r: 100 * r["duty"], BLUE, lambda r: True)], 0, 100)
        self.root.after(200, self.update)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed (1 = real time)")
    ap.add_argument("--learn", type=float, default=60.0, help="on-device learning time (s)")
    ap.add_argument("--fog", default="http://localhost:8060/api/ingest",
                    help="fog server to send packets to (start python ui/fog_server.py); 'off' = do not send")
    a = ap.parse_args()
    root = tk.Tk()
    App(root, Board(a.learn, a.speed, None if a.fog == 'off' else a.fog))
    root.mainloop()


if __name__ == "__main__":
    main()
