# Step-by-step guide

Project: **Dual-Core Real-Time Anomaly Detection for IoT Sensors** (ESP32 + MPU6050). This is the
proposed work plus the Review 1 remarks, software only.

Everything happens in **Ubuntu (WSL)** and **VS Code**. Do the steps in order.

---

## Part 1: Set up (only once)

**Step 1. Start fresh.** Open Ubuntu and delete the old copies:
```bash
cd ~ && rm -rf dualcore_anomaly tk_simulator wokwi_addon.zip dualcore_virtual_board.zip tk_simulator.zip
```

**Step 2. Copy the zip from Windows and unzip it.** Replace `YOURNAME` with your Windows user name.
If you're not sure of it, `ls /mnt/c/Users/` shows it.
```bash
cp /mnt/c/Users/YOURNAME/Downloads/dualcore_anomaly_final.zip ~
unzip -o dualcore_anomaly_final.zip
```

**Step 3. Install the tools:**
```bash
sudo apt update
sudo apt install -y gcc python3-venv python3-tk
```

**Step 4. Create the Python environment and install the packages:**
```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -r ~/dualcore_anomaly/requirements.txt
```

**Step 5. Open the project in VS Code:**
```bash
code ~/dualcore_anomaly
```
From now on, run every command in VS Code's terminal (**Terminal → New Terminal**).

---

## Part 2: Every time you open a new terminal

```bash
cd ~/dualcore_anomaly
source ~/venv/bin/activate
```
You should see `(venv)` at the start of the line.

---

## Part 3: See the results (nothing to run)

All results are already in the zip:

| File | What it is |
|---|---|
| `results/RESULTS.md` | the full report. Right-click it in VS Code and choose **Open Preview** |
| `results/figures/` | every chart (click a `.png` to view it) |
| `results/dashboard_snapshot.html` | the dashboard. Right-click it, choose **Reveal in Explorer**, then double-click it |
| `results/r3_maintenance_reports.md` | GenAI maintenance reports |

---

## Part 4: Re-run all experiments (optional)

```bash
python run_all.py --quick
```
This takes about 10–15 minutes and overwrites `results/` with numbers measured on **your** laptop.
Without `--quick` it takes about 20 minutes and does the full-size runs. Each stage prints a
`=====` header. When it's finished it prints `Done in ...s`.

---

## Part 5: The live dashboard (fog/cloud UI)

```bash
python ui/fog_server.py
```
Open **http://localhost:8060** in your browser. Four simulated machines (FAN-03, FAN-04, PUMP-01, PUMP-02) replay
a 25-minute run, and three of them develop faults. Any live board that is running, such as the simulator in Part 6,
appears as an extra card. Press **Ctrl + C** to stop.

---

## Part 6: The simulator, connected to the dashboard

Keep the dashboard from Part 5 running. Open a **second terminal** (click **+** in the terminal panel), run Part 2 in
it, then run:
```bash
python simulator/board_tk.py --speed 2
```
A window opens (the first run takes a few seconds to compile). It contains:
- **left side:** the board (LCD, NORMAL / WARNING / STOP LEDs, buzzer, motor), the sliders and buttons;
- **right side:** live values, charts and the zero-sample-loss counters. The line at the bottom says
  `Dashboard uplink: connected as SIM-01`.

In the browser, a **SIM-01** card appears. Click it to see its vibration, temperature, similarity and motor-speed
charts update live while you move the sliders. Alarms appear in the alarm log.

**Demo:**
1. Wait until the LCD shows `NORMAL  sim100%` (about 30 s).
2. **Fault severity** → 40: warning, buzzer, and the speed drops to 75 % (also shown on the dashboard).
3. **RESTART motor**, then **Healthy sliders**: back to normal.
4. **Fault type** → bearing, **severity** → 100: the motor stops, and SIM-01 shows **Stopped**.
5. **RESTART** + **Healthy**, then **Room temperature** → 47: the motor stops.
6. **Motor temperature** → 55: warning; → 61: stop.

The simulator also works without the dashboard; its uplink line then says the fog server is not running and keeps
the packets until it is (store-and-forward). Close the window to stop it.

---

## Part 7: Check the code (optional)

```bash
python tests/test_edge_equivalence.py
```
It should print `PASS`.

---

## Where each part of the project is

| Requirement | Where |
|---|---|
| Objective 1: functionalities | `csrc/` (MPU6050, features, detector), `results/figures/o1_*` |
| Objective 2: architecture justification | `py/exp_architecture.py`, `results/o2_architecture.json` |
| Objective 3: dual-core (TLP) | `csrc/edge_dualcore.c`, `results/figures/o3_edge_dualcore.png` |
| Objective 3: OpenMP threads, schedule/chunk, DLP, ILP | `csrc/fleet_omp.c`, `results/figures/o3_*` |
| Objective 3: edge / fog / cloud + UI | `ui/fog_server.py`, `ui/dashboard.html` (Part 5) |
| Objective 4: performance metrics | `results/o4_*`, `results/RESULTS.md` |
| Remark: multi-input Z-score (temperature, vibration, humidity, RPM, sound, current) | `csrc/ms_core.c`, `results/figures/r1_*` |
| Remark: zero sample loss and how to measure it | `csrc/edge_ms.c`, `results/r2_zero_loss.csv`, simulator counters |
| Remark: AI/GenAI, self-optimising | `py/exp_selfopt.py`, `results/r3_*` |
| Remark: sample project (warn → slow down → stop) | `py/exp_ms_control.py`, `results/figures/r6_*`, simulator (Part 6) |
| Real ESP32 build (future hardware) | `esp32/sample_project/sample_project.ino` |
| AI vision, engine-sound case study | the write-up document |

## If something goes wrong

| Message | Fix |
|---|---|
| `ModuleNotFoundError: numpy` | you forgot Part 2 (`source ~/venv/bin/activate`) |
| `No module named tkinter` | `sudo apt install -y python3-tk` |
| `gcc: not found` | `sudo apt install -y gcc` |
| `Address already in use` | a server is still running in another terminal: press Ctrl + C there |
| `No such file or directory` | you're not in the project folder: `cd ~/dualcore_anomaly` |
