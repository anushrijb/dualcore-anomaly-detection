# Tkinter simulator: dual-core edge AI motor protection

A virtual ESP32 board (sample project: MPU6050, Hall RPM, DHT22, ACS712 current, microphone, DC motor
with an L298N driver), in a desktop window. It needs only Python, Tk and gcc; no pip packages.

## Run it

```bash
sudo apt install -y python3-tk gcc        # once
cd ~/dualcore_anomaly
python3 simulator/board_tk.py --speed 2
```

`--speed 2` runs twice as fast, so learning takes 30 s instead of 60 s. Leave it off to run in real time.

## Connected to the dashboard

Start the dashboard first in another terminal (`cd ~/dualcore_anomaly && python ui/fog_server.py`). The simulator
then sends every decision to it, as an ESP32 would over Wi-Fi, and appears live at http://localhost:8060 as
**SIM-01** (speed, controller state, similarity, vibration, temperatures, alarms). Packets stay buffered until the
dashboard acknowledges them. Use `--fog off` to run without sending.

## How it works

- `csrc/` holds the edge code: the same C files as the ESP32 firmware. They are compiled
  automatically the first time you run it.
- **Core 0** (a thread pinned to CPU 0) samples the sensors at 1 kHz and stamps every sample with a
  sequence number.
- **Core 1** (a thread pinned to CPU 1) works on 256-sample windows: 18 features, the multi-input
  Z-score model, and the controller (warning → step the speed down 25 % every 10 s → stop).
- The window: sliders = sensors and faults; LCD, LEDs, buzzer and motor = board outputs; charts and
  zero-sample-loss counters on the right.

## Demo

1. Wait until the LCD shows `NORMAL  sim100%`.
2. Fault severity 40 → warning, and the speed drops to 75 %.
3. RESTART motor + Healthy sliders → back to normal.
4. Fault type **bearing**, severity 100 → the motor stops.
5. RESTART + Healthy, then Room temperature 47 °C → the motor stops (limit 45 °C).
6. Motor temperature 55 °C → warning; 61 °C → stop. Humidity 85 % → warning; 91 % → stop.
