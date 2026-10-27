# ESP32 sample project (Review 1)

This sketch is ready to flash once you have the parts. It uses the same C files (`ms_core`, `features`,
`detector`, `mpu6050_sim`) that the PC simulation uses and tests.

## Parts and wiring

| Part | Pin(s) on the ESP32 Dev Module | Notes |
|---|---|---|
| MPU6050 | SDA 21, SCL 22, 3V3, GND | I2C at 400 kHz, address 0x68 |
| A3144 Hall sensor + 1 magnet on the shaft | OUT -> GPIO 18 | open collector; internal pull-up is used |
| DHT22 | DATA -> GPIO 4 | add a 10 kOhm pull-up if your module has none |
| ACS712-05B | OUT -> GPIO 35 through a divider (5 V -> 3.3 V) | ADC1 pin, so it still works with Wi-Fi on |
| INMP441 microphone | SCK 14, WS 15, SD 32, L/R -> GND | 8 kHz, I2S |
| L298N | ENA 25 (PWM), IN1 26, IN2 27 | motor supply on the L298N's 12 V input; common GND |
| Buzzer, LED | 13, 2 | LED 2 is the on-board LED |

## Before you flash

1. Install Arduino-ESP32 core 3.x, plus the libraries "DHT sensor library" and "Adafruit Unified Sensor".
2. Set `WIFI_SSID`, `WIFI_PASS`, `FOG_URL` (your PC's IP address) and `RPM_NOM` (your motor's speed
   at full PWM).
3. Copy `CENTER` from `results/r1_params.json` → `"all 6 sensors (Z-score)"`, or keep 3.0.
4. On the PC, run `python ui/fog_server.py --ingest`. It prints the zero-loss counters for every
   packet it receives.

## What happens after you flash it

- **Learning:** the motor runs at 100 % for 300 s while the board learns what healthy looks like.
- **Warning:** buzzer and LED come on, the speed steps down 75 → 50 → 25 %, and the motor stops on
  hard limits (`ms_ctrl_step`).
- **Zero loss:** the serial monitor prints `gaps`, `lost_queue`, `lost_late` and `sf_dropped`.
  All four should stay at 0.

This sketch was syntax-checked against stub Arduino headers, and its C files were compiled with
`xtensa-esp32-elf-gcc`. It has not been flashed to a board here.
