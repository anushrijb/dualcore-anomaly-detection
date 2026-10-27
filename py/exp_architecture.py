"""
Objective 2 (architecture justification): numbers behind the hardware choice.

  * ESP32 footprint of the edge code: cross-compiled with the Espressif Xtensa
    GCC (xtensa-esp32-elf-gcc -Os) when it is installed; sizes from the object
    sections of the functions the edge node actually runs
  * RAM budget of the edge node (buffers, model, stacks)
  * I/O data rates: sensor stream vs the base paper's 9600-baud UART vs I2C vs
    what the edge sends over Wi-Fi after local processing
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from config import CSRC, BUILD, RESULTS, FS_HZ, WIN, HOP, N_FEAT, I2C_READ_US
import cbind

XTENSA_CANDIDATES = [shutil.which("xtensa-esp32-elf-gcc"),
                     *[str(p) for p in Path.home().glob("tools/xtensa/*/bin/xtensa-esp32-elf-gcc")],
                     *[str(p) for p in Path("/home/claude/tools/xtensa").glob("*/bin/xtensa-esp32-elf-gcc")],
                     *[str(p) for p in Path.home().glob(".espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc")]]
EDGE_FUNCS = ("fft_init", "fft_radix2", "features_window", "features_window_raw", "mpu6050_to_phys",
              "det_init", "det_learn", "det_finalize", "det_distance", "det_similarity", "det_model_bytes",
              "HANN", "TW_C", "TW_S", "g_fft_ready", "BAND_EDGE", "AXIS")


def find_xtensa():
    for c in XTENSA_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


def esp32_footprint():
    gcc = find_xtensa()
    if not gcc:
        return None
    size = gcc[:-3] + "size"
    if not Path(size).exists():
        size = str(Path(gcc).with_name("xtensa-esp-elf-size"))
    out = BUILD / "esp32"
    out.mkdir(exist_ok=True)
    sec = {}
    for f in ("features", "detector", "mpu6050_sim"):
        subprocess.run([gcc, "-Os", "-mlongcalls", "-ffunction-sections", "-fdata-sections",
                        "-DFFT_MAX=256", "-c", str(CSRC / f"{f}.c"), "-o", str(out / f"{f}.o")], check=True)
        txt = subprocess.run([size, "-A", str(out / f"{f}.o")], capture_output=True, text=True, check=True).stdout
        for line in txt.splitlines():
            p = line.split()
            if len(p) >= 2 and p[0].startswith(".") and p[1].isdigit():
                name = p[0]
                if any(name.endswith("." + fn) or name.endswith("." + fn + "$0") or ("." + fn + "$") in name
                       for fn in EDGE_FUNCS) and f != "mpu6050_sim" or (f == "mpu6050_sim" and "to_phys" in name):
                    kind = name.split(".")[1]
                    sec[kind] = sec.get(kind, 0) + int(p[1])
    flash = sec.get("text", 0) + sec.get("literal", 0) + sec.get("rodata", 0) + sec.get("data", 0)
    ram = sec.get("bss", 0) + sec.get("data", 0)
    return {"compiler": Path(gcc).name, "sections": sec, "flash_bytes": flash, "static_ram_bytes": ram}


def run(verbose=True):
    det = cbind.Detector(1, N_FEAT, 0.0, 1.0)
    rb_item = 7 * 2 + 2 + 4            # 7 registers + pad + 32-bit timestamp on ESP32
    ram = {
        "Sensor queue Core 0 -> Core 1 (512 samples)": 512 * rb_item,
        "Window buffer (256 x 7 float)": WIN * 7 * 4,
        "Detector state (mean, scale, co-moment, Cholesky, 31 features)": int(cbind.lib().api_det_sizeof()),
        "Core 1 task stack (window copy + FFT scratch)": 12 * 1024,
        "Core 0 task stack": 4 * 1024,
    }
    fp = esp32_footprint()
    if fp:
        ram["Static tables (FFT twiddles, Hann window)"] = fp["static_ram_bytes"]
    ram_total = sum(ram.values())

    sample_bytes = 14
    rates = [
        ("MPU6050 raw stream (14 B x 1 kHz)", sample_bytes * FS_HZ),
        ("Base paper UART link capacity (9600 baud, 8N1)", 9600 / 10),
        ("I2C capacity at 400 kHz (9 bits per byte)", 400e3 / 9),
        ("Edge to cloud after local processing (~120 B JSON per 128 ms decision)", 120 / (HOP / FS_HZ)),
    ]
    res = {
        "esp32_code": fp,
        "ram_budget_bytes": ram,
        "ram_total_bytes": ram_total,
        "esp32_sram_bytes": 520 * 1024,
        "ram_share_of_esp32_sram_pct": 100 * ram_total / (520 * 1024),
        "learned_model_bytes": det.model_bytes(),
        "data_rates_bytes_per_s": rates,
        "i2c_read_us": float(cbind.lib().api_i2c_us()),
        "i2c_core0_load_at_1khz_pct": float(cbind.lib().api_i2c_us()) * FS_HZ / 1e4,
        "uplink_reduction_x": sample_bytes * FS_HZ / (120 / (HOP / FS_HZ)),
    }
    (RESULTS / "o2_architecture.json").write_text(json.dumps(res, indent=2))
    if verbose:
        print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
