# Results - Dual-Core Real-Time Anomaly Detection for IoT Sensors

Produced by `python run_all.py`. Data are simulated (register-level MPU6050 on a motor model); the edge pipeline and the OpenMP kernels are real C code measured on this machine.

## Objective 1 + 4 - detection (test sessions: 6 healthy + 25 faulty motors)

Thresholds chosen on separate calibration sessions (best balanced accuracy with window specificity >= 99 %). Labels: healthy = before onset; anomaly = fault severity >= 0.2.

| method                            |   Balanced accuracy (%) |   Recall (%) |   Specificity (%) |   Precision (%) |   F1 (%) |   MCC (%) |   False alarms per hour |
|:----------------------------------|------------------------:|-------------:|------------------:|----------------:|---------:|----------:|------------------------:|
| Base paper: temperature + ZSM     |                   73.83 |        47.67 |            100    |          100    |    64.56 |     58.57 |                       0 |
| Vibration + ZSM (ablation)        |                   97.08 |        94.18 |             99.98 |           99.97 |    96.99 |     95.04 |                       0 |
| Proposed: vibration + Mahalanobis |                   97.52 |        95.04 |            100    |           99.99 |    97.45 |     95.78 |                       0 |

Median detection delay after fault onset (minutes; 5 motors per fault):

| fault        | base_temp_zsm   | static_50C     |   vib_maha |   vib_zsm |
|:-------------|:----------------|:---------------|-----------:|----------:|
| bearing      | 5.3             | - (5 missed)   |        0.5 |       0.5 |
| cooling      | 3.2             | 9.0 (2 missed) |        3.2 |       3.2 |
| imbalance    | 6.9             | - (5 missed)   |        2.3 |       2.6 |
| looseness    | 8.8 (4 missed)  | - (5 missed)   |        1.4 |       1.9 |
| misalignment | 5.1             | - (5 missed)   |        0.9 |       1.5 |

## Objective 2 - architecture numbers

- ESP32 code size of features + detector (xtensa-esp32-elf-gcc -Os): **2992 B flash**, 2052 B static RAM
- Edge RAM budget: **43.4 KB** = 8.3 % of the ESP32's 520 KB SRAM
- Learned model: 7952 B
- I2C burst read: 385 us -> 38.5 % of Core 0 at 1 kHz
- Uplink reduced 15x by deciding at the edge

| Link                                                                   |   Bytes per second |
|:-----------------------------------------------------------------------|-------------------:|
| MPU6050 raw stream (14 B x 1 kHz)                                      |            14000   |
| Base paper UART link capacity (9600 baud, 8N1)                         |              960   |
| I2C capacity at 400 kHz (9 bits per byte)                              |            44444.4 |
| Edge to cloud after local processing (~120 B JSON per 128 ms decision) |              937.5 |

## Objective 3a - dual-core vs single-core edge (1 kHz, real time)

| mode   |   processing per window (us) |   sample lateness p99 (us) |   samples lost (%) |   decision latency p99 (us) |   core 0 load |   core 1 load |
|:-------|-----------------------------:|---------------------------:|-------------------:|----------------------------:|--------------:|--------------:|
| single |                            0 |               30.95        |               0    |               480.04        |          0.39 |          0    |
| dual   |                            0 |               55.27        |               0    |               499.87        |          0.39 |          0    |
| single |                         1000 |              911.13        |               0    |              1397.07        |          0.4  |          0    |
| dual   |                         1000 |                2.62        |               0    |              1474.66        |          0.39 |          0.01 |
| single |                        10000 |             8772.92        |               0    |             10461           |          0.46 |          0    |
| dual   |                        10000 |              106.26        |               0    |             12175.3         |          0.39 |          0.08 |
| single |                        40000 |            59514.8         |               0.3  |             61081.2         |          0.7  |          0    |
| dual   |                        40000 |               14.88        |               0    |             41141.1         |          0.39 |          0.3  |
| single |                        70000 |            69386.7         |               0    |             73639           |          0.91 |          0    |
| dual   |                        70000 |             2590.68        |               0    |             75914.6         |          0.39 |          0.52 |
| single |                        90000 |           575643           |              88.35 |            600001           |          0.98 |          0    |
| dual   |                        90000 |               25.42        |               0    |             91323.4         |          0.39 |          0.68 |
| single |                       110000 |                1.46948e+06 |              93.73 |                 1.50425e+06 |          0.98 |          0    |
| dual   |                       110000 |              315.81        |               0    |            110782           |          0.39 |          0.83 |
| single |                       140000 |                2.8417e+06  |              95.08 |                 2.87895e+06 |          0.98 |          0    |
| dual   |                       140000 |              935.62        |               0    |            668664           |          0.36 |          0.96 |

## Objective 3b - OpenMP fog/cloud

Dataset: 64 motors x 512 windows = 32768 windows, 117.4 MB raw; 7233 flagged (22.1 %).
Fault-type diagnosis on held-out motors: 100.0 % (n = 3224). Parallel vs serial results: max |diff| features 0, envelope 0, class models (relative) 6.27e-16.

| stage          |   threads |   seconds |   speedup |   efficiency |   karp_flatt |
|:---------------|----------:|----------:|----------:|-------------:|-------------:|
| S1_gen         |         1 |     5.164 |     1     |        1     |      nan     |
| S2_feat        |         1 |     0.268 |     1     |        1     |      nan     |
| S3_train       |         1 |     0.003 |     1     |        1     |      nan     |
| S4_score_simd  |         1 |     0.004 |     1     |        1     |      nan     |
| S5_deep        |         1 |     2.327 |     1     |        1     |      nan     |
| S5_deep_static |         1 |     2.445 |     1     |        1     |      nan     |
| S1_gen         |         2 |     2.627 |     1.966 |        0.983 |        0.017 |
| S2_feat        |         2 |     0.158 |     1.693 |        0.847 |        0.181 |
| S3_train       |         2 |     0.002 |     1.72  |        0.86  |        0.163 |
| S4_score_simd  |         2 |     0.002 |     1.866 |        0.933 |        0.072 |
| S5_deep        |         2 |     1.273 |     1.827 |        0.914 |        0.095 |
| S5_deep_static |         2 |     2.261 |     1.081 |        0.541 |        0.849 |

Schedule x chunk size (seconds):

|                        |   default |     1 |     4 |    16 |    64 |   512 |
|:-----------------------|----------:|------:|------:|------:|------:|------:|
| ('S2_feat', 'dynamic') |     0.147 | 0.152 | 0.151 | 0.156 | 0.142 | 0.142 |
| ('S2_feat', 'guided')  |     0.128 | 0.146 | 0.139 | 0.158 | 0.159 | 0.137 |
| ('S2_feat', 'static')  |     0.134 | 0.14  | 0.155 | 0.141 | 0.169 | 0.16  |
| ('S5_deep', 'dynamic') |     1.425 | 1.333 | 1.196 | 1.191 | 1.323 | 1.29  |
| ('S5_deep', 'guided')  |     1.326 | 1.231 | 1.176 | 1.141 | 1.304 | 1.392 |
| ('S5_deep', 'static')  |     2.352 | 1.288 | 1.422 | 1.322 | 1.333 | 1.273 |

DLP and ILP:

| experiment   |   threads | variant   |   seconds |   gflops |
|:-------------|----------:|:----------|----------:|---------:|
| dlp          |         1 | scalar    |    0.0054 |   7.4993 |
| dlp          |         1 | simd      |    0.0039 |  10.4804 |
| dlp          |         2 | scalar    |    0.0033 |  12.2706 |
| dlp          |         2 | simd      |    0.0019 |  21.4042 |
| ilp          |         1 | acc1      |    0.0643 |   2.5474 |
| ilp          |         1 | acc2      |    0.0606 |   2.7029 |
| ilp          |         1 | acc4      |    0.0346 |   4.7376 |
| ilp          |         1 | acc8      |    0.0303 |   5.4027 |
| ilp          |         1 | simd      |    0.0087 |  18.9004 |

## UI

Fog alarms in the dashboard snapshot:

- PUMP-02: alarm at 15.68 min (similarity 47 %)
- FAN-03: alarm at 16.05 min (similarity 39 %)
- FAN-04: alarm at 17.97 min (similarity 89 %)

## Review 1 remarks

### Multi-input one-class Z-score model (sensor ablation, 7 faults, test motors)

| sensor set                  |   features |   balanced accuracy (%) |   recall (%) |   specificity (%) |   false alarms per hour |   faults detected (%) |
|:----------------------------|-----------:|------------------------:|-------------:|------------------:|------------------------:|----------------------:|
| temperature                 |          2 |                   76.27 |        52.74 |             99.8  |                    2.55 |                 57.14 |
| humidity                    |          1 |                   57.14 |        14.42 |             99.86 |                    2.55 |                 14.29 |
| rpm                         |          1 |                   56.95 |        13.9  |            100    |                    0    |                 14.29 |
| current                     |          2 |                   57.69 |        15.47 |             99.91 |                    0    |                 21.43 |
| sound                       |          3 |                   77.51 |        55.13 |             99.89 |                    0    |                 71.43 |
| vibration                   |          9 |                   84.91 |        69.85 |             99.97 |                    0    |                 71.43 |
| all 6 sensors (Z-score)     |         18 |                   97.3  |        94.63 |             99.96 |                    0    |                100    |
| all 6 sensors (Mahalanobis) |         18 |                  100    |       100    |            100    |                    0    |                100    |

Median minutes to a sustained alarm (blank = never):

| fault        |   all 6 sensors (Mahalanobis) |   all 6 sensors (Z-score) |   current |   humidity |   rpm |   sound |   temperature |   vibration |
|:-------------|------------------------------:|--------------------------:|----------:|-----------:|------:|--------:|--------------:|------------:|
| bearing      |                           0.7 |                       0.5 |       8.1 |      nan   | nan   |     1.8 |           3.4 |         0.5 |
| cooling      |                           1.1 |                       4.4 |     nan   |      nan   | nan   |   nan   |           1.7 |       nan   |
| humidity     |                           1.6 |                       3.9 |     nan   |        0.7 | nan   |   nan   |         nan   |       nan   |
| imbalance    |                           0.6 |                       0.8 |     nan   |      nan   | nan   |     1.4 |         nan   |         0.8 |
| looseness    |                           1.9 |                       1.8 |     nan   |      nan   | nan   |     6.7 |         nan   |         1.7 |
| misalignment |                           1   |                       0.8 |     nan   |      nan   | nan   |     6.4 |           4.5 |         0.7 |
| overload     |                           0.7 |                       1.2 |       2.1 |      nan   |   0.4 |     1.1 |           2   |         2.5 |

Fault-type diagnosis (signature match): 100.0 % of 71224 windows.

### Sample project: closed-loop control (3 motors per fault)

| fault        | mode            |   warning_min |   stopped |   final_severity |   speed_pct |   false_warn |
|:-------------|:----------------|--------------:|----------:|-----------------:|------------:|-------------:|
| bearing      | AI + thresholds |          0.58 |         0 |             0.47 |       55.98 |            0 |
| bearing      | thresholds only |          3.86 |         0 |             0.6  |       64.23 |            0 |
| cooling      | AI + thresholds |          4.42 |         1 |             1    |      100    |            0 |
| cooling      | thresholds only |         11.77 |         1 |             1    |      100    |            0 |
| humidity     | AI + thresholds |          3.65 |         1 |             1    |      100    |            0 |
| humidity     | thresholds only |         12.33 |         1 |             1    |      100    |            0 |
| imbalance    | AI + thresholds |          0.43 |         0 |             0.7  |       70.39 |            0 |
| imbalance    | thresholds only |          5.35 |         0 |             0.91 |       81.89 |            0 |
| looseness    | AI + thresholds |          1.79 |         0 |             0.95 |       83.72 |            0 |
| looseness    | thresholds only |          9.23 |         0 |             1    |       86.92 |            0 |
| misalignment | AI + thresholds |          0.82 |         0 |             0.93 |       82.75 |            0 |
| misalignment | thresholds only |         10.53 |         0 |             1    |       88.49 |            0 |
| normal       | AI + thresholds |        nan    |         0 |             0    |      100    |            0 |
| normal       | thresholds only |        nan    |         0 |             0    |      100    |            0 |
| overload     | AI + thresholds |          1.03 |         0 |             0.35 |       44.78 |            0 |
| overload     | thresholds only |          4.44 |         3 |             0.31 |       27.64 |            0 |

### Zero sample loss

| run                                                      |   samples_produced |   L1_sensor_fifo_lost |   L2_queue_lost |   L3_seq_gaps_seen_core1 |   L4_uplink_lost_packets |   fog_missing_windows |   fog_coverage_pct |   loss_rate_upper95_if_zero |   lateness_us_p99 |
|:---------------------------------------------------------|-------------------:|----------------------:|----------------:|-------------------------:|-------------------------:|----------------------:|-------------------:|----------------------------:|------------------:|
| A: real time 240 s, dual core, closed loop               |             240000 |                    63 |               0 |                       63 |                        0 |                     0 |            99.9737 |                    0.000262 |            141.49 |
| B: real time 30 s, 90 ms model, single core              |              30000 |                  3991 |               0 |                     3991 |                        0 |                     0 |            86.7406 |                    0.133    |          96079.3  |
| B: real time 30 s, 90 ms model, dual core                |              30000 |                     0 |               0 |                        0 |                        0 |                     0 |           100      |                    0.0001   |             38.07 |
| C: 6 h soak, 2 % loss + 3 outages, store-and-forward off |           21600000 |                     0 |               0 |                        0 |                     3751 |                  3751 |            97.7772 |                    1.39e-07 |              0    |
| C: 6 h soak, 2 % loss + 3 outages, store-and-forward on  |           21600000 |                     0 |               0 |                        0 |                        0 |                     0 |           100      |                    1.39e-07 |              0    |

### Self-optimising baseline (healthy drift +10 C, +20 %RH over 2 h)

| fault   | baseline        |   fa_per_h |   anomalous_pct |   delay_s |
|:--------|:----------------|-----------:|----------------:|----------:|
| bearing | self-optimising |       0    |            0.07 |     37.63 |
| bearing | static          |     141.81 |           61.12 |      0.13 |
| normal  | self-optimising |       0    |            0.07 |    nan    |
| normal  | static          |      67.83 |           81.41 |    nan    |

Generated maintenance reports: results/r3_maintenance_reports.md


Note: the real-time 240 s run above shared the 2-core machine with other experiments; one host
stall over 85 ms lost 63 samples (counted identically by L1 and L3). The same run alone:
results/r2_zero_loss_isolated_rt240.json (0 lost of 240 000, worst lateness 43 ms).
