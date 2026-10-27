# Generated maintenance reports

One AI-controlled run per fault type. Evidence is computed on the device; the text is written from it (template offline, LLM when ANTHROPIC_API_KEY is set).

### Work order - MOTOR-01
- **Alarm**: similarity fell to 88 % at 10.95 min.
- **Evidence**: vib_ay_rms +8.5 sigma; vib_ax_rms +8.4 sigma; vib_gx_rms +7.7 sigma; vib_band_lo +6.8 sigma
- **Sensors involved**: vibration (MPU6050).
- **Likely cause**: imbalance (signature match 0.98).
- **Automatic actions**: 75 % at 12.4 min, 50 % at 23.78 min.
- **Recommended**: Check rotor/coupling for mass imbalance or build-up; perform field balancing.

### Work order - MOTOR-02
- **Alarm**: similarity fell to 88 % at 10.57 min.
- **Evidence**: vib_band_hi +13.9 sigma; vib_ax_kurt +8.7 sigma; vib_ax_rms +5.0 sigma; vib_az_rms +3.9 sigma
- **Sensors involved**: vibration (MPU6050).
- **Likely cause**: bearing (signature match 0.62).
- **Automatic actions**: 75 % at 12.33 min, 50 % at 15.8 min.
- **Recommended**: Inspect the drive-end bearing (outer race) for wear and lubrication; plan replacement.

### Work order - MOTOR-03
- **Alarm**: similarity fell to 66 % at 10.96 min.
- **Evidence**: vib_az_rms +8.7 sigma; vib_band_mid +8.4 sigma; vib_ax_kurt +4.4 sigma; vib_ax_rms +1.9 sigma
- **Sensors involved**: vibration (MPU6050).
- **Likely cause**: looseness (signature match 0.85).
- **Automatic actions**: 75 % at 16.43 min.
- **Recommended**: Check mounting bolts, base and coupling for mechanical looseness; re-torque.

### Work order - MOTOR-04
- **Alarm**: similarity fell to 87 % at 11.89 min.
- **Evidence**: vib_band_mid +8.7 sigma; vib_ax_rms +3.4 sigma; vib_az_rms +2.5 sigma; vib_band_lo +1.6 sigma
- **Sensors involved**: vibration (MPU6050).
- **Likely cause**: looseness (signature match 0.89).
- **Automatic actions**: 75 % at 16.3 min, 50 % at 29.18 min.
- **Recommended**: Check mounting bolts, base and coupling for mechanical looseness; re-torque.

### Work order - MOTOR-05
- **Alarm**: similarity fell to 88 % at 11.08 min.
- **Evidence**: temp_motor +8.8 sigma; rpm_ratio -8.4 sigma; sound_db +6.7 sigma; current_mean +4.2 sigma
- **Sensors involved**: motor current (ACS712), sound (INMP441), speed (Hall sensor), temperature (MPU6050 die / DHT22).
- **Likely cause**: overload (signature match 0.90).
- **Automatic actions**: 75 % at 15.08 min, 50 % at 15.25 min, 25 % at 15.42 min.
- **Recommended**: Check the driven load for jamming or excess friction; verify current rating.

### Work order - MOTOR-06
- **Alarm**: similarity fell to 89 % at 14.37 min.
- **Evidence**: temp_motor +59.9 sigma; current_mean -1.3 sigma; rpm_ratio +1.3 sigma; vib_ay_rms -1.2 sigma
- **Sensors involved**: motor current (ACS712), speed (Hall sensor), temperature (MPU6050 die / DHT22), vibration (MPU6050).
- **Likely cause**: cooling (signature match 1.00).
- **Automatic actions**: none.
- **Recommended**: Check the cooling fan, vents and ambient airflow; clean dust from the housing.

### Work order - MOTOR-07
- **Alarm**: similarity fell to 83 % at 13.77 min.
- **Evidence**: humidity +13.7 sigma; rpm_ratio +1.4 sigma; current_mean -1.4 sigma; vib_ay_rms -1.3 sigma
- **Sensors involved**: humidity (DHT22), motor current (ACS712), speed (Hall sensor), vibration (MPU6050).
- **Likely cause**: humidity (signature match 0.95).
- **Automatic actions**: 0 % at 24.23 min; motor stopped at 24.23 min.
- **Recommended**: Check the enclosure for water ingress or condensation; dry and reseal.
