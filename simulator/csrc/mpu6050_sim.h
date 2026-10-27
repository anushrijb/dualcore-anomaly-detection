/* ---------------------------------------------------------------------------
 * mpu6050_sim.h - register-level MPU6050 simulator mounted on an AC motor.
 *
 * Produces exactly what an ESP32 gets from one 14-byte I2C burst read starting
 * at register 0x3B (ACCEL_XOUT_H): accel X/Y/Z, TEMP, gyro X/Y/Z as int16.
 *   accel  : +-2 g      -> 16384 LSB/g
 *   gyro   : +-250 dps  -> 131 LSB/(deg/s)
 *   temp   : T(degC) = raw/340 + 36.53
 *
 * Every sample is a closed-form function of (motor config, sample index), so
 * any window can be generated independently - this is what lets the OpenMP
 * fog/cloud program build a huge dataset in parallel.
 * ------------------------------------------------------------------------- */
#ifndef MPU6050_SIM_H
#define MPU6050_SIM_H
#include <stdint.h>

#define MPU_ACCEL_LSB_PER_G   16384.0f
#define MPU_GYRO_LSB_PER_DPS  131.0f
#define MPU_TEMP_LSB_PER_C    340.0f
#define MPU_TEMP_OFFSET_C     36.53f
#define MPU_REGS              7        /* ax ay az temp gx gy gz */
#define MPU_FIFO_SAMPLES      85       /* 1024-byte FIFO / 12 bytes per accel+gyro sample */

enum fault_type {
    FAULT_NONE = 0,
    FAULT_IMBALANCE = 1,     /* 1x running-speed vibration grows             */
    FAULT_MISALIGNMENT = 2,  /* 2x radial + axial 1x/2x                       */
    FAULT_BEARING = 3,       /* outer-race impulses at BPFO exciting resonance */
    FAULT_LOOSENESS = 4,     /* many harmonics, sub-harmonic, rectified shape */
    FAULT_COOLING = 5,       /* cooling failure: heat only, no vibration change */
    FAULT_N = 6
};

typedef struct {
    uint64_t seed;
    float fs;              /* sample rate (Hz)                                  */
    float rpm;             /* nominal shaft speed                               */
    float speed_drift;     /* fractional slow speed variation (e.g. 0.01)       */
    float load_amp;        /* fractional load swing modulating 1x amplitude     */
    float load_period_s;
    float a1;              /* healthy 1x radial amplitude (g)                   */
    float case_temp_c;     /* healthy motor case temperature (degC)             */
    float ambient_amp;     /* slow ambient swing (degC)                         */
    float ambient_period_s;
    int   fault;           /* enum fault_type                                   */
    float onset_s;         /* fault onset time                                  */
    float ramp_s;          /* time for severity 0 -> 1                          */
    float tau_s;           /* thermal time constant                             */
} motor_cfg_t;

void  motor_default(motor_cfg_t *c, uint64_t seed);
float motor_severity(const motor_cfg_t *c, double t);
float motor_case_temp(const motor_cfg_t *c, double t);   /* true case temperature */

/* One 14-byte burst read at sample index n (time n/fs). */
void  mpu6050_read(const motor_cfg_t *c, uint64_t n, int16_t raw[MPU_REGS]);

/* Raw register values -> physical units: g, degC, deg/s (same order). */
void  mpu6050_to_phys(const int16_t raw[MPU_REGS], float out[MPU_REGS]);

/* I2C 400 kHz: 14 data bytes + address/register bytes, 9 bits each. */
float mpu6050_i2c_read_us(void);

#endif
