/* ---------------------------------------------------------------------------
 * ms_core.h - multi-sensor "sample project" (Review 1 remarks)
 *
 *   ESP32 dev board driving a DC motor through an L298N (PWM duty), with
 *     MPU6050  vibration (accel + gyro) + die temperature, 1 kHz, I2C
 *     A3144    Hall-effect sensor, 1 magnet -> one pulse per revolution (RPM)
 *     DHT22    ambient temperature + humidity, one reading every 2 s
 *     ACS712   motor current, sampled with the MPU6050 (1 kHz)
 *     INMP441  I2S microphone, 8 kHz (8 audio samples per 1 ms tick)
 *
 * The plant is stateful (speed follows the PWM duty the controller sets), so
 * the controller's speed reductions really change the vibration, current,
 * sound and heat it sees - and slow down speed-dependent wear.
 * ------------------------------------------------------------------------- */
#ifndef MS_CORE_H
#define MS_CORE_H
#include <stdint.h>
#include "detector.h"

#define MS_FS        1000
#define MS_AUD       8                 /* audio samples per tick (8 kHz) */
#define MS_WIN       256
#define MS_HOP       128
#define MS_NF        18
#define MS_DHT_TICKS 2000              /* DHT22: one reading per 2 s */

enum ms_fault {
    MSF_NONE = 0, MSF_IMBALANCE, MSF_BEARING, MSF_MISALIGN, MSF_LOOSENESS,
    MSF_OVERLOAD, MSF_COOLING, MSF_HUMIDITY, MSF_N
};

typedef struct {
    uint64_t seed;
    float rpm_nom;                     /* speed at 100 % duty, healthy (rpm)   */
    float a1;                          /* healthy 1x vibration at nominal (g)  */
    float ambient_c, rh_pct;           /* starting ambient                     */
    float drift_c, drift_rh, drift_s;  /* healthy linear drift over drift_s    */
    float load_amp, load_period_s;
    int   fault;
    float onset_s, ramp_s;             /* ramp_s = time to full severity at 100 % speed */
} ms_cfg_t;

typedef struct {                       /* what the sensors deliver in one 1 ms tick */
    uint32_t seq;                      /* sample sequence number (zero-loss check) */
    int16_t  mpu[7];                   /* ax ay az temp gx gy gz registers          */
    float    current_a;                /* ACS712                                    */
    int16_t  audio[MS_AUD];            /* INMP441                                   */
    uint8_t  hall;                     /* 1 = Hall edge inside this tick            */
    float    hall_us;                  /* edge time within the tick (ISR micros())  */
    uint8_t  dht_new;
    float    dht_t, dht_rh;
} ms_tick_t;

typedef struct {
    ms_cfg_t c;
    uint64_t n;                        /* tick counter */
    double   theta, omega, duty, sev, t_motor, wear_clock;
    float    dht_t, dht_rh;
    uint64_t rng;
} ms_plant_t;

void  ms_cfg_default(ms_cfg_t *c, uint64_t seed);
void  ms_plant_init(ms_plant_t *p, const ms_cfg_t *c);
void  ms_plant_set_duty(ms_plant_t *p, float duty);       /* L298N ENA PWM, 0..1 */
void  ms_plant_step(ms_plant_t *p, ms_tick_t *out);       /* advance 1 ms */
float ms_plant_rpm(const ms_plant_t *p);

/* -------- window features (18) ------------------------------------------ */
typedef struct {
    int16_t  mpu[MS_WIN][7];
    float    cur[MS_WIN];
    int16_t  aud[MS_WIN * MS_AUD];
    double   hall_t[64];               /* Hall edge times (s) in the window */
    int      n_hall;
    float    dht_t, dht_rh, duty;
} ms_window_t;

void ms_features(const ms_window_t *w, float rpm_nom, float out[MS_NF]);

/* windower on Core 1: collects ticks, checks the sequence numbers, emits a
   256-tick window every 128 ticks. gaps = samples missing in the sequence. */
typedef struct {
    ms_tick_t ring[MS_WIN];
    uint32_t  head, fill, since, next_seq, started;
    uint64_t  gaps, received;
} ms_windower_t;
int  ms_windower_push(ms_windower_t *wd, const ms_tick_t *tk, float duty, ms_window_t *out);
/* raw (un-normalised) values the threshold rules use:
   [vib_rms_g, rpm_ratio, t_motor, t_amb, rh, current_a] */
void ms_raw_values(const ms_window_t *w, const float feat[MS_NF], float raw[6]);
extern const char *MS_FEATURE_NAMES[MS_NF];

/* sensor groups (for the "which input helps" ablation) */
enum { MSG_VIB = 1, MSG_TEMP = 2, MSG_HUM = 4, MSG_RPM = 8, MSG_CUR = 16, MSG_SND = 32, MSG_ALL = 63 };
extern const int MS_FEATURE_GROUP[MS_NF];

/* -------- staged protection controller ------------------------------------ */
enum { ST_NORMAL = 0, ST_WARNING, ST_DERATED, ST_STOPPED };
enum { CTRL_NONE = -1, CTRL_THRESHOLD = 0, CTRL_AI = 1 };

typedef struct {
    float vib_warn_g, vib_stop_g;      /* raw radial vibration RMS (g)          */
    float rpm_dev_warn;                /* |rpm / expected - 1|                  */
    float tmotor_warn_c, tmotor_stop_c;
    float tamb_warn_c, tamb_stop_c;    /* DHT22                                 */
    float rh_warn, rh_stop;
    float cur_warn_a, cur_stop_a;
    float step_s;                      /* time between speed steps              */
    float vib_ref, cur_ref;            /* healthy raw vibration / current at 100 %, learned */
    int   debounce;                    /* consecutive AI anomalies for warning  */
} ms_limits_t;

typedef struct {
    int   mode, state, streak, ok_streak, started;
    float duty, t_last_step, t_warn, t_stop;
    uint32_t reason;                   /* bit mask of what triggered (sensor groups, 64 = AI) */
    ms_limits_t L;
} ms_ctrl_t;

void ms_limits_default(ms_limits_t *L);
/* sensor-accuracy floors for the environment features + which features the
   self-optimising baseline may follow (ambient temperature, humidity, motor temp) */
void ms_detector_setup(detector_t *d);
/* learn healthy raw references during on-device learning */
void ms_ctrl_learn(ms_ctrl_t *c, const float raw[6]);
void ms_ctrl_init(ms_ctrl_t *c, int mode);
/* raw: [vib_rms_g, rpm_ratio, t_motor, t_amb, rh, current] ; sim: AI similarity (255 = n/a) */
float ms_ctrl_step(ms_ctrl_t *c, float t, const float raw[6], int sim);

#endif
