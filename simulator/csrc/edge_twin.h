/* ---------------------------------------------------------------------------
 * edge_twin.h - the Wokwi build's edge logic (plain C, no Arduino calls)
 *
 * Wokwi has no DC motor, L298N, Hall sensor, current sensor or microphone,
 * and its MPU6050 returns fixed slider values instead of a vibration waveform.
 * So the motor and those sensors run as a "digital twin" inside the ESP32
 * (ms_plant_*, the same model as the PC simulation), while the parts Wokwi
 * does have are read for real:
 *
 *   MPU6050 (I2C, 1 kHz)  : temperature register -> motor temperature
 *   DHT22                 : ambient temperature + humidity
 *   slide potentiometer   : fault severity (0 = healthy)
 *   push button           : which mechanical fault the motor has
 *
 * The same file is compiled on the PC by test/test_twin.c, so the behaviour
 * can be checked without Wokwi.
 * ------------------------------------------------------------------------- */
#ifndef EDGE_TWIN_H
#define EDGE_TWIN_H
#include <stdint.h>
#include "ms_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {                 /* real (Wokwi) sensor readings, Core 0 */
    int     fault_type;          /* MSF_IMBALANCE .. MSF_OVERLOAD        */
    float   fault_sev;           /* 0..1 from the slide potentiometer    */
    int     have_mpu_temp;
    int16_t mpu_temp_raw;        /* MPU6050 TEMP_OUT register             */
    int     have_dht;
    float   dht_t, dht_rh;
} twin_inputs_t;

typedef struct {                 /* one packet per 128 ms window -> fog  */
    uint32_t win, seq_last, n_new;
    int      sim;                /* -1 while learning                    */
    int      state;              /* ST_NORMAL .. ST_STOPPED              */
    float    duty, vib, t_motor, t_amb, rh, rpm, current;
    uint32_t reason;
    uint64_t gaps;
} twin_packet_t;

typedef struct {                 /* what the LCD / LEDs show             */
    int   learning, sim, state;
    float t, duty, vib, t_motor, t_amb, rh, rpm, current;
    uint32_t reason;
} twin_status_t;

void  twin_init(float learn_s, float center, float sens, uint64_t seed);
/* Core 0, every 1 ms: advance the motor twin and fill one sensor tick */
void  twin_plant_tick(const twin_inputs_t *in, float duty, ms_tick_t *out);
/* Core 0 fell more than the MPU6050 FIFO (85 ms) behind: k samples are lost */
void  twin_plant_skip(uint32_t k);
/* Core 1: windowing + features + Z-score model + controller.
   Returns 1 when a window was finished (packet filled). *duty = new L298N duty. */
int   twin_process(const ms_tick_t *tk, float *duty, twin_packet_t *pkt);
/* restart button: clear the warning/stop and run at 100 % again
   (the learned model and healthy references are kept) */
void  twin_request_restart(void);
void  twin_status(twin_status_t *s);
float twin_learn_s(void);
extern const char *TWIN_FAULT_NAMES[MSF_N];

#ifdef __cplusplus
}
#endif
#endif
