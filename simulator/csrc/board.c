/* ---------------------------------------------------------------------------
 * board.c - virtual ESP32 for the sample project, run on the PC.
 *
 * The same edge code as the Wokwi/ESP32 firmware (wokwi/src/edge_twin.c,
 * ms_core.c, features.c, detector.c) on two threads pinned to two CPU cores:
 *
 *   "Core 0"  1 kHz acquisition: motor twin + sensors -> sequence-numbered tick
 *             -> lock-free queue
 *   "Core 1"  windows (256, hop 128), 18 features, multi-input Z-score model,
 *             warning -> step the speed down -> stop
 *
 * Python (board_server.py) sets the sensor inputs (sliders) and reads the
 * per-window packets through the functions at the bottom.
 * ------------------------------------------------------------------------- */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "edge_twin.h"

#define QCAP 4096                         /* tick queue Core 0 -> Core 1 (power of 2) */
#define PCAP 1024                         /* packet queue Core 1 -> Python            */
#define FIFO_MS 85                        /* MPU6050 FIFO depth at 1 kHz              */

static ms_tick_t Q[QCAP];
static _Atomic uint32_t qh, qt;
static twin_packet_t PK[PCAP];
static _Atomic uint32_t ph, pt;

static volatile int running;
static volatile float duty_shared = 1.0f;
static volatile float speed = 1.0f;
static volatile twin_inputs_t IN;
static _Atomic uint64_t lost_queue, lost_late, pkt_dropped, ticks;
static pthread_t th0, th1;

static uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}
static void sleep_until(uint64_t ns)
{
    struct timespec t = {(time_t)(ns / 1000000000ull), (long)(ns % 1000000000ull)};
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);
}
static void pin(int cpu)
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu % (int)sysconf(_SC_NPROCESSORS_ONLN), &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

static void *core0(void *a)
{
    (void)a;
    pin(0);
    uint64_t t0 = now_ns(), n = 0;
    while (running) {
        uint64_t period = (uint64_t)(1000000.0f / speed);
        uint64_t target = t0 + n * period;
        sleep_until(target);
        uint64_t late = now_ns() - target;
        if (late > FIFO_MS * period) {               /* FIFO would have overflowed */
            uint32_t k = (uint32_t)(late / period);
            twin_plant_skip(k);
            n += k;
            atomic_fetch_add(&lost_late, k);
        }
        twin_inputs_t in;
        memcpy(&in, (const void *)&IN, sizeof in);
        ms_tick_t tk;
        twin_plant_tick(&in, duty_shared, &tk);
        n++;
        atomic_fetch_add(&ticks, 1);
        uint32_t h = atomic_load(&qh), t = atomic_load(&qt);
        if (h - t >= QCAP) { atomic_fetch_add(&lost_queue, 1); continue; }
        Q[h & (QCAP - 1)] = tk;
        atomic_store(&qh, h + 1);
    }
    return NULL;
}

static void *core1(void *a)
{
    (void)a;
    pin(1);
    float duty = 1.0f;
    while (running) {
        uint32_t t = atomic_load(&qt), h = atomic_load(&qh);
        if (t == h) { usleep(500); continue; }
        ms_tick_t tk = Q[t & (QCAP - 1)];
        atomic_store(&qt, t + 1);
        twin_packet_t p;
        if (!twin_process(&tk, &duty, &p)) continue;
        duty_shared = duty;
        uint32_t w = atomic_load(&ph), r = atomic_load(&pt);
        if (w - r >= PCAP) { atomic_fetch_add(&pkt_dropped, 1); continue; }
        PK[w & (PCAP - 1)] = p;
        atomic_store(&ph, w + 1);
    }
    return NULL;
}

/* ------------------------------------------------------------ Python API -- */
void board_set_inputs(int fault_type, float sev, float motor_c, float amb_c, float rh)
{
    IN.fault_type = fault_type;
    IN.fault_sev = sev;
    IN.have_mpu_temp = 1;
    IN.mpu_temp_raw = (int16_t)((motor_c - 36.53f) * 340.0f);   /* MPU6050 TEMP_OUT */
    IN.have_dht = 1;
    IN.dht_t = amb_c;
    IN.dht_rh = rh;
}

int board_start(float learn_s, float spd)
{
    if (running) return 1;
    twin_init(learn_s, 3.0f, 3.0f, 7);
    qh = qt = ph = pt = 0;
    lost_queue = lost_late = pkt_dropped = ticks = 0;
    duty_shared = 1.0f;
    speed = spd > 0 ? spd : 1.0f;
    running = 1;
    pthread_create(&th0, NULL, core0, NULL);
    pthread_create(&th1, NULL, core1, NULL);
    return 0;
}

void board_stop(void)
{
    if (!running) return;
    running = 0;
    pthread_join(th0, NULL);
    pthread_join(th1, NULL);
}

void board_restart(void) { twin_request_restart(); }

int board_packets(twin_packet_t *out, int max)
{
    int n = 0;
    while (n < max) {
        uint32_t r = atomic_load(&pt), w = atomic_load(&ph);
        if (r == w) break;
        out[n++] = PK[r & (PCAP - 1)];
        atomic_store(&pt, r + 1);
    }
    return n;
}

void board_counters(uint64_t out[5])
{
    out[0] = ticks; out[1] = lost_queue; out[2] = lost_late; out[3] = pkt_dropped;
    out[4] = atomic_load(&qh) - atomic_load(&qt);
}

int board_sizeof_packet(void) { return (int)sizeof(twin_packet_t); }
