/* ---------------------------------------------------------------------------
 * ringbuf.h - lock-free single-producer / single-consumer ring buffer.
 * Plays the role of the FreeRTOS queue / stream buffer between ESP32 Core 0
 * (producer: sensor acquisition) and Core 1 (consumer: DSP + detection).
 * One writer and one reader, so only acquire/release ordering is needed.
 * ------------------------------------------------------------------------- */
#ifndef RINGBUF_H
#define RINGBUF_H
#include <stdatomic.h>
#include <stdint.h>
#include "mpu6050_sim.h"

#define RB_CAP 4096                     /* power of two */

typedef struct {
    int16_t  raw[MPU_REGS];
    uint64_t t_target_ns;               /* when the sensor produced the sample */
} rb_item_t;

typedef struct {
    _Alignas(64) _Atomic uint32_t head; /* written by producer */
    _Alignas(64) _Atomic uint32_t tail; /* written by consumer */
    _Alignas(64) rb_item_t buf[RB_CAP];
} ringbuf_t;

static inline int rb_push(ringbuf_t *r, const rb_item_t *it)
{
    uint32_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    uint32_t t = atomic_load_explicit(&r->tail, memory_order_acquire);
    if (h - t >= RB_CAP) return 0;                                   /* full */
    r->buf[h & (RB_CAP - 1)] = *it;
    atomic_store_explicit(&r->head, h + 1, memory_order_release);
    return 1;
}

static inline int rb_pop(ringbuf_t *r, rb_item_t *it)
{
    uint32_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    uint32_t h = atomic_load_explicit(&r->head, memory_order_acquire);
    if (t == h) return 0;                                            /* empty */
    *it = r->buf[t & (RB_CAP - 1)];
    atomic_store_explicit(&r->tail, t + 1, memory_order_release);
    return 1;
}

static inline uint32_t rb_count(ringbuf_t *r)
{
    return atomic_load_explicit(&r->head, memory_order_acquire) -
           atomic_load_explicit(&r->tail, memory_order_acquire);
}
#endif
