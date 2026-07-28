/*
 * avr_bench - cycle counts per softmax call on an ATmega328P.
 *
 * Runs under simavr, which is instruction-accurate, so the numbers match
 * what a real Uno executes. Timing uses TIMER1 at prescaler 1 plus an
 * overflow ISR, giving single-cycle resolution; the ~30 cycles each
 * overflow ISR costs are included in the totals, which is under 0.1 % for
 * the kernels here.
 *
 * Cycle counts are the input to the energy estimate:
 *
 *     E_kernel = (cycles / F_CPU) * V * I_active
 *
 * They are not a substitute for measuring current - a float multiply and
 * an SRAM read draw different amounts - but they tell you which kernels
 * are worth putting on the analyser and what window length to expect.
 *
 *   make -C test avr-bench
 */

#include <avr/interrupt.h>
#include <avr/io.h>
#include <stdint.h>

#include "../arduino/softmax_power_bench/softmax_kernels.h"

#define BAUD_UBRR 8  /* 115200 @ 16 MHz */

static volatile uint16_t g_ovf;

ISR(TIMER1_OVF_vect)
{
    g_ovf++;
}

static void uart_init(void)
{
    UBRR0 = BAUD_UBRR;
    UCSR0B = (1 << TXEN0);
    UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

static void put(char c)
{
    while (!(UCSR0A & (1 << UDRE0))) {
        ;
    }
    UDR0 = c;
}

static void puts_(const char *s)
{
    while (*s) {
        put(*s++);
    }
}

static void put_u32(uint32_t v)
{
    char buf[11];
    int8_t i = 0;

    if (v == 0) {
        put('0');
        return;
    }
    while (v) {
        buf[i++] = (char)('0' + (v % 10));
        v /= 10;
    }
    while (i--) {
        put(buf[i]);
    }
}

static void timer_start(void)
{
    TCCR1A = 0;
    TCNT1 = 0;
    g_ovf = 0;
    TIFR1 = (1 << TOV1);
    TIMSK1 = (1 << TOIE1);
    TCCR1B = (1 << CS10); /* prescaler 1 */
}

static uint32_t timer_stop(void)
{
    const uint16_t t = TCNT1;
    TCCR1B = 0;
    TIMSK1 = 0;
    return (uint32_t)g_ovf * 65536UL + t;
}

static float   gIn[SM_MAX_LEN];
static float   gOut[SM_MAX_LEN];
static int16_t gInQ[SM_MAX_LEN];
static int16_t gOutQ[SM_MAX_LEN];
static volatile int32_t gSink;

#define SM_BARRIER() __asm__ __volatile__("" ::: "memory")

static void makeLogits(uint8_t n, uint32_t seed)
{
    uint32_t s = seed;

    for (uint8_t i = 0; i < n; ++i) {
        s = s * 1664525UL + 1013904223UL;
        const float u = (float)((s >> 16) & 0xFFFF) / 65535.0f;
        gIn[i] = (u * 2.0f - 1.0f) * 8.0f;
        gInQ[i] = SM_FLOAT_TO_Q8(gIn[i]);
    }
}

/* REPS calls per measurement so the fixed cost of starting and stopping
 * the timer is amortised away; the report divides it back out. */
#define REPS 8

static void report(const char *name, uint8_t n, uint32_t cycles)
{
    puts_(name);
    put(',');
    put_u32(n);
    put(',');
    put_u32(cycles / REPS);
    put(',');
    /* microseconds at 16 MHz, two decimals, integer maths only */
    put_u32(cycles / REPS / 16);
    put('.');
    {
        const uint32_t frac = ((cycles / REPS) % 16) * 100 / 16;
        if (frac < 10) {
            put('0');
        }
        put_u32(frac);
    }
    put('\n');
}

#define BENCH(NAME, BODY)                          \
    do {                                           \
        timer_start();                             \
        for (uint8_t r = 0; r < REPS; ++r) {       \
            SM_BARRIER();                          \
            BODY;                                  \
            SM_BARRIER();                          \
        }                                          \
        report(NAME, n, timer_stop());             \
        gSink += gOutQ[0] + (int32_t)gOut[0];      \
    } while (0)

int main(void)
{
    static const uint8_t lens[] = {8, 16, 32, 64};

    uart_init();
    sei();

    puts_("# ATmega328P @ 16 MHz, cycles per call (simavr)\n");
    puts_("kernel,n,cycles,us\n");

    for (uint8_t li = 0; li < 4; ++li) {
        const uint8_t n = lens[li];

        makeLogits(n, 12345UL);

        BENCH("loop_overhead", (void)0);
        BENCH("f32_naive", softmax_f32_naive(gIn, gOut, n));
        BENCH("f32_stable", softmax_f32_stable(gIn, gOut, n));
        BENCH("f32_fastexp", softmax_f32_fastexp(gIn, gOut, n));
        BENCH("q15_lut", softmax_q15_lut(gInQ, gOutQ, n));
        BENCH("q15_poly", softmax_q15_poly(gInQ, gOutQ, n));
        BENCH("argmax", gSink = softmax_argmax_f32(gIn, n));
    }

    puts_("# done\n");

    while (!(UCSR0A & (1 << TXC0))) {
        ;
    }
    cli();
    for (;;) {
        __asm__ __volatile__("sleep");
    }
}
