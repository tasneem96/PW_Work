/*
 * avr_selftest - runs the softmax kernels on a real ATmega328P (or under
 * simavr) and dumps the raw outputs over UART0 for check_avr_output.py to
 * verify.
 *
 * The point of this test is codegen, not maths: `int` is 16 bits on AVR,
 * so a missing int32_t cast in the fixed-point kernels passes the host
 * test and silently overflows here. Everything is printed as hex integers
 * (floats as their raw bit pattern) so no float formatting - and no
 * rounding of the evidence - is involved.
 *
 *   make -C test avr-check
 */

#include <avr/io.h>
#include <stdint.h>

#include "../arduino/softmax_power_bench/softmax_kernels.h"

#define BAUD_UBRR 8  /* 115200 @ 16 MHz, U2X off */

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

static void put_hex32(uint32_t v)
{
    for (int8_t sh = 28; sh >= 0; sh -= 4) {
        const uint8_t nib = (uint8_t)((v >> sh) & 0xF);
        put((char)(nib < 10 ? '0' + nib : 'a' + nib - 10));
    }
}

static void put_dec(uint16_t v)
{
    char buf[6];
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

static uint32_t float_bits(float f)
{
    union {
        float f;
        uint32_t u;
    } c;
    c.f = f;
    return c.u;
}

static float   gIn[SM_MAX_LEN];
static float   gOut[SM_MAX_LEN];
static int16_t gInQ[SM_MAX_LEN];
static int16_t gOutQ[SM_MAX_LEN];

/* Must stay identical to makeLogits() in the sketch and make_logits() in
 * the host test; check_avr_output.py re-derives it a third time. */
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

/*
 * One value per line: "<tag>,<n>,<index>,<hex>". Verbose, but a 64-entry
 * vector on one line overruns the line buffer of simavr's console view and
 * the tail is silently lost.
 */
static void emit_one(const char *tag, uint8_t n, uint8_t i, uint32_t v)
{
    puts_(tag);
    put(',');
    put_dec(n);
    put(',');
    put_dec(i);
    put(',');
    put_hex32(v);
    put('\n');
}

static void emit_f(const char *tag, uint8_t n)
{
    for (uint8_t i = 0; i < n; ++i) {
        emit_one(tag, n, i, float_bits(gOut[i]));
    }
}

static void emit_q(const char *tag, uint8_t n)
{
    for (uint8_t i = 0; i < n; ++i) {
        emit_one(tag, n, i, (uint32_t)(uint16_t)gOutQ[i]);
    }
}

int main(void)
{
    static const uint8_t lens[] = {8, 16, 32, 64};

    uart_init();
    puts_("# avr_selftest\n");

    for (uint8_t li = 0; li < 4; ++li) {
        const uint8_t n = lens[li];

        makeLogits(n, 12345UL);

        /* The logits themselves, so the generator is verified too. */
        for (uint8_t i = 0; i < n; ++i) {
            gOut[i] = gIn[i];
        }
        emit_f("logits", n);

        softmax_f32_naive(gIn, gOut, n);
        emit_f("f32_naive", n);

        softmax_f32_stable(gIn, gOut, n);
        emit_f("f32_stable", n);

        softmax_f32_fastexp(gIn, gOut, n);
        emit_f("f32_fastexp", n);

        softmax_q15_lut(gInQ, gOutQ, n);
        emit_q("q15_lut", n);

        softmax_q15_poly(gInQ, gOutQ, n);
        emit_q("q15_poly", n);

        emit_one("argmax", n, 0, softmax_argmax_f32(gIn, n));
    }

    puts_("# done\n");

    /* Let the last byte leave the shift register, then halt: simavr exits
     * on sleep with interrupts disabled. */
    while (!(UCSR0A & (1 << TXC0))) {
        ;
    }
    for (;;) {
        __asm__ __volatile__("sleep");
    }
}
