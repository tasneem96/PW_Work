/*
 * softmax_kernels.h - several softmax implementations for power/energy
 * profiling on Arduino-class microcontrollers.
 *
 * Every kernel computes the same thing mathematically:
 *
 *     out[i] = exp(in[i]) / sum_j exp(in[j])
 *
 * but they trade accuracy for arithmetic cost in different ways, which is
 * what makes the energy-per-inference comparison interesting.
 *
 * The kernels are plain C++ with no Arduino dependencies so the same code
 * compiles on the host for the accuracy tests (see test/).
 */

#ifndef SOFTMAX_KERNELS_H
#define SOFTMAX_KERNELS_H

#include <stdint.h>

/* Longest vector any kernel is asked to handle. Bump with care: the bench
 * sketch statically allocates 6 * SM_MAX_LEN bytes of buffers, which is
 * already ~768 B of the ATmega328P's 2 kB at 64. */
#ifndef SM_MAX_LEN
#define SM_MAX_LEN 64
#endif

/* ---- float kernels -------------------------------------------------- */

/* Textbook softmax: exp() on the raw logits, then normalise. Cheapest in
 * instruction count of the float variants but overflows to inf once a
 * logit exceeds ~88, so it is only safe on bounded logits. */
void softmax_f32_naive(const float *in, float *out, uint8_t n);

/* Numerically stable softmax: subtract the max before exp() and multiply
 * by the reciprocal of the sum instead of dividing n times. This is what
 * you would actually ship. */
void softmax_f32_stable(const float *in, float *out, uint8_t n);

/* Stable softmax with expf() replaced by Schraudolph's exponent-field
 * bit hack. ~2 % relative error on the exponential, no libm call. */
void softmax_f32_fastexp(const float *in, float *out, uint8_t n);

/* ---- fixed-point kernels -------------------------------------------- */
/*
 * Integer kernels take Q8.8 logits (int16, scale 256, range [-128, +128))
 * and produce Q15 probabilities (int16, scale 32768, so 1.0 saturates at
 * 32767). Use SM_FLOAT_TO_Q8 / SM_Q15_TO_FLOAT to convert.
 */

#define SM_FLOAT_TO_Q8(x) ((int16_t)((x) * 256.0f + ((x) < 0 ? -0.5f : 0.5f)))
#define SM_Q15_TO_FLOAT(q) ((float)(q) * (1.0f / 32768.0f))

/* Integer softmax using a two-stage exp() table: exp(-integer part) from
 * a 16-entry table, exp(-fractional part) linearly interpolated from a
 * 17-entry table. No floating point anywhere. */
void softmax_q15_lut(const int16_t *in_q8, int16_t *out_q15, uint8_t n);

/* Integer softmax with no tables at all: the exponential is evaluated as
 * 2^-t via a shift for the integer part and a 2-term polynomial for the
 * fractional part. Smallest flash footprint of the integer variants. */
void softmax_q15_poly(const int16_t *in_q8, int16_t *out_q15, uint8_t n);

/* ---- baseline ------------------------------------------------------- */

/* Not a softmax: just the argmax. Included because for plain
 * classification the softmax is often redundant at inference time, so
 * this is the floor that the softmax variants are paying against. */
uint8_t softmax_argmax_f32(const float *in, uint8_t n);

/* Exposed for the tests / anyone wanting the fast exp on its own. */
float sm_fast_expf(float x);

#endif /* SOFTMAX_KERNELS_H */
