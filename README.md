# PW_Work — softmax kernels for Arduino, built for power measurement

Five softmax implementations plus a baseline, and a benchmark sketch that
drives a marker pin so a power analyser can integrate current over each
kernel individually.

The point is the comparison: every kernel computes
`out[i] = exp(in[i]) / sum(exp(in))`, but they buy speed with accuracy in
different ways, so you can plot energy against error and pick a point on
that curve.

## Layout

| Path | What it is |
| --- | --- |
| `arduino/softmax_power_bench/softmax_kernels.{h,cpp}` | The kernels. No Arduino dependencies — plain C++. |
| `arduino/softmax_power_bench/softmax_power_bench.ino` | Benchmark sketch: marker pin, CSV over serial, serial commands. |
| `test/test_softmax.cpp` | Host accuracy test against a `double` reference. |
| `test/avr_selftest.cpp` | Same kernels run on an emulated ATmega328P. |
| `test/avr_bench.cpp` | Cycle counts per call on an emulated ATmega328P. |
| `tools/gen_exp_tables.py` | Regenerates the Q15 `exp()` lookup tables. |
| `docs/POWER_MEASUREMENT.md` | How to actually take the measurement. |

## The kernels

| Kernel | Arithmetic | Notes |
| --- | --- | --- |
| `f32_naive` | float, `expf()` | Textbook version. Overflows to `inf` for logits above ~88. |
| `f32_stable` | float, `expf()` | Subtracts the max, multiplies by `1/sum`. The one you would ship. |
| `f32_fastexp` | float, bit-hack `exp` | Schraudolph's exponent-field trick instead of `expf()`. |
| `q15_lut` | int only | Two-stage `exp()` table (integer part + interpolated fraction). |
| `q15_poly` | int only | No tables: shift for the integer part, quadratic for the fraction. |
| `argmax` | float compare | Not a softmax. The floor you are paying against — for plain classification the softmax is often redundant at inference time. |

Integer kernels take Q8.8 logits (`int16`, scale 256) and return Q15
probabilities (`int16`, scale 32768). Convert with `SM_FLOAT_TO_Q8` and
`SM_Q15_TO_FLOAT`.

## Quick start

1. Open `arduino/softmax_power_bench/softmax_power_bench.ino` in the
   Arduino IDE (the two `softmax_kernels` files sit in the same folder, so
   they are picked up automatically) and upload.
2. Connect your analyser's logic/trigger input to **D7**. It is HIGH for
   exactly the duration of each measured kernel loop and LOW everywhere
   else, including while results are printed over UART.
3. Open the serial monitor at **115200**. The sketch verifies accuracy and
   runs one sweep on boot, then waits for commands:

```
h        help
v        verify each kernel against f32_stable
r        run one sweep (every kernel x every vector length)
l        toggle continuous sweeping
0..6     hold one kernel, repeating forever (for scope averaging)
.        stop holding
+ / -    double / halve the repetition count
```

Output is CSV, ready to paste into a spreadsheet:

```
kernel,n,reps,total_us,us_per_call,cycles_per_call
f32_stable,32,200,1130824,5654.120,90465.9
q15_lut,32,200,181000,905.000,14480.0
```

For power work, `4` (hold `q15_lut`) is usually what you want: one kernel,
identical windows forever, easy to average.

## Measured reference numbers

These are from this repo's tests, not estimates. Cycles come from simavr,
which is instruction-accurate for the ATmega328P, and the accuracy figures
from 3600 random vectors checked against a `double` reference.

**Cycles per call, ATmega328P @ 16 MHz** (`make -C test avr-bench`):

| Kernel | n=8 | n=16 | n=32 | n=64 | µs @ n=32 |
| --- | --- | --- | --- | --- | --- |
| `f32_naive` | 25 536 | 51 517 | 102 428 | 206 043 | 6402 |
| `f32_stable` | 22 668 | 45 552 | 90 466 | 180 932 | 5654 |
| `f32_fastexp` | 7 096 | 13 872 | 27 594 | 54 571 | 1725 |
| `q15_lut` | 4 264 | 7 666 | 14 481 | 28 120 | 905 |
| `q15_poly` | 5 420 | 10 375 | 20 438 | 41 574 | 1277 |
| `argmax` | 625 | 1 265 | 2 545 | 5 105 | 159 |

**Accuracy and footprint** (`make -C test check`, `make -C test avr-size`):

| Kernel | Max abs error | Flash, linked | Speedup vs `f32_stable` at n=64 |
| --- | --- | --- | --- |
| `f32_naive` | 1.7e-07 | 1672 B | 0.88x |
| `f32_stable` | 2.5e-07 | 1794 B | 1.0x |
| `f32_fastexp` | 1.2e-02 | 1518 B | 3.3x |
| `q15_lut` | 9.8e-04 | 976 B | 6.4x |
| `q15_poly` | 1.9e-03 | 960 B | 4.4x |
| `argmax` | n/a | 240 B | 35x |

Error is the worst absolute deviation of any single probability. Flash is
measured by linking each kernel alone into a bare AVR binary and
subtracting the empty-`main` baseline, so it includes the libm and
float-support code each one drags in — that is most of why the float
kernels cost 1.5–1.8 kB.

Since time dominates energy at a fixed clock and supply, the cycle column
is roughly the energy column: expect `q15_lut` to use about a sixth of the
energy of `f32_stable` per inference, and `f32_fastexp` about a third.
Measure it rather than trusting that — instruction mix affects current
draw, which is the whole reason for the marker pin.

Two results worth knowing about, both of which surprised me:

- **`q15_poly` is slower than `q15_lut`** and saves essentially no flash
  (960 B vs 976 B). Its polynomial multiplies cost more than two table
  reads, and the table is only 66 bytes. It is kept because the trade-off
  moves on other architectures — a Cortex-M4 with single-cycle multiply
  and no PROGMEM penalty flips it — but on AVR, `q15_lut` wins outright.
- **The integer kernels only beat the float ones once normalisation uses
  a reciprocal.** Dividing `n` times instead of multiplying by `1/sum`
  made `q15_lut` *slower* than `f32_fastexp`, because a 32-bit divide on
  AVR is ~250 cycles against ~40 for the multiply. That single change is
  worth 2.5x at n=64. See `sm_normalise_q15()` for the range argument that
  keeps the fixed-point reciprocal from overflowing.

## Running the tests

```sh
make -C test check       # host accuracy test vs double reference
make -C test avr-check   # run the kernels on an emulated ATmega328P
make -C test avr-bench   # cycles per call
make -C test avr-size    # per-symbol flash cost
```

`check` needs only `g++`. The `avr-*` targets need `gcc-avr`, `avr-libc`
and `simavr`:

```sh
sudo apt-get install gcc-avr avr-libc simavr
```

`avr-check` exists because `int` is **16 bits** on AVR: a missing
`int32_t` cast in the fixed-point kernels passes the host test and
silently overflows on the board. It runs the real AVR build, dumps every
output over UART as raw hex, and checks it against a `double` reference in
Python. It also re-derives the test-data generator a third time, so the
board, the host test and the checker all have to agree bit-for-bit.

## Porting notes

- **Tested on** ATmega328P (Uno / Nano / Pro Mini). The kernels are plain
  C++ and use `PROGMEM` only under `ARDUINO_ARCH_AVR`, so SAMD, ESP32 and
  RP2040 compile unchanged — but the cycle table above is AVR-specific,
  and on a core with hardware float the ranking will shift toward the
  float kernels.
- **SRAM on an Uno is tight.** At `SM_MAX_LEN` 64 the sketch uses roughly
  1 kB of the ATmega328P's 2 kB. Drop `SM_MAX_LEN` to 32 if you add
  anything substantial.
- The `expf()` in `f32_naive` and `f32_stable` comes from libm, so those
  two are the only kernels whose cost depends on your toolchain's libm.
