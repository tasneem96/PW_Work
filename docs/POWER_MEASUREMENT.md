# Measuring the energy per softmax call

The sketch is built around one idea: **measure differentially**. You are
not trying to find the absolute current of the board, you are trying to
find the extra energy a softmax costs. That difference is measurable even
on a board whose baseline current is dominated by things you cannot remove.

## The marker pin

D7 (`MARKER_PIN` in the sketch) is HIGH for exactly the duration of the
measured kernel loop and LOW everywhere else — including during the
`Serial.print` of the results, which is deliberate. UART transmission is
not free, and if it landed inside the window it would be attributed to the
softmax.

Each measured window is preceded by a 250 ms quiet gap
(`IDLE_SETTLE_MS`) so the supply settles and the trace has an obvious flat
section to use as a baseline.

Use the marker as your analyser's trigger or gate:

- **Joulescope**: connect to a GPI input, then use the marker edges to set
  the integration window.
- **Nordic PPK2**: feed it to a logic input and gate on it in the
  Power Profiler app.
- **Oscilloscope + shunt**: trigger channel 1 on the marker's rising edge,
  measure the voltage across the shunt on channel 2, and use the scope's
  mean-over-gated-region function.
- **INA219 / INA260**: these sample far too slowly (~1 kHz) to resolve a
  single call, but they work fine for the *hold* mode below, where the same
  kernel repeats for seconds at a time.

## Procedure

1. Wire your current measurement in series with the MCU supply — see
   "Where to measure" below, it matters more than anything else here.
2. Upload the sketch, open the serial monitor at 115200.
3. Record the **baseline**: send `0` to hold `idle_spin`. This is a
   fixed-length window (20 ms) in which the CPU is awake running a trivial
   loop. Average current here is your floor, call it `I_idle`.
4. Record each **kernel**: send `1` … `6` in turn. Each holds one kernel,
   repeating forever with identical windows, which is what lets you
   average away noise. Average current inside the marker-HIGH region is
   `I_kernel`.
5. Read `us_per_call` for that kernel from the CSV that the `r` sweep
   printed, or from the reference table in the README.

Then, per call:

```
E_call = V_supply * I_kernel * t_call            # total
E_extra = V_supply * (I_kernel - I_idle) * t_call # attributable to the softmax
```

`E_extra` is the number that lets you compare kernels honestly, because it
cancels everything constant about your board.

Worked example, using the measured n=32 timings and a plausible 5 V /
12 mA for a bare ATmega328P at 16 MHz — substitute your own current:

| Kernel | t_call | E_call at 60 mW |
| --- | --- | --- |
| `f32_stable` | 5654 µs | ~339 µJ |
| `f32_fastexp` | 1725 µs | ~104 µJ |
| `q15_lut` | 905 µs | ~54 µJ |
| `argmax` | 159 µs | ~9.5 µJ |

Those energies are illustrative — the point of measuring is that current
is *not* actually constant across kernels. Float code keeps different
units busy than integer code, so expect the ratio of measured energies to
differ somewhat from the ratio of times.

## Where to measure

This is the part that ruins most microcontroller power measurements.

**On an Arduino Uno, do not measure at the USB or barrel jack.** The
onboard ATmega16U2 USB-serial bridge and the linear regulator together
draw more than the ATmega328P does, so the softmax disappears into the
noise. Options, best first:

1. **A bare ATmega328P or a Pro Mini** (no USB bridge, no regulator if you
   feed 5 V or 3.3 V directly). Cleanest by far. Use a USB-serial adapter
   for the UART, powered separately from the MCU.
2. **Measure only the MCU's VCC pins.** On an Uno you have to cut or lift
   the 328P's VCC connection to insert the shunt, which is invasive but
   gives a clean signal.
3. **Measure at the 5 V rail and rely on the differential.** The
   `I_kernel - I_idle` subtraction removes the constant USB-bridge and
   regulator draw. Absolute numbers are useless, relative ones are still
   valid. This is the no-hardware-modification option and it is good
   enough for comparing kernels.

Whichever you choose, keep it consistent across all kernels — the
comparison is what carries the information.

## Things that will skew the result

- **The onboard LED on D13.** Do not use D13 as the marker pin: the LED
  and its resistor add a few mA inside your measurement window. D7 is the
  default for this reason.
- **Serial traffic inside the window.** The sketch already keeps prints
  outside the marker region. If you add your own instrumentation, keep it
  outside too, and call `Serial.flush()` before the window as the sketch
  does.
- **Too few repetitions.** With `reps` too low, `micros()` quantisation
  (4 µs on AVR) and the analyser's sample rate both matter. Raise `reps`
  with `+` until the window is at least a few hundred milliseconds.
- **Compiler optimisation.** The timing loop uses a memory barrier
  (`SM_BARRIER()`) around every call, because the AVR and ESP32 cores build
  with `-flto` and will otherwise hoist a pure function out of the loop and
  report an impossibly fast kernel. If you write your own loop, keep the
  barrier and read the outputs afterwards.
- **Clock and supply.** Energy per call scales with `V²` and with time, so
  an 8 MHz / 3.3 V Pro Mini is dramatically cheaper per inference than a
  16 MHz / 5 V Uno. Compare kernels at one operating point.
- **Sleeping between calls.** In a real duty-cycled application, the
  softmax's share of total energy depends on how long the MCU sleeps
  between inferences. This bench deliberately measures the active window
  only; scale it by your duty cycle yourself.

## Changing the test data

The logits come from a fixed-seed LCG (`makeLogits()`), spanning
±`SM_LOGIT_SPAN` (default ±8) nats, so every run and every kernel sees
identical inputs and the power trace is repeatable. If you want to bench
your own model's logits, replace `makeLogits()` — but keep it
deterministic, otherwise you cannot average windows against each other.

Note that `f32_naive` overflows to `inf` once a logit exceeds ~88, so do
not raise `SM_LOGIT_SPAN` far without dropping that kernel.
