/*
 * softmax_power_bench - measure the time and energy cost of several
 * softmax implementations on an Arduino.
 *
 * Wiring: connect your power analyser's trigger/logic input to
 * MARKER_PIN. The pin is driven HIGH for exactly the duration of the
 * measured kernel loop and LOW everywhere else, including while the
 * results are printed, so the analyser can integrate current over the
 * kernel alone and not over the UART traffic.
 *
 * Serial commands (115200 baud, send a single character):
 *
 *   h        this help
 *   v        verify accuracy of each kernel against f32_stable
 *   r        run one full sweep (every kernel x every vector length)
 *   l        toggle continuous sweeping
 *   0..6     run one kernel repeatedly at SM_HOLD_LEN, for scope capture
 *   .        stop the single-kernel hold
 *   +/-      double / halve the repetition count
 *
 * See docs/POWER_MEASUREMENT.md for the measurement procedure.
 */

#include "softmax_kernels.h"

/* ---- configuration --------------------------------------------------- */

/* Driven HIGH while a kernel is running. Pick a pin with nothing else on
 * it; avoid 0/1 (UART) and 13 (LED, wastes current through the resistor). */
#define MARKER_PIN 7

/* Idle gap before each measured window. Gives the supply time to settle
 * and leaves a clearly identifiable flat section in the current trace to
 * use as the sleep/idle baseline. */
#define IDLE_SETTLE_MS 250

/* Vector length used by the single-kernel hold mode (0..6 commands). */
#define SM_HOLD_LEN 32

/* Length of the idle_spin reference window. Long enough to read a stable
 * average current off the analyser. */
#define SM_IDLE_WINDOW_US 20000UL

/* Logit range of the synthetic test data, in nats. +-8 is representative
 * of a small classifier's pre-softmax outputs and stays well inside the
 * range where the naive kernel does not overflow. */
#define SM_LOGIT_SPAN 8.0f

/* Repetitions per measured window. Bigger = better resolution on both
 * micros() and the power analyser, at the cost of a longer trace.
 * Adjust at runtime with + / -. */
#define SM_DEFAULT_REPS 200

static const uint8_t kLengths[] = {8, 16, 32, 64};
#define SM_NUM_LENGTHS (sizeof(kLengths) / sizeof(kLengths[0]))

/* Stops the optimiser from hoisting a kernel call out of the timing loop
 * or discarding it as dead. Costs zero instructions - it is a fence for
 * the compiler, not the CPU. Needed because the AVR and ESP32 cores build
 * with -flto, which can see through translation units. */
#define SM_BARRIER() __asm__ __volatile__("" ::: "memory")

/* ---- state ----------------------------------------------------------- */

enum KernelId {
    K_IDLE_SPIN = 0,
    K_F32_NAIVE,
    K_F32_STABLE,
    K_F32_FASTEXP,
    K_Q15_LUT,
    K_Q15_POLY,
    K_ARGMAX,
    K_COUNT
};

static const char *kernelName(uint8_t id)
{
    switch (id) {
    case K_IDLE_SPIN:   return "idle_spin";
    case K_F32_NAIVE:   return "f32_naive";
    case K_F32_STABLE:  return "f32_stable";
    case K_F32_FASTEXP: return "f32_fastexp";
    case K_Q15_LUT:     return "q15_lut";
    case K_Q15_POLY:    return "q15_poly";
    case K_ARGMAX:      return "argmax";
    default:            return "?";
    }
}

static float   gIn[SM_MAX_LEN];
static float   gOut[SM_MAX_LEN];
static int16_t gInQ[SM_MAX_LEN];
static int16_t gOutQ[SM_MAX_LEN];

/* Volatile sinks: the kernel results are written here after each measured
 * window so the results are observably used. */
static volatile float   gSinkF;
static volatile int32_t gSinkI;

static uint16_t gReps = SM_DEFAULT_REPS;
static bool     gLooping = false;
static int8_t   gHoldKernel = -1;

/* ---- test data ------------------------------------------------------- */

/* Deterministic LCG so every run - and the host-side accuracy test -
 * sees identical logits. Fixed data also keeps the power trace
 * repeatable, which matters more than realism here. */
static void makeLogits(uint8_t n, uint32_t seed)
{
    uint32_t s = seed;

    for (uint8_t i = 0; i < n; ++i) {
        s = s * 1664525UL + 1013904223UL;
        const float u = (float)((s >> 16) & 0xFFFF) / 65535.0f;
        gIn[i] = (u * 2.0f - 1.0f) * SM_LOGIT_SPAN;
        gInQ[i] = SM_FLOAT_TO_Q8(gIn[i]);
    }
}

/* ---- measurement ----------------------------------------------------- */

static inline void markerHigh(void) { digitalWrite(MARKER_PIN, HIGH); }
static inline void markerLow(void)  { digitalWrite(MARKER_PIN, LOW); }

/*
 * Runs `reps` iterations of one kernel inside a marked window and returns
 * the elapsed microseconds.
 *
 * The switch is outside the loop on purpose: dispatching per iteration
 * would add its own cost to every kernel and blur the comparison.
 */
static uint32_t benchKernel(uint8_t id, uint8_t n, uint16_t reps)
{
    uint32_t t0, dt;

#define SM_TIMED_LOOP(BODY)                                 \
    do {                                                    \
        markerHigh();                                       \
        t0 = micros();                                      \
        for (uint16_t r = 0; r < reps; ++r) {               \
            SM_BARRIER();                                   \
            BODY;                                           \
            SM_BARRIER();                                   \
        }                                                   \
        dt = micros() - t0;                                 \
        markerLow();                                        \
    } while (0)

    switch (id) {
    case K_IDLE_SPIN:
        /* Reference level, not a per-call cost: a fixed-length window in
         * which the CPU is awake running a trivial loop. Its average
         * current is the floor to compare the kernels against, and the
         * difference between it and a kernel window is what the softmax
         * actually costs you. An empty reps loop is useless here - the
         * optimiser folds it to a couple of cycles - so this spins for a
         * wall-clock duration instead. */
        markerHigh();
        t0 = micros();
        while (micros() - t0 < SM_IDLE_WINDOW_US) {
            SM_BARRIER();
        }
        dt = micros() - t0;
        markerLow();
        break;
    case K_F32_NAIVE:
        SM_TIMED_LOOP(softmax_f32_naive(gIn, gOut, n));
        break;
    case K_F32_STABLE:
        SM_TIMED_LOOP(softmax_f32_stable(gIn, gOut, n));
        break;
    case K_F32_FASTEXP:
        SM_TIMED_LOOP(softmax_f32_fastexp(gIn, gOut, n));
        break;
    case K_Q15_LUT:
        SM_TIMED_LOOP(softmax_q15_lut(gInQ, gOutQ, n));
        break;
    case K_Q15_POLY:
        SM_TIMED_LOOP(softmax_q15_poly(gInQ, gOutQ, n));
        break;
    case K_ARGMAX:
        SM_TIMED_LOOP(gSinkI = softmax_argmax_f32(gIn, n));
        break;
    default:
        dt = 0;
        break;
    }

#undef SM_TIMED_LOOP

    /* Consume the outputs outside the timed window. */
    gSinkF = gOut[0] + gOut[n / 2] + gOut[n - 1];
    gSinkI += (int32_t)gOutQ[0] + gOutQ[n / 2] + gOutQ[n - 1];

    return dt;
}

static void printCsvHeader(void)
{
    Serial.println(F("kernel,n,reps,total_us,us_per_call,cycles_per_call"));
}

static void measureAndReport(uint8_t id, uint8_t n)
{
    /* idle_spin is one fixed-length window, not gReps calls. */
    const uint16_t reps = (id == K_IDLE_SPIN) ? 1 : gReps;

    makeLogits(n, 12345UL);

    /* Marker low and UART quiet, so the analyser sees a clean idle
     * section before the window opens. */
    markerLow();
    Serial.flush();
    delay(IDLE_SETTLE_MS);

    const uint32_t dt = benchKernel(id, n, reps);

    const float usPerCall = (float)dt / (float)reps;
    const float cycles = usPerCall * ((float)F_CPU / 1000000.0f);

    Serial.print(kernelName(id));
    Serial.print(',');
    Serial.print(n);
    Serial.print(',');
    Serial.print(reps);
    Serial.print(',');
    Serial.print(dt);
    Serial.print(',');
    Serial.print(usPerCall, 3);
    Serial.print(',');
    Serial.println(cycles, 1);
}

static void runSweep(void)
{
    Serial.print(F("# sweep, reps="));
    Serial.print(gReps);
    Serial.print(F(", F_CPU="));
    Serial.println(F_CPU);
    printCsvHeader();

    for (uint8_t k = 0; k < K_COUNT; ++k) {
        /* idle_spin does not depend on the vector length, so measure it
         * once rather than four identical times. */
        const uint8_t lengths = (k == K_IDLE_SPIN) ? 1 : SM_NUM_LENGTHS;

        for (uint8_t li = 0; li < lengths; ++li) {
            measureAndReport(k, kLengths[li]);
        }
    }
    Serial.println(F("# sweep done"));
}

/* ---- accuracy check on-device ---------------------------------------- */

/*
 * Compares each kernel against f32_stable on the board itself. This is a
 * sanity check that the fixed-point paths were not broken by the
 * compiler's idea of 16-bit int; the rigorous comparison against a double
 * reference lives in test/test_softmax.cpp, since double is just float on
 * AVR and cannot serve as a reference here.
 */
static void verifyAccuracy(void)
{
    const uint8_t n = SM_HOLD_LEN;
    float ref[SM_MAX_LEN];

    makeLogits(n, 12345UL);
    softmax_f32_stable(gIn, ref, n);

    Serial.println(F("# accuracy vs f32_stable"));
    Serial.println(F("kernel,n,max_abs_err,sum"));

    for (uint8_t k = K_F32_NAIVE; k <= K_Q15_POLY; ++k) {
        float maxErr = 0.0f;
        float sum = 0.0f;

        switch (k) {
        case K_F32_NAIVE:   softmax_f32_naive(gIn, gOut, n);   break;
        case K_F32_STABLE:  softmax_f32_stable(gIn, gOut, n);  break;
        case K_F32_FASTEXP: softmax_f32_fastexp(gIn, gOut, n); break;
        case K_Q15_LUT:
            softmax_q15_lut(gInQ, gOutQ, n);
            for (uint8_t i = 0; i < n; ++i) {
                gOut[i] = SM_Q15_TO_FLOAT(gOutQ[i]);
            }
            break;
        case K_Q15_POLY:
            softmax_q15_poly(gInQ, gOutQ, n);
            for (uint8_t i = 0; i < n; ++i) {
                gOut[i] = SM_Q15_TO_FLOAT(gOutQ[i]);
            }
            break;
        default: break;
        }

        for (uint8_t i = 0; i < n; ++i) {
            const float e = fabsf(gOut[i] - ref[i]);
            if (e > maxErr) {
                maxErr = e;
            }
            sum += gOut[i];
        }

        Serial.print(kernelName(k));
        Serial.print(',');
        Serial.print(n);
        Serial.print(',');
        Serial.print(maxErr, 6);
        Serial.print(',');
        Serial.println(sum, 6);
    }
}

/* ---- command interface ----------------------------------------------- */

static void printHelp(void)
{
    Serial.println(F("# softmax_power_bench"));
    Serial.print(F("# marker pin D"));
    Serial.print(MARKER_PIN);
    Serial.print(F(", reps="));
    Serial.print(gReps);
    Serial.print(F(", hold len="));
    Serial.println(SM_HOLD_LEN);
    Serial.println(F("# h help | v verify | r sweep | l loop sweeps"));
    Serial.println(F("# 0..6 hold one kernel | . stop hold | +/- reps"));
    Serial.println(F("# kernels: 0 idle_spin 1 f32_naive 2 f32_stable"));
    Serial.println(F("#          3 f32_fastexp 4 q15_lut 5 q15_poly 6 argmax"));
}

static void handleCommand(char c)
{
    if (c >= '0' && c < '0' + K_COUNT) {
        gHoldKernel = (int8_t)(c - '0');
        gLooping = false;
        Serial.print(F("# holding "));
        Serial.println(kernelName((uint8_t)gHoldKernel));
        return;
    }

    switch (c) {
    case 'h':
        printHelp();
        break;
    case 'v':
        gHoldKernel = -1;
        verifyAccuracy();
        break;
    case 'r':
        gHoldKernel = -1;
        runSweep();
        break;
    case 'l':
        gLooping = !gLooping;
        gHoldKernel = -1;
        Serial.println(gLooping ? F("# looping on") : F("# looping off"));
        break;
    case '.':
        gHoldKernel = -1;
        Serial.println(F("# hold off"));
        break;
    case '+':
        if (gReps <= 16384) {
            gReps *= 2;
        }
        Serial.print(F("# reps="));
        Serial.println(gReps);
        break;
    case '-':
        if (gReps > 1) {
            gReps /= 2;
        }
        Serial.print(F("# reps="));
        Serial.println(gReps);
        break;
    default:
        break; /* ignore whitespace and stray line endings */
    }
}

/* ---- Arduino entry points -------------------------------------------- */

void setup()
{
    pinMode(MARKER_PIN, OUTPUT);
    markerLow();

    Serial.begin(115200);
    while (!Serial && millis() < 3000) {
        ; /* native-USB boards: wait briefly for the host, then continue */
    }

    printHelp();
    verifyAccuracy();
    runSweep();
}

void loop()
{
    while (Serial.available()) {
        handleCommand((char)Serial.read());
    }

    if (gHoldKernel >= 0) {
        /* Repeat one kernel forever so a scope or power analyser can
         * average many identical windows. */
        makeLogits(SM_HOLD_LEN, 12345UL);
        markerLow();
        delay(IDLE_SETTLE_MS);
        benchKernel((uint8_t)gHoldKernel, SM_HOLD_LEN, gReps);
        return;
    }

    if (gLooping) {
        runSweep();
        return;
    }

    delay(50);
}
