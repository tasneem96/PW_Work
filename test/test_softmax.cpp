/*
 * Host-side accuracy test for the softmax kernels.
 *
 *   make -C test && test/test_softmax
 *
 * Compares every kernel against a double-precision reference over a set
 * of awkward inputs and prints the worst-case absolute probability error,
 * so the accuracy cost of each variant is measured rather than assumed.
 */

#include "../arduino/softmax_power_bench/softmax_kernels.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

void reference_softmax(const float *in, double *out, uint8_t n)
{
    double max = in[0];
    for (uint8_t i = 1; i < n; ++i) {
        if (in[i] > max) {
            max = in[i];
        }
    }
    double sum = 0.0;
    for (uint8_t i = 0; i < n; ++i) {
        out[i] = std::exp((double)in[i] - max);
        sum += out[i];
    }
    for (uint8_t i = 0; i < n; ++i) {
        out[i] /= sum;
    }
}

/* Same deterministic generator the sketch uses, so the numbers the test
 * checks are the numbers the board actually runs. */
void make_logits(std::vector<float> &v, uint8_t n, uint32_t seed, float span)
{
    v.resize(n);
    uint32_t s = seed;
    for (uint8_t i = 0; i < n; ++i) {
        s = s * 1664525UL + 1013904223UL;
        const float u = (float)((s >> 16) & 0xFFFF) / 65535.0f; /* [0,1] */
        v[i] = (u * 2.0f - 1.0f) * span;
    }
}

struct Stats {
    double max_abs_err = 0.0;
    int    argmax_mismatches = 0;
};

/*
 * tie_tol: a kernel picking a different argmax is only a real defect if
 * the reference probabilities of the two candidates actually differ. The
 * integer kernels quantise the logits to 1/256, so genuine near-ties can
 * legitimately swap order; a swap where the reference gap exceeds the
 * kernel's own error budget is a defect.
 */
void accumulate(Stats &st, const double *ref, const float *got, uint8_t n,
                double tie_tol)
{
    uint8_t ref_arg = 0, got_arg = 0;
    for (uint8_t i = 0; i < n; ++i) {
        const double e = std::fabs(ref[i] - (double)got[i]);
        if (e > st.max_abs_err) {
            st.max_abs_err = e;
        }
        if (ref[i] > ref[ref_arg]) {
            ref_arg = i;
        }
        if (got[i] > got[got_arg]) {
            got_arg = i;
        }
    }
    if (ref_arg != got_arg && ref[ref_arg] - ref[got_arg] > tie_tol) {
        st.argmax_mismatches++;
    }
}

void check(const char *what, bool ok)
{
    std::printf("  %-46s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) {
        g_failures++;
    }
}

void report(const char *name, const Stats &st, double err_budget)
{
    char label[96];
    std::snprintf(label, sizeof label,
                  "%-14s max abs err %.2e  argmax swaps %d", name,
                  st.max_abs_err, st.argmax_mismatches);
    check(label, st.max_abs_err <= err_budget && st.argmax_mismatches == 0);
}

} /* namespace */

int main()
{
    /* Logit spans deliberately include a case wide enough to make the
     * integer kernels flush small terms to zero. */
    const float spans[] = {0.5f, 2.0f, 8.0f};
    const uint8_t lens[] = {1, 2, 8, 16, 32, 64};

    Stats naive, stable, fastexp, lut, poly;
    int argmax_errors = 0;

    for (uint32_t seed = 1; seed <= 200; ++seed) {
        for (float span : spans) {
            for (uint8_t n : lens) {
                std::vector<float> in;
                make_logits(in, n, seed, span);

                std::vector<double> ref(n);
                reference_softmax(in.data(), ref.data(), n);

                std::vector<float> out(n);
                std::vector<int16_t> qin(n), qout(n);
                for (uint8_t i = 0; i < n; ++i) {
                    qin[i] = SM_FLOAT_TO_Q8(in[i]);
                }

                softmax_f32_naive(in.data(), out.data(), n);
                accumulate(naive, ref.data(), out.data(), n, 1e-6);

                softmax_f32_stable(in.data(), out.data(), n);
                accumulate(stable, ref.data(), out.data(), n, 1e-6);

                softmax_f32_fastexp(in.data(), out.data(), n);
                accumulate(fastexp, ref.data(), out.data(), n, 2e-2);

                softmax_q15_lut(qin.data(), qout.data(), n);
                for (uint8_t i = 0; i < n; ++i) {
                    out[i] = SM_Q15_TO_FLOAT(qout[i]);
                }
                accumulate(lut, ref.data(), out.data(), n, 1e-3);

                softmax_q15_poly(qin.data(), qout.data(), n);
                for (uint8_t i = 0; i < n; ++i) {
                    out[i] = SM_Q15_TO_FLOAT(qout[i]);
                }
                accumulate(poly, ref.data(), out.data(), n, 4e-3);

                uint8_t ref_arg = 0;
                for (uint8_t i = 0; i < n; ++i) {
                    if (ref[i] > ref[ref_arg]) {
                        ref_arg = i;
                    }
                }
                if (softmax_argmax_f32(in.data(), n) != ref_arg) {
                    argmax_errors++;
                }
            }
        }
    }

    std::printf("accuracy vs double reference (3600 vectors, n = 1..64)\n");
    report("f32_naive",   naive,   1e-6);
    report("f32_stable",  stable,  1e-6);
    report("f32_fastexp", fastexp, 2e-2);
    report("q15_lut",     lut,     1e-3);
    report("q15_poly",    poly,    4e-3);
    check("argmax matches reference", argmax_errors == 0);

    /* Properties that must hold regardless of the error budget. */
    std::printf("\ninvariants\n");
    {
        std::vector<float> in;
        make_logits(in, 32, 7, 4.0f);
        std::vector<int16_t> qin(32), qout(32);
        for (uint8_t i = 0; i < 32; ++i) {
            qin[i] = SM_FLOAT_TO_Q8(in[i]);
        }

        std::vector<float> out(32);
        softmax_f32_stable(in.data(), out.data(), 32);
        double s = 0.0;
        for (float v : out) {
            s += v;
        }
        check("f32_stable sums to 1", std::fabs(s - 1.0) < 1e-5);

        softmax_q15_lut(qin.data(), qout.data(), 32);
        bool nonneg = true;
        int32_t qs = 0;
        for (int16_t v : qout) {
            nonneg = nonneg && v >= 0;
            qs += v;
        }
        check("q15_lut output is a valid Q15 probability", nonneg);
        check("q15_lut sums to 1.0 +-1 %",
              std::fabs((double)qs / 32768.0 - 1.0) < 0.01);

        softmax_q15_poly(qin.data(), qout.data(), 32);
        qs = 0;
        for (int16_t v : qout) {
            qs += v;
        }
        check("q15_poly sums to 1.0 +-2 %",
              std::fabs((double)qs / 32768.0 - 1.0) < 0.02);
    }

    /* A logit spread wide enough that every exponential underflows to
     * zero in Q15 must not divide by zero. */
    {
        int16_t qin[4] = {SM_FLOAT_TO_Q8(100.0f), SM_FLOAT_TO_Q8(-100.0f),
                          SM_FLOAT_TO_Q8(-100.0f), SM_FLOAT_TO_Q8(-100.0f)};
        int16_t qout[4] = {0, 0, 0, 0};
        softmax_q15_lut(qin, qout, 4);
        bool sane = true;
        for (int i = 0; i < 4; ++i) {
            if (qout[i] < 0) {
                sane = false;
            }
        }
        check("q15_lut survives fully-underflowing input", sane);
    }

    /* Schraudolph exp on its own. */
    {
        double worst = 0.0;
        for (int i = -800; i <= 0; ++i) {
            const float x = (float)i / 100.0f;
            const double got = sm_fast_expf(x);
            const double want = std::exp((double)x);
            worst = std::max(worst, std::fabs(got - want) / want);
        }
        char label[96];
        std::snprintf(label, sizeof label,
                      "sm_fast_expf max rel err %.2e on [-8, 0]", worst);
        check(label, worst < 0.05);
    }

    std::printf("\n%s\n", g_failures ? "FAILED" : "all checks passed");
    return g_failures ? 1 : 0;
}
