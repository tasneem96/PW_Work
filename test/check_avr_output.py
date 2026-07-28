#!/usr/bin/env python3
"""Verify the UART dump produced by avr_selftest running on AVR/simavr.

Reads the dump on stdin, recomputes the reference softmax in double
precision, and checks every kernel against the same error budgets the host
test uses. Exits non-zero on any violation.

    make -C test avr-check
"""

import math
import re
import struct
import sys

Q15 = 32768.0

# kernel -> (max abs error vs double reference, is the output Q15?)
BUDGETS = {
    "f32_naive": (1e-6, False),
    "f32_stable": (1e-6, False),
    "f32_fastexp": (2e-2, False),
    "q15_lut": (1e-3, True),
    "q15_poly": (4e-3, True),
}


def f32(x):
    """Round a Python double to the nearest float32, as the AVR would."""
    return struct.unpack("f", struct.pack("f", x))[0]


def expected_logits(n, seed=12345, span=8.0):
    """Third independent implementation of the sketch's LCG.

    Rounds to float32 after every operation, because the AVR has no double
    and does exactly that. Rounding only once at the end disagrees in the
    last bit or two.
    """
    out = []
    s = seed
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        u = f32(f32((s >> 16) & 0xFFFF) / f32(65535.0))
        out.append(f32(f32(f32(u * 2.0) - 1.0) * span))
    return out


def reference_softmax(logits):
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    s = sum(e)
    return [v / s for v in e]


def parse(text):
    """-> {(tag, n): [ints]}

    simavr's console view colourises the UART stream and renders the
    firmware's newlines as '.', so normalise both before parsing. The
    firmware never emits a literal '.', which is what makes that safe.
    """
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = text.replace(".", "\n")

    indexed = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Loaded"):
            continue
        parts = line.split(",")
        if len(parts) != 4:
            continue
        tag, n, idx, val = parts
        if not (n.isdigit() and idx.isdigit()):
            continue
        try:
            indexed.setdefault((tag, int(n)), {})[int(idx)] = int(val, 16)
        except ValueError:
            continue

    # Flatten to dense lists, but only where every index is present, so a
    # truncated capture shows up as a length mismatch instead of silently
    # shifting values into the wrong slots.
    rows = {}
    for key, values in indexed.items():
        count = max(values) + 1
        if set(values) == set(range(count)):
            rows[key] = [values[i] for i in range(count)]
    return rows


def main():
    rows = parse(sys.stdin.read())
    if not rows:
        print("no parseable kernel output found on stdin", file=sys.stderr)
        return 1

    failures = 0

    def check(label, ok):
        nonlocal failures
        print("  %-52s %s" % (label, "ok" if ok else "FAIL"))
        if not ok:
            failures += 1

    lengths = sorted({n for (_, n) in rows})
    print("avr_selftest output vs double reference")

    for n in lengths:
        logits_bits = rows.get(("logits", n))
        if logits_bits is None:
            check("n=%d logits present" % n, False)
            continue

        got_logits = [struct.unpack("f", struct.pack("I", b))[0] for b in logits_bits]
        want_logits = expected_logits(n)
        same = len(got_logits) == n and all(
            a == b for a, b in zip(got_logits, want_logits)
        )
        check("n=%2d logit generator matches host bit-for-bit" % n, same)
        if len(got_logits) != n:
            continue  # truncated capture; the rest of this length is noise

        ref = reference_softmax(got_logits)

        for kernel, (budget, is_q15) in BUDGETS.items():
            raw = rows.get((kernel, n))
            if raw is None:
                check("n=%2d %s present" % (n, kernel), False)
                continue

            if is_q15:
                got = [v / Q15 for v in raw]
            else:
                got = [struct.unpack("f", struct.pack("I", v))[0] for v in raw]

            err = max(abs(a - b) for a, b in zip(ref, got))
            total = sum(got)
            check(
                "n=%2d %-12s max abs err %.2e  sum %.4f"
                % (n, kernel, err, total),
                err <= budget and abs(total - 1.0) <= 0.02,
            )

        arg = rows.get(("argmax", n))
        ref_arg = max(range(n), key=lambda i: got_logits[i])
        check(
            "n=%2d argmax" % n,
            arg is not None and len(arg) == 1 and arg[0] == ref_arg,
        )

    print("\n%s" % ("FAILED" if failures else "all AVR checks passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
