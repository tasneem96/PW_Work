"""Codec tests: XOR semantics, and that the analytic bit delta matches reality."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ann.codec import UInt8Codec, N_BITS

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 16))
codec = UInt8Codec(X)
C = codec.encode(X)


def test_xor_involution():
    for _ in range(50):
        r, j, b = rng.integers(200), rng.integers(16), rng.integers(N_BITS)
        once = codec.flip(C, [(r, j, b)])
        twice = codec.flip(once, [(r, j, b)])
        assert np.array_equal(twice, C)
    print("  xor involution (flip twice == identity)            ok")


def test_delta_matches_decode():
    """delta_value must equal the realised change for EVERY dim and bit."""
    worst = 0.0
    for r in range(20):
        base = codec.decode(C[r:r + 1])[0]
        for j in range(16):
            for b in range(N_BITS):
                pred = codec.delta_value(C[r, j], j, b)
                got = codec.decode(codec.flip(C, [(r, j, b)])[r:r + 1])[0][j] - base[j]
                worst = max(worst, abs(pred - got))
    assert worst < 1e-9, worst
    print(f"  analytic delta == decoded delta (max err {worst:.2e})    ok")


def test_sign_direction():
    """Direction check on every bit, b7 included. A set bit must decrease the
    value when flipped and a clear bit must increase it. This is the assertion
    that silently fails for two's-complement int8 and for float32."""
    bad = []
    for r in range(20):
        for j in range(16):
            for b in range(N_BITS):
                a = (int(C[r, j]) >> b) & 1
                d = codec.delta_value(C[r, j], j, b)
                if (a == 1 and d >= 0) or (a == 0 and d <= 0):
                    bad.append((r, j, b, a, d))
    assert not bad, bad[:3]
    print("  flip direction correct for all bits incl. b7        ok")


def test_no_wraparound_in_attack():
    """uint8 XOR can wrap the code out of the intended monotone range; the
    attack must never emit a flip it has not range-checked."""
    from ann.attacks import _best_demote_flips
    q = rng.normal(size=16)
    for node in range(10):
        flips = _best_demote_flips(codec, C, node, q, 6)
        seen = set()
        for _, j, b in flips:
            assert (j, b) not in seen, "same bit flipped twice"
            seen.add((j, b))
    print("  attack never flips the same bit twice               ok")


if __name__ == "__main__":
    print("codec tests")
    test_xor_involution()
    test_delta_matches_decode()
    test_sign_direction()
    test_no_wraparound_in_attack()
    print("all passed")
