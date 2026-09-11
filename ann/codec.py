"""Scalar-quantisation codec with explicit XOR bit-flip semantics.

Only unsigned affine uint8 quantisation is implemented. The relaxed-value
formula in the route-delay spec,

    c'_vj = c_vj + sum_l 2^l (1 - 2 a_vjl) m_vjl

is correct for THIS codec and no other. Two's-complement int8 and IEEE
float32 both break it (sign bit and exponent are not affine in the code
word), so they are rejected rather than silently mishandled.
"""

import numpy as np

N_BITS = 8


class UInt8Codec:
    """x_hat[j] = scale[j] * (code[j] - zero[j]), code in [0, 255]."""

    def __init__(self, X):
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.maximum(hi - lo, 1e-12)
        self.scale = span / 255.0
        self.zero = -lo / self.scale
        self.d = X.shape[1]

    def encode(self, X):
        c = np.rint(X / self.scale + self.zero)
        return np.clip(c, 0, 255).astype(np.uint8)

    def decode(self, C):
        return (C.astype(np.float64) - self.zero) * self.scale

    def flip(self, C, flips):
        """Apply [(row, dim, bit), ...] by XOR. Returns a new code array."""
        out = C.copy()
        for row, dim, bit in flips:
            out[row, dim] ^= np.uint8(1 << bit)
        return out

    def delta_value(self, code_byte, dim, bit):
        """Exact change in the dequantised value from flipping one bit."""
        a = (int(code_byte) >> bit) & 1
        return self.scale[dim] * (1 << bit) * (1 - 2 * a)
