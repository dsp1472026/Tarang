"""
Byte <-> bit helpers for the image pipeline, MSB first.

    bytes_to_bits: bit_i = (byte >> (7 - i)) & 1, i = 0..7
    bits_to_bytes: byte  = sum(bit_i << (7 - i)); trailing bits short of a byte are dropped

Both reuse synthetic.framing (np.unpackbits / np.packbits, MSB first), the helpers every
receiver in the project already uses, so the bit order cannot drift between them.
"""

import numpy as np

from synthetic import framing


def bytes_to_bits(data: bytes) -> np.ndarray:
    """uint8 array of 8 * len(data) bits, MSB first."""
    return framing.bytes_to_bits(bytes(data)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack bits (MSB first) into bytes; a trailing partial byte is dropped."""
    return framing.bits_to_bytes(np.asarray(bits, dtype=np.uint8) & 1)
