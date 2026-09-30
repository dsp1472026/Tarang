"""
Convolutional (Forney) interleaver: B branches, branch i delays its bits by i*M commutator
cycles; the de-interleaver uses delays (B-1-i)*M, so every bit is delayed by (B-1)*M*B bits
end to end. The transmitter flushes the delay lines with (B-1)*M*B zero bits so the whole
frame leaves the interleaver; the receiver drops the first (B-1)*M*B de-interleaved bits.

Blind search (Gan et al. / ICACT repetitive-pattern references): B and M are hypothesised
over a bounded grid (the burst length constrains B(B-1)M), each reconstruction is scored by
downstream FEC syndrome / sync / CRC evidence, and candidates are ranked, never accepted on
pattern evidence alone.
"""

from typing import List, Tuple

import numpy as np


def _delay_lines(bits: np.ndarray, B: int, delays: List[int], phase: int = 0) -> np.ndarray:
    x = np.asarray(bits)                             # bits or LLRs: keep the dtype
    out = np.zeros(len(x), dtype=x.dtype)
    for i in range(B):
        br = (i + phase) % B
        idx = np.arange(i, len(x), B)               # positions served by this branch
        d = delays[br]
        seq = x[idx]
        shifted = np.concatenate([np.zeros(d, dtype=x.dtype), seq])[:len(seq)]
        out[idx] = shifted
    return out


def flush_bits(B: int, M: int) -> int:
    return (B - 1) * M * B


def interleave(bits: np.ndarray, B: int, M: int) -> np.ndarray:
    b = np.asarray(bits)
    x = np.concatenate([b, np.zeros(flush_bits(B, M), dtype=b.dtype)])
    return _delay_lines(x, B, [i * M for i in range(B)])


def deinterleave(bits: np.ndarray, B: int, M: int, phase: int = 0) -> np.ndarray:
    """phase: branch index of the first received bit (0 when the burst starts at the frame start)."""
    y = _delay_lines(bits, B, [(B - 1 - i) * M for i in range(B)], phase)
    return y[flush_bits(B, M):]


def hypotheses(n_bits: int, max_B: int = 16, max_M: int = 16) -> List[Tuple[int, int]]:
    """(B, M) pairs whose flush length leaves at least half the burst as data."""
    return [(B, M) for B in range(2, max_B + 1) for M in range(1, max_M + 1)
            if flush_bits(B, M) <= n_bits // 2]
