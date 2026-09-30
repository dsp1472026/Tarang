"""
Pseudo-random (PR) interleavers.

A PR interleaver is an arbitrary permutation of a block of P bits, so it cannot be found by a
geometric search like the block / diagonal / convolutional types. What is implemented:

1. **LFSR address-generator family** (the construction used by address-generator interleavers):
   a Fibonacci LFSR of n = ceil(log2(P + 1)) bits with a primitive feedback polynomial runs
   from a seed; every state s (1 .. 2^n - 1) gives the address s - 1; addresses >= P are
   skipped. The first P addresses form the permutation: out[i] = in[addr[i]].
   Parameters: (P, polynomial, seed). The polynomial table below is checked for maximal
   length at import (non-maximal entries are dropped).
2. **QPP family** pi(i) = (f1*i + f2*i^2) mod K (the structure of the LTE turbo interleaver),
   for explicitly given (K, f1, f2); validity is checked. No standard parameter table is
   included.
3. **Presence detection**: a rank-deficiency period P (Burel-Gautier) that no block,
   diagonal or convolutional hypothesis explains, together with code structure visible only
   in whole P-bit rows, indicates a PR interleaver of period P.
4. **Permuted dual code** (fec/dual.py): with enough aligned P-bit blocks, the parity checks of
   the (unknown) permuted code are recovered as sparse vectors of the null space. This
   identifies the code under an unknown permutation; reconstructing the permutation itself
   from those checks is NOT implemented.

Blind search (postdemod/analyzer.py): LFSR hypotheses for the detected period (all seeds for
n <= 10) and for the data span taken as one interleaver block (P within 16 bits of the span,
standard seeds 1 and all-ones), each screened by FEC structure and confirmed by CRC.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

# Maximal-length Fibonacci LFSR feedback taps (x^n + ... ) for n = 2..16.
_TAPS_CANDIDATES: Dict[int, List[Tuple[int, ...]]] = {
    2: [(2, 1)], 3: [(3, 2)], 4: [(4, 3)], 5: [(5, 3)], 6: [(6, 5)], 7: [(7, 6)], 8: [(8, 6, 5, 4)],
    9: [(9, 5)], 10: [(10, 7)], 11: [(11, 9)], 12: [(12, 6, 4, 1)], 13: [(13, 4, 3, 1)],
    14: [(14, 5, 3, 1)], 15: [(15, 14)], 16: [(16, 15, 13, 4)],
}


def _lfsr_states(n: int, taps: Tuple[int, ...], seed: int, count: int) -> np.ndarray:
    """count successive n-bit states of a Fibonacci LFSR (state as integer, MSB = stage 1)."""
    mask = (1 << n) - 1
    s = seed & mask
    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        out[i] = s
        fb = 0
        for t in taps:
            fb ^= (s >> (n - t)) & 1
        s = ((s >> 1) | (fb << (n - 1))) & mask
    return out


def _is_maximal(n: int, taps: Tuple[int, ...]) -> bool:
    st = _lfsr_states(n, taps, 1, (1 << n))
    return len(np.unique(st[:(1 << n) - 1])) == (1 << n) - 1 and 0 not in st[:(1 << n) - 1]


PRIMITIVE_TAPS: Dict[int, List[Tuple[int, ...]]] = {
    n: [t for t in ts if _is_maximal(n, t)] for n, ts in _TAPS_CANDIDATES.items()}


def lfsr_bits(P: int) -> int:
    return max(2, int(np.ceil(np.log2(P + 1))))


@lru_cache(maxsize=256)
def lfsr_permutation(P: int, taps: Tuple[int, ...], seed: int) -> np.ndarray:
    n = lfsr_bits(P)
    st = _lfsr_states(n, taps, seed, (1 << n) - 1)
    addr = st - 1
    perm = addr[addr < P]
    if len(perm) != P:
        raise ValueError("seed / polynomial do not give a full permutation")
    return perm


def qpp_permutation(K: int, f1: int, f2: int) -> np.ndarray:
    i = np.arange(K, dtype=np.int64)
    perm = (f1 * i + f2 * i * i) % K
    if len(np.unique(perm)) != K:
        raise ValueError(f"(f1={f1}, f2={f2}) is not a QPP permutation for K={K}")
    return perm


def interleave(bits: np.ndarray, perm: np.ndarray) -> np.ndarray:
    x = np.asarray(bits)
    P = len(perm)
    x = np.concatenate([x, np.zeros((-len(x)) % P, dtype=x.dtype)])
    return x.reshape(-1, P)[:, perm].reshape(-1)


def deinterleave(bits: np.ndarray, perm: np.ndarray, offset: int = 0) -> np.ndarray:
    x = np.asarray(bits)[offset:]
    P = len(perm)
    nb = len(x) // P
    if nb == 0:
        return x[:0]
    inv = np.empty(P, dtype=np.int64)
    inv[perm] = np.arange(P)
    return x[:nb * P].reshape(nb, P)[:, inv].reshape(-1)


@dataclass(frozen=True)
class PRHypothesis:
    P: int
    taps: Tuple[int, ...]
    seed: int

    def describe(self) -> str:
        poly = " + ".join(f"x^{t}" for t in self.taps) + " + 1"
        return f"LFSR address generator, P = {self.P}, n = {lfsr_bits(self.P)}, poly {poly}, seed 0x{self.seed:X}"

    def perm(self) -> np.ndarray:
        return lfsr_permutation(self.P, self.taps, self.seed)


def hypotheses(P: int, all_seeds: bool = False, max_all_seeds_n: int = 10) -> List[PRHypothesis]:
    n = lfsr_bits(P)
    taps = PRIMITIVE_TAPS.get(n, [])
    if all_seeds and n <= max_all_seeds_n:
        seeds = range(1, 1 << n)
    else:
        seeds = sorted({1, (1 << n) - 1})
    return [PRHypothesis(P, t, s) for t in taps for s in seeds]


def presence_evidence(period: Optional[int], structured_found: bool) -> Optional[str]:
    """Text for the report when a rank-deficiency period exists but no structured interleaver fits."""
    if period and not structured_found:
        return (f"GF(2) rank deficiency repeats with period {period} bits, but no block, diagonal or "
                f"convolutional interleaver of that period validates: a pseudo-random (permutation) "
                f"interleaver of period {period} is suspected")
    return None
