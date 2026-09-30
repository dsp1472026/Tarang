"""
Diagonal (helical) block interleaver.

Per block of P = R*C bits the transmitter writes the bits row by row into an R x C matrix
and reads them along the wrapped diagonals:

    for j in 0..C-1:                  (diagonal index)
        for r in 0..R-1:
            out.append(M[r, (j + s*r) mod C])       s = +1 "down-right", s = -1 "down-left"

Consecutive output bits come from different rows AND different columns, which is the
property the diagonal interleaver adds over the rectangular one. This is a permutation for
every R and C. (The expression pi(k) = ((k mod R)*C + (k mod R) + floor(k/R)) mod (R*C)
that is sometimes quoted for it is not a permutation for any R, C >= 2 and is not used.)
The last block is zero-padded by the transmitter. Blind use: (R, C, direction) hypotheses
screened by FEC structure and confirmed by CRC (postdemod/analyzer.py).
"""

import numpy as np

DIRECTIONS = (1, -1)


def permutation(R: int, C: int, direction: int = 1) -> np.ndarray:
    """out[m] = in[perm[m]] for one block."""
    j = np.repeat(np.arange(C), R)
    r = np.tile(np.arange(R), C)
    return (r * C + (j + direction * r) % C).astype(np.int64)


def interleave(bits: np.ndarray, R: int, C: int, direction: int = 1) -> np.ndarray:
    x = np.asarray(bits)
    P = R * C
    x = np.concatenate([x, np.zeros((-len(x)) % P, dtype=x.dtype)])
    return x.reshape(-1, P)[:, permutation(R, C, direction)].reshape(-1)


def deinterleave(bits: np.ndarray, R: int, C: int, direction: int = 1, offset: int = 0) -> np.ndarray:
    """Whole blocks from `offset`; bits or LLRs (dtype preserved)."""
    x = np.asarray(bits)[offset:]
    P = R * C
    nb = len(x) // P
    if nb == 0:
        return x[:0]
    inv = np.empty(P, dtype=np.int64)
    inv[permutation(R, C, direction)] = np.arange(P)
    return x[:nb * P].reshape(nb, P)[:, inv].reshape(-1)
