"""
GF(2) linear algebra on bit-packed rows (Gauss-Jordan elimination with 64-bit words).

rank(M) for an r x L binary matrix costs ~ min(r, L) pivots x r x ceil(L/64) word XORs,
so a few-hundred-column matrix takes milliseconds.
"""

import numpy as np


def pack_rows(M: np.ndarray) -> np.ndarray:
    """(r, L) 0/1 -> (r, W) uint64, bit j of the row at word j // 64, position j % 64."""
    M = np.asarray(M, dtype=np.uint8) & 1
    r, L = M.shape
    W = (L + 63) // 64
    pad = np.zeros((r, W * 64), dtype=np.uint8)
    pad[:, :L] = M
    b = np.packbits(pad.reshape(r, W, 8, 8)[:, :, :, ::-1], axis=-1, bitorder="big")   # (r, W, 8, 1)
    return b.reshape(r, W * 8).view("<u8").reshape(r, W)


def rank(M: np.ndarray) -> int:
    """Rank over GF(2)."""
    M = np.asarray(M)
    if M.size == 0:
        return 0
    A = pack_rows(M).copy()
    r, W = A.shape
    L = M.shape[1]
    rk = 0
    for col in range(L):
        if rk == r:
            break
        w, bit = divmod(col, 64)
        mask = np.uint64(1) << np.uint64(bit)
        rows = np.flatnonzero((A[rk:, w] & mask) != 0)
        if len(rows) == 0:
            continue
        p = rk + rows[0]
        if p != rk:
            A[[rk, p]] = A[[p, rk]]
        others = np.flatnonzero((A[:, w] & mask) != 0)
        others = others[others != rk]
        if len(others):
            A[others] ^= A[rk]
        rk += 1
    return rk


def random_rank_pmf(n: int, m: int) -> np.ndarray:
    """P(rank = r), r = 0..min(n, m), of a uniformly random n x m binary matrix."""
    q = 2.0
    k = min(n, m)
    logp = np.full(k + 1, -np.inf)
    for r in range(k + 1):
        # |{rank r}| = prod_{i<r} (q^n - q^i)(q^m - q^i) / (q^r - q^i), divided by q^{nm}
        s = 0.0
        for i in range(r):
            s += np.log2(1 - q ** (i - n)) + np.log2(1 - q ** (i - m)) - np.log2(1 - q ** (i - r))
        s += r * (n + m - r) - n * m
        logp[r] = s
    p = 2.0 ** logp
    return p / p.sum()
