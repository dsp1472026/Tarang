"""
Binary linear block codes (Hamming family): systematic encoder, syndrome decoder,
and the parity-check test used as FEC evidence.

A correctly aligned codeword stream satisfies H c = 0 for every n-bit block, so the
fraction of blocks with a non-zero syndrome is ~ n x BER when right and ~ 1 - 2^-(n-k)
for random data or a wrong alignment.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


_MATS = {}


@dataclass(frozen=True)
class BlockCode:
    name: str
    n: int
    k: int
    P: tuple                       # k x (n-k) parity part of G = [I_k | P]

    @property
    def G(self) -> np.ndarray:
        if ("G", self.name) not in _MATS:
            _MATS[("G", self.name)] = np.hstack([np.eye(self.k, dtype=np.uint8), np.array(self.P, dtype=np.uint8)])
        return _MATS[("G", self.name)]

    @property
    def H(self) -> np.ndarray:
        if ("H", self.name) not in _MATS:
            _MATS[("H", self.name)] = np.hstack([np.array(self.P, dtype=np.uint8).T,
                                                 np.eye(self.n - self.k, dtype=np.uint8)])
        return _MATS[("H", self.name)]

    @property
    def rate(self) -> str:
        return f"{self.k}/{self.n}"

    def describe(self) -> str:
        return f"({self.n},{self.k}) binary linear block code, rate {self.rate}, systematic, corrects 1 error"


def _hamming(m: int) -> BlockCode:
    n, k = 2 ** m - 1, 2 ** m - 1 - m
    cols = [c for c in range(1, n + 1) if c & (c - 1)]           # non-powers of two -> data columns
    P = tuple(tuple((c >> (m - 1 - i)) & 1 for i in range(m)) for c in cols)
    return BlockCode(f"Hamming({n},{k})", n, k, P)


LIBRARY: List[BlockCode] = [_hamming(3), _hamming(4), _hamming(5)]
BY_NAME = {c.name: c for c in LIBRARY}


def get_code(name: str) -> BlockCode:
    return BY_NAME[name]


def encode(bits: np.ndarray, code: BlockCode) -> np.ndarray:
    u = np.asarray(bits, dtype=np.uint8) & 1
    u = np.concatenate([u, np.zeros((-len(u)) % code.k, dtype=np.uint8)])
    return ((u.reshape(-1, code.k).astype(np.int64) @ code.G) % 2).astype(np.uint8).ravel()


def syndrome_stats(hard: np.ndarray, code: BlockCode, offset: int = 0):
    """(fraction of non-zero blocks with a non-zero syndrome, number of non-zero blocks).
    All-zero blocks (padding) are skipped: they are codewords of every linear code."""
    h = np.asarray(hard[offset:], dtype=np.int64) & 1
    B = len(h) // code.n
    if B < 4:
        return 1.0, 0
    blocks = h[:B * code.n].reshape(B, code.n)
    blocks = blocks[blocks.any(axis=1)]
    if len(blocks) < 4:
        return 1.0, 0
    s = (blocks @ code.H.T.astype(np.int64)) % 2
    return float(np.mean(np.any(s, axis=1))), int(len(blocks))


def syndrome_fail_rate(hard: np.ndarray, code: BlockCode, offset: int = 0) -> float:
    return syndrome_stats(hard, code, offset)[0]


def decode(hard: np.ndarray, code: BlockCode, offset: int = 0):
    """Syndrome decoding (single-error correction). Returns (data bits, corrected, uncorrectable)."""
    h = np.asarray(hard[offset:], dtype=np.int64) & 1
    B = len(h) // code.n
    c = h[:B * code.n].reshape(B, code.n).copy()
    Ht = code.H.T.astype(np.int64)
    s = (c @ Ht) % 2
    cols = {tuple(code.H[:, j]): j for j in range(code.n)}
    corrected = bad = 0
    for i in np.flatnonzero(np.any(s, axis=1)):
        j = cols.get(tuple(s[i]))
        if j is None:
            bad += 1
        else:
            c[i, j] ^= 1
            corrected += 1
    return c[:, :code.k].astype(np.uint8).ravel(), corrected, bad
