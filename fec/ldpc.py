"""
LDPC codes: construction, systematic encoding, normalised min-sum decoding, codeword
alignment by syndrome, and blind parity-check recovery (via fec/dual.py).

Library
-------
* "LDPC 802.11 n=648 r3/4 (Z=27)": the IEEE 802.11 quasi-cyclic code whose prototype matrix
  is reproduced from the MathWorks ldpcEncode / ldpcDecode documentation (block size 27,
  codeword 648, information 486). Entry s >= 0 of the prototype = Z x Z identity cyclically
  shifted right by s; -1 = zero block.
* "LDPC regular (3,6) n=96 [project]" and "LDPC regular (3,6) n=504 [project]": regular
  Gallager-style codes generated here from a fixed seed, no duplicate columns, few 4-cycles
  (NOT standard codes);
  used to exercise decoding and the blind parity-check recovery.
Other standards (802.11 rate 1/2 / 2/3 / 5/6, 802.16e, DVB-S2, 5G NR) are not included: their
matrices can be added with `from_prototype` / `from_dense` when a verified copy is available.

Encoding: H is reduced over GF(2) with pivots taken from the LAST columns first, so the
information positions are the first non-pivot columns (for the 802.11 codes, the first k
columns, as in the standard). codeword[info] = u, codeword[parity] = M u.

Decoding: layered-free (flooding) normalised min-sum, alpha = 0.75, early stop when the
syndrome is zero. LLR convention ln P(0)/P(1).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from fec.dual import rref

PROTO_80211_648_R34 = [
    [16, 17, 22, 24, 9, 3, 14, -1, 4, 2, 7, -1, 26, -1, 2, -1, 21, -1, 1, 0, -1, -1, -1, -1],
    [25, 12, 12, 3, 3, 26, 6, 21, -1, 15, 22, -1, 15, -1, 4, -1, -1, 16, -1, 0, 0, -1, -1, -1],
    [25, 18, 26, 16, 22, 23, 9, -1, 0, -1, 4, -1, 4, -1, 8, 23, 11, -1, -1, -1, 0, 0, -1, -1],
    [9, 7, 0, 1, 17, -1, -1, 7, 3, -1, 3, 23, -1, 16, -1, -1, 21, -1, 0, -1, -1, 0, 0, -1],
    [24, 5, 26, 7, 1, -1, -1, 15, 24, 15, -1, 8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1, 0, 0],
    [2, 2, 19, 14, 24, 1, 15, 19, -1, 21, -1, 2, -1, 24, -1, 3, -1, 2, 1, -1, -1, -1, -1, 0],
]


@dataclass
class LDPCCode:
    name: str
    H: np.ndarray                     # dense uint8 (m x n), full row rank not required
    standard: bool
    info_pos: np.ndarray = field(default=None, repr=False)
    parity_pos: np.ndarray = field(default=None, repr=False)
    enc: np.ndarray = field(default=None, repr=False)      # parity = enc @ u (mod 2)
    _edges: tuple = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return self.H.shape[1]

    @property
    def k(self) -> int:
        self._prepare()
        return len(self.info_pos)

    @property
    def rate(self) -> float:
        return self.k / self.n

    def describe(self) -> str:
        w = self.H.sum(axis=1)
        return (f"LDPC (n={self.n}, k={self.k}), rate {self.rate:.3f}, {self.H.shape[0]} checks of weight "
                f"{int(w.min())}-{int(w.max())}{'' if self.standard else ', project-generated (not a standard)'}")

    def _prepare(self):
        if self.info_pos is not None:
            return
        n = self.n
        R, piv = rref(self.H, np.arange(n)[::-1])           # parity pivots from the right
        piv_set = set(piv)
        info = np.array([c for c in range(n) if c not in piv_set], dtype=np.int64)
        self.info_pos = info
        self.parity_pos = np.array(piv, dtype=np.int64)
        self.enc = R[:, info].astype(np.uint8)              # each pivot row: p = sum over info cols

    def edges(self):
        if self._edges is None:
            c, v = np.nonzero(self.H)
            order = np.lexsort((v, c))
            c, v = c[order], v[order]
            starts = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
            self._edges = (c, v, starts)
        return self._edges


def from_prototype(name: str, proto, Z: int, standard: bool) -> LDPCCode:
    P = np.asarray(proto)
    m, n = P.shape
    H = np.zeros((m * Z, n * Z), dtype=np.uint8)
    I = np.arange(Z)
    for r in range(m):
        for c in range(n):
            s = P[r, c]
            if s >= 0:
                H[r * Z + I, c * Z + (I + s) % Z] = 1
    return LDPCCode(name, H, standard)


def regular_gallager(name: str, n: int, dv: int, dc: int, seed: int, tries: int = 25) -> LDPCCode:
    """Gallager construction: dv stacked bands, each a column permutation of a block-diagonal band
    of weight-dc rows. For each band, `tries` permutations are drawn and the one with the fewest
    column pairs sharing two or more checks (4-cycles) is kept; duplicate columns (which would
    create weight-2 codewords) are never accepted."""
    rng = np.random.default_rng(seed)
    m_band = n // dc
    band = np.zeros((m_band, n), dtype=np.uint8)
    for r in range(m_band):
        band[r, r * dc:(r + 1) * dc] = 1
    H = band.copy()
    for k in range(dv - 1):
        best = None
        for _try in range(tries):
            cand = np.vstack([H, band[:, rng.permutation(n)]])
            G = cand.T.astype(np.int32) @ cand.astype(np.int32)
            np.fill_diagonal(G, 0)
            if k == dv - 2 and G.max() >= dv:          # final matrix: two identical columns
                continue
            cycles = int((G >= 2).sum())
            if best is None or cycles < best[0]:
                best = (cycles, cand)
                if cycles == 0:
                    break
        if best is None:
            raise RuntimeError("could not avoid duplicate columns; try another seed")
        H = best[1]
    return LDPCCode(name, H, False)


LIBRARY: List[LDPCCode] = [
    from_prototype("LDPC 802.11 n=648 r3/4 (Z=27)", PROTO_80211_648_R34, 27, True),
    regular_gallager("LDPC regular (3,6) n=96 [project]", 96, 3, 6, 7),
    regular_gallager("LDPC regular (3,6) n=504 [project]", 504, 3, 6, 11),
]
BY_NAME = {c.name: c for c in LIBRARY}


def get_code(name: str) -> LDPCCode:
    return BY_NAME[name]


def encode(bits: np.ndarray, code: LDPCCode) -> np.ndarray:
    code._prepare()
    u = np.asarray(bits, dtype=np.uint8) & 1
    k = code.k
    u = np.concatenate([u, np.zeros((-len(u)) % k, dtype=np.uint8)]).reshape(-1, k)
    cw = np.zeros((len(u), code.n), dtype=np.uint8)
    cw[:, code.info_pos] = u
    cw[:, code.parity_pos] = (u.astype(np.int64) @ code.enc.T.astype(np.int64)) % 2
    return cw.reshape(-1)


def syndrome_fraction(hard: np.ndarray, code: LDPCCode, offset: int = 0) -> Tuple[float, int]:
    """(fraction of unsatisfied checks over whole codewords from offset, checks evaluated)."""
    h = np.asarray(hard[offset:], dtype=np.int64) & 1
    B = len(h) // code.n
    if B == 0:
        return 0.5, 0
    cw = h[:B * code.n].reshape(B, code.n)
    active = cw.any(axis=1)
    if not active.any():
        return 0.5, 0
    s = (cw[active] @ code.H.T.astype(np.int64)) % 2
    return float(s.mean()), int(s.size)


def align(hard: np.ndarray, code: LDPCCode, max_offset: Optional[int] = None,
          max_codewords: int = 4) -> Tuple[int, float, float]:
    """Codeword alignment (blind LDPC frame synchronisation): the offset with the fewest
    unsatisfied checks over up to `max_codewords` codewords (all-zero codewords skipped).
    Returns (offset, fraction, z = sigma below the random 0.5). Computed on the edge list."""
    h = np.asarray(hard, dtype=np.int64) & 1
    c, v, starts = code.edges()
    n = code.n
    lim = min(n - 1, len(h) - n) if max_offset is None else min(max_offset, len(h) - n)
    best = (0, 0.5, 0.0)
    for o in range(max(lim, -1) + 1):
        B = min(max_codewords, (len(h) - o) // n)
        if B == 0:
            continue
        cw = h[o:o + B * n].reshape(B, n)
        cw = cw[cw.any(axis=1)]
        if len(cw) == 0:
            continue
        synd = np.add.reduceat(cw[:, v], starts, axis=1) % 2          # (B', m)
        f, cnt = float(synd.mean()), synd.size
        z = (0.5 - f) / (0.5 / np.sqrt(cnt))
        if z > best[2]:
            best = (o, f, z)
    return best


@dataclass
class LDPCDecode:
    bits: np.ndarray                 # information bits of all codewords
    codewords: int
    converged: int                   # codewords whose final syndrome is zero
    iterations: List[int]
    initial_unsatisfied: float
    corrected_bits: int


def decode_codeword(llr: np.ndarray, code: LDPCCode, max_iter: int = 50, alpha: float = 0.75):
    """Normalised min-sum. Returns (hard codeword, iterations used, converged)."""
    c, v, starts = code.edges()
    L = np.asarray(llr, dtype=np.float64)
    msg_cv = np.zeros(len(c))
    H64 = code.H.astype(np.int64)
    hard = (L < 0).astype(np.int64)
    for it in range(1, max_iter + 1):
        tot = L + np.bincount(v, msg_cv, minlength=code.n)
        m_vc = tot[v] - msg_cv
        sgn = np.where(m_vc < 0, -1.0, 1.0)
        mag = np.abs(m_vc)
        prod = np.multiply.reduceat(sgn, starts)
        min1 = np.minimum.reduceat(mag, starts)
        # second minimum per check: mask the first occurrence of min1
        grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(c)]))
        is_min = mag == min1[grp]
        first = np.zeros(len(c), dtype=bool)
        idx_first = np.minimum.reduceat(np.where(is_min, np.arange(len(c)), len(c)), starts)
        first[idx_first] = True
        min2 = np.minimum.reduceat(np.where(first, np.inf, mag), starts)
        other_min = np.where(first, min2[grp], min1[grp])
        msg_cv = alpha * prod[grp] * sgn * other_min
        tot = L + np.bincount(v, msg_cv, minlength=code.n)
        hard = (tot < 0).astype(np.int64)
        if not ((H64 @ hard) % 2).any():
            return hard.astype(np.uint8), it, True
    return hard.astype(np.uint8), max_iter, False


def decode(llrs: np.ndarray, code: LDPCCode, offset: int = 0, max_iter: int = 50) -> LDPCDecode:
    code._prepare()
    x = np.asarray(llrs, dtype=np.float64)[offset:]
    B = len(x) // code.n
    info, its, conv, corr = [], [], 0, 0
    unsat = []
    for b in range(B):
        l = x[b * code.n:(b + 1) * code.n]
        h0 = (l < 0).astype(np.int64)
        unsat.append(float(((code.H.astype(np.int64) @ h0) % 2).mean()))
        cw, it, ok = decode_codeword(l, code, max_iter)
        its.append(it)
        conv += int(ok)
        corr += int(np.count_nonzero(cw != h0))
        info.append(cw[code.info_pos])
    bits = np.concatenate(info) if info else np.zeros(0, dtype=np.uint8)
    return LDPCDecode(bits, B, conv, its, float(np.mean(unsat)) if unsat else 0.5, corr)


def code_from_checks(name: str, H_est: np.ndarray) -> LDPCCode:
    """Wrap a blindly recovered parity-check matrix (fec/dual.py) as a decodable code. The
    information positions are the leftmost information set of H_est."""
    return LDPCCode(name, np.asarray(H_est, dtype=np.uint8), False)
