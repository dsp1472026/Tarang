"""
Rectangular block interleaver and blind period / alignment estimation.

Interleaver (R rows x C columns, period P = R*C), per P-bit block:
    write row by row, read column by column      out = blk.reshape(R, C).T.ravel()
("write columns, read rows" on R x C is the same permutation as "write rows, read columns"
on C x R, so searching ordered (R, C) pairs covers both orientations.)
The transmitter zero-pads the last block, so an interleaved burst is a whole number of blocks.

Blind estimation
----------------
* Burel & Gautier (2009): reshape the stream into rows of L bits; coded data keeps its
  parity constraints inside a row when L is a multiple of the interleaver period and the
  rows are aligned with block starts, so rank_GF(2) < L ("rank deficiency"). Random data
  gives full rank (P(deficiency >= 1) ~ 2^-(rows - L)).
* Wee, Choi & Jeong (2021): the ranks of several L x L matrices are compared with the
  random-matrix rank distribution by a Kolmogorov-Smirnov statistic D = max |F_obs - F_rand|.
Both need rows > L, i.e. N > L^2 bits; for larger interleavers the period is hypothesised
from the burst length (a whole number of blocks) and validated by FEC / sync / CRC evidence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from deinterleaving.gf2 import random_rank_pmf, rank


def interleave(bits: np.ndarray, R: int, C: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    P = R * C
    x = np.concatenate([bits, np.zeros((-len(bits)) % P, dtype=np.uint8)])
    return x.reshape(-1, R, C).transpose(0, 2, 1).reshape(-1)


def deinterleave_perm(R: int, C: int) -> np.ndarray:
    """Index map for one block: deinterleaved[j] = received[perm[j]]."""
    return np.arange(R * C).reshape(C, R).T.ravel()


def deinterleave(bits: np.ndarray, R: int, C: int, offset: int = 0) -> np.ndarray:
    """De-interleave whole blocks starting at `offset` (a trailing partial block is dropped)."""
    x = np.asarray(bits)[offset:]
    P = R * C
    nb = len(x) // P
    if nb == 0:
        return x[:0]
    return x[:nb * P].reshape(nb, C, R).transpose(0, 2, 1).reshape(-1)


# ---------------------------------------------------------------------------
# Rank-based evidence
# ---------------------------------------------------------------------------

def rank_deficiency(bits: np.ndarray, L: int, offset: int = 0, max_rows: Optional[int] = None) -> Tuple[int, int]:
    """(L - rank, rows) of the matrix whose rows are consecutive L-bit segments from `offset`."""
    x = np.asarray(bits)[offset:]
    rows = len(x) // L
    if max_rows:
        rows = min(rows, max_rows)
    if rows <= L:
        return 0, rows
    M = x[:rows * L].reshape(rows, L)
    return L - rank(M), rows


def rank_profile(bits: np.ndarray, L_max: int = 64, margin: int = 16) -> Dict[int, float]:
    """Normalised rank deficiency (L - rank)/L for L = 2..L_max (only where rows >= L + margin)."""
    out = {}
    for L in range(2, L_max + 1):
        d, rows = rank_deficiency(bits, L, 0, max_rows=2 * L + margin)
        if rows < L + margin:
            break
        out[L] = d / L
    return out


def period_offset_map(bits: np.ndarray, periods: List[int], margin: int = 16) -> Dict[int, np.ndarray]:
    """For each candidate period P, normalised deficiency at every offset 0..P-1 (Burel's heat map)."""
    out = {}
    for P in periods:
        vals = np.zeros(P)
        for o in range(P):
            d, rows = rank_deficiency(bits, P, o, max_rows=2 * P + margin)
            vals[o] = d / P if rows >= P + margin else 0.0
        out[P] = vals
    return out


def ks_score(bits: np.ndarray, L: int, offset: int = 0) -> Optional[float]:
    """Wee-Choi-Jeong K-S statistic between the ranks of L x L blocks and the random rank law.
    None when fewer than 3 matrices fit."""
    x = np.asarray(bits)[offset:]
    nm = len(x) // (L * L)
    if nm < 3:
        return None
    ranks = [rank(x[i * L * L:(i + 1) * L * L].reshape(L, L)) for i in range(nm)]
    pmf = random_rank_pmf(L, L)
    F_rand = np.cumsum(pmf)
    F_obs = np.array([np.mean(np.array(ranks) <= r) for r in range(L + 1)])
    return float(np.max(np.abs(F_obs - F_rand)))


@dataclass
class PeriodEstimate:
    period: Optional[int]
    offset: int
    deficiency: float                     # normalised, at (period, offset)
    profile: Dict[int, float]             # L -> normalised deficiency (offset 0)
    ks: Optional[float]
    heat: Dict[int, np.ndarray]           # period -> deficiency per offset
    candidates: List[Tuple[int, float]]   # (L, deficiency) above the random threshold


def estimate_period(bits: np.ndarray, L_max: int = 64) -> PeriodEstimate:
    """Blind period estimate: smallest L whose multiples all show rank deficiency, then the
    alignment offset that maximises the deficiency (Burel & Gautier)."""
    prof = rank_profile(bits, L_max)
    cands = [(L, d) for L, d in prof.items() if d > 0]
    if not cands:
        return PeriodEstimate(None, 0, 0.0, prof, None, {}, [])
    # fundamental: the smallest L such that most tested multiples also show deficiency
    Ls = [L for L, _ in cands]
    best = None
    for L in Ls:
        mults = [m for m in range(2 * L, max(prof) + 1, L)]
        hit = np.mean([prof.get(m, 0) > 0 for m in mults]) if mults else 1.0
        if hit >= 0.75:
            best = L
            break
    if best is None:
        best = max(cands, key=lambda c: c[1])[0]
    heat = period_offset_map(bits, [best])
    off = int(np.argmax(heat[best]))
    return PeriodEstimate(best, off, float(heat[best][off]), prof, ks_score(bits, best, off), heat, cands)


# ---------------------------------------------------------------------------
# Hypotheses from the burst length
# ---------------------------------------------------------------------------

def dimension_hypotheses(n_bits: int, slack: int = 24, min_dim: int = 2, max_dim: int = 256,
                         max_period: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """(R, C, offset) such that a whole number of R*C blocks fits the burst within `slack` bits.
    The transmitter pads to whole blocks, so the interleaved span is k*R*C <= n_bits."""
    out = set()
    max_period = max_period or n_bits
    for R in range(min_dim, max_dim + 1):
        for C in range(min_dim, max_dim + 1):
            P = R * C
            if P > max_period or P > n_bits:
                continue
            k = n_bits // P
            spare = n_bits - k * P
            if k >= 1 and spare <= slack:
                for o in range(0, min(spare, slack) + 1):
                    out.add((R, C, o))
    return sorted(out, key=lambda t: (t[2], t[0] * t[1], t[0]))
