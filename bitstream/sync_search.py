"""
Sync-word search in a bit stream (Chiani & Martini 2006, Massey 1972).

For every candidate position tau of an L-bit sync word s (s_i = +1 for bit 0, -1 for bit 1)
and received LLRs l (ln P0/P1; hard bits give l = +-1):

    hard correlation   rho = (L - 2 d_H) / L             (d_H = Hamming distance)
    soft correlation   sum s_i l_i / sum |l_i|
    LRT (optimal, AWGN, equiprobable data):
                       Lambda = sum [ s_i l_i / 2 - ln cosh(l_i / 2) ]
      = log P(r | sync at tau) / P(r | random data) for exact LLRs (Chiani's LRT)
    GLRT (no SNR / data model; high-SNR form):
                       G = sum [ s_i l_i - |l_i| ] / 2  = - sum_{sign disagrees} |l_i|

The hard threshold is the largest Hamming distance whose chance of a random emulation
anywhere in the stream stays below `p_fa` (binomial tail x number of positions).
Occurrences that recur give the frame period.
"""

from dataclasses import dataclass, field
from math import comb
from typing import Dict, List, Optional

import numpy as np

from synthetic.framing import SYNC_BITS


def _hex_bits(h: str) -> np.ndarray:
    v = int(h, 16)
    n = 4 * len(h)
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


SYNC_LIBRARY: Dict[str, np.ndarray] = {
    "SIH test frame PN127/96": SYNC_BITS,
    "CCSDS ASM 0x1ACFFC1D": _hex_bits("1ACFFC1D"),
    "Telemetry marker 0xD3917E42": _hex_bits("D3917E42"),
    "0xFAF33420 (CCSDS rate-1/2 ASM)": _hex_bits("FAF33420"),
    "PCM sync 0xEB90": _hex_bits("EB90"),
    "0xF628 (IRIG/PCM)": _hex_bits("F628"),
    "Barker-13": np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8),
    "Header 0xAA55": _hex_bits("AA55"),
}


def hard_threshold(L: int, n_positions: int, p_fa: float = 1e-3) -> int:
    """Largest e with n_positions * P(Binomial(L, 1/2) <= e) <= p_fa (-1 = never)."""
    tail, e_max = 0.0, -1
    for e in range(L + 1):
        tail += comb(L, e) / 2.0 ** L
        if n_positions * tail > p_fa:
            break
        e_max = e
    return e_max


@dataclass
class SyncCandidate:
    name: str
    length: int
    offset: int                     # best bit offset
    hamming: int
    hard_corr: float
    soft_corr: float
    lrt: float
    glrt: float
    threshold: int                  # max Hamming distance accepted
    detected: bool
    inverted: bool                  # matched the complemented word (polarity / 180 deg ambiguity)
    occurrences: List[int] = field(default_factory=list)
    period: Optional[int] = None    # most common spacing between occurrences (frame length)
    p_emulation: float = 1.0        # chance of a random match this good anywhere in the stream

    @property
    def confidence(self) -> str:
        if not self.detected:
            return "None"
        return "High" if self.p_emulation < 1e-6 else "Medium" if self.p_emulation < 1e-3 else "Low"


def _metrics(llr_win: np.ndarray, s: np.ndarray):
    sl = llr_win * s[None, :]
    a = np.abs(llr_win)
    soft = sl.sum(axis=1) / (a.sum(axis=1) + 1e-12)
    lrt = (sl / 2 - (np.logaddexp(a / 2, -a / 2) - np.log(2))).sum(axis=1)
    glrt = ((sl - a) / 2).sum(axis=1)
    return soft, lrt, glrt


def search(bits: np.ndarray, llrs: Optional[np.ndarray] = None, library: Optional[Dict[str, np.ndarray]] = None,
           p_fa: float = 1e-3, min_length: int = 16) -> List[SyncCandidate]:
    """Search every library word (and its complement) at every bit offset."""
    bits = np.asarray(bits, dtype=np.uint8) & 1
    llr = np.asarray(llrs, dtype=np.float64) if llrs is not None and len(llrs) >= len(bits) else \
        1.0 - 2.0 * bits.astype(np.float64)
    llr = llr[:len(bits)]
    out = []
    for name, word in (library or SYNC_LIBRARY).items():
        L = len(word)
        if L < min_length or len(bits) < L:
            continue
        win_b = np.lib.stride_tricks.sliding_window_view(bits, L)
        win_l = np.lib.stride_tricks.sliding_window_view(llr, L)
        npos = len(win_b)
        thr = hard_threshold(L, 2 * npos, p_fa)           # x2: complement searched too
        best = None
        for inv in (False, True):
            w = word ^ 1 if inv else word
            d = np.count_nonzero(win_b != w[None, :], axis=1)
            k = int(np.argmin(d))
            if best is None or d[k] < best[0]:
                best = (int(d[k]), k, inv, d)
        dmin, k, inv, d = best
        w = word ^ 1 if inv else word
        s = 1.0 - 2.0 * w.astype(np.float64)
        soft, lrt, glrt = _metrics(win_l[k:k + 1], s)
        occ = [int(i) for i in np.flatnonzero(d <= thr)] if thr >= 0 else []
        occ = _dedupe(occ, L)
        period = None
        if len(occ) >= 2:
            gaps = np.diff(occ)
            vals, cnt = np.unique(gaps, return_counts=True)
            period = int(vals[np.argmax(cnt)])
        tail = sum(comb(L, e) for e in range(dmin + 1)) / 2.0 ** L
        out.append(SyncCandidate(name, L, k, dmin, (L - 2 * dmin) / L, float(soft[0]), float(lrt[0]),
                                 float(glrt[0]), thr, thr >= 0 and dmin <= thr, inv, occ, period,
                                 min(1.0, 2 * npos * tail)))
    out.sort(key=lambda c: (not c.detected, c.p_emulation, -c.length))
    return out


def _dedupe(positions: List[int], L: int) -> List[int]:
    res = []
    for p in positions:
        if not res or p - res[-1] >= L:
            res.append(p)
    return res


def best_detection(cands: List[SyncCandidate]) -> Optional[SyncCandidate]:
    det = [c for c in cands if c.detected]
    return det[0] if det else None
