"""
Constellation-agnostic symbol-rate check for linear modulations (PSK and QAM).

The PSK front end scores rate hypotheses by the amplitude CV of the strobes
(constant envelope). That is the right score for PSK but not for QAM, whose
strobes have an intrinsic CV of 0.3-0.4: on short QAM bursts (25-70 symbols)
a wrong, slower rate with a handful of strobes can look "more constant".

This module scores a rate by how well the strobe AMPLITUDES fall on the rings
of some PSK/QAM constellation (carrier-free, so the scan is cheap):

    residual(sps) = min over hypotheses, gains | |s|/g - nearest ring | / local ring gap

~0 at the right rate (strobes on a lattice), ~0.3 for scattered strobes.
Integer sub-rates (2x, 3x sps) strobe valid symbols too, so the fastest rate
within tolerance of the best is taken. The caller confirms the result with the
full lattice fit (phase + amplitude) before adopting it.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from dsp.constellations import get_constellation
from dsp.filters import matched_filter, rrc_taps
from synchronization.symbol_timing import feedforward_symbols, oerder_meyr

HYPOTHESES = ("BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "64QAM", "256QAM")
GAINS = np.linspace(0.8, 1.25, 91)
MIN_STROBES = 20


def _rings(name: str) -> Tuple[np.ndarray, float]:
    r = np.unique(np.round(np.abs(get_constellation(name).points), 9))
    spacing = float(np.min(np.diff(r))) if len(r) > 1 else float(r[0])
    return r, spacing


_RINGS = {}
for _h in HYPOTHESES:
    _r, _s = _rings(_h)
    key = tuple(np.round(_r, 6))
    if key not in _RINGS:              # all PSK orders share the unit ring
        _RINGS[key] = (_h, _r, _s)


@dataclass
class RateScan:
    sps: float                 # chosen (fastest within tolerance of the best)
    residual: float
    best_sps: float
    best_residual: float
    ring_hypothesis: str
    grid: np.ndarray
    residuals: np.ndarray
    candidates: List[float] = None   # shortlist (refined sps) for a full lattice test


def strobes(x: np.ndarray, sps: float, beta: float = 0.35, trim: int = 5) -> np.ndarray:
    mf = matched_filter(x, rrc_taps(beta, sps, span=10))
    om = oerder_meyr(mf, sps, refine=False)
    s = feedforward_symbols(mf, om, 0, len(mf)).symbols
    s = s[trim:-trim] if len(s) > 4 * trim else s
    # Contiguous core above half the median amplitude: the burst estimate of a short
    # capture can include noise-only guard samples (same rule as the PSK front end).
    idx = np.flatnonzero(np.abs(s) > 0.5 * np.median(np.abs(s)))
    return s[idx[0]:idx[-1] + 1] if len(idx) > 1 else s


def ring_residual(s: np.ndarray) -> Tuple[float, str]:
    """
    Distance of the strobe amplitudes to the nearest ring, in units of the LOCAL ring
    gap (the gap the amplitude falls in). Dense constellations (256QAM: 32 radii, some
    nearly coincident) would otherwise be judged against their smallest gap and never fit.
    """
    a = np.abs(s)
    a = a / (np.sqrt(np.mean(a ** 2)) + 1e-30)
    best = (np.inf, "")
    for name, r, _ in _RINGS.values():
        v = a[None, :] / GAINS[:, None]                       # (gains, symbols)
        if len(r) == 1:
            e = np.sqrt(np.mean((v - r[0]) ** 2, axis=1))
        else:
            k = np.clip(np.searchsorted(r, v), 1, len(r) - 1)
            lo, hi = r[k - 1], r[k]
            inside = (v > r[0]) & (v < r[-1])
            d = np.where(inside, np.minimum(v - lo, hi - v), np.where(v <= r[0], r[0] - v, v - r[-1]))
            gap = np.where(inside, hi - lo, np.where(v <= r[0], r[1] - r[0], r[-1] - r[-2]))
            e = np.sqrt(np.mean((d / gap) ** 2, axis=1))
        best = min(best, (float(np.min(e)), name))
    return best


def _minima(grid, e, k, min_sep=0.03):
    """Indices of the k lowest local minima of e, at least min_sep (relative) apart."""
    idx = [i for i in np.argsort(e)
           if (i == 0 or e[i] <= e[i - 1]) and (i == len(e) - 1 or e[i] <= e[i + 1])]
    out = []
    for i in idx:
        if all(abs(grid[i] / grid[j] - 1) > min_sep for j in out):
            out.append(i)
        if len(out) == k:
            break
    return out


def scan_rate(x: np.ndarray, sps_lo: float, sps_hi: float, max_samples: int = 1024,
              rel_tol: float = 1.5, abs_tol: float = 0.02, n_candidates: int = 4, n_coarse: int = 8) -> RateScan:
    """
    Coarse-to-fine: a coarse grid on a short segment (few strobes -> a wide minimum, so a
    coarse step cannot miss it) over the whole range, then fine grids on up to
    max_samples around the n_coarse best coarse minima.
    """
    x = x[:max_samples]
    sps_hi = min(sps_hi, len(x) / (MIN_STROBES + 10.0))
    sps_lo = min(sps_lo, sps_hi / 1.01)
    xc = x[:int(min(len(x), max(256, (MIN_STROBES + 12) * sps_hi)))]
    step_c = float(np.clip(0.08 / (len(xc) / sps_lo), 0.001, 0.01))
    grid_c = np.exp(np.arange(np.log(sps_lo), np.log(sps_hi), step_c))
    e_c = np.array([ring_residual(strobes(xc, sv))[0] for sv in grid_c])

    # A relative rate error e drifts the outermost of N strobes by e*N/2 symbols; the ring
    # fit of dense QAM tolerates ~0.02 symbol, so the fine grid is sized for that.
    step = float(np.clip(0.04 / (len(x) / sps_lo), 0.0003, 0.004))
    grid, res = [], []
    for i in _minima(grid_c, e_c, n_coarse):
        for sv in grid_c[i] * np.exp(np.arange(-2 * step_c, 2 * step_c + 1e-12, step)):
            grid.append(sv)
            res.append(ring_residual(strobes(x, sv)))
    order = np.argsort(grid)
    grid = np.asarray(grid)[order]
    res = [res[i] for i in order]
    e = np.array([r[0] for r in res])
    kb = int(np.argmin(e))
    ok = np.flatnonzero(e <= e[kb] * rel_tol + abs_tol)
    kc = int(ok[np.argmin(grid[ok])])
    picks = [kb, kc] + [i for i in _minima(grid, e, n_candidates + 2) if i not in (kb, kc)]
    cands = []
    for i in picks:
        if all(abs(grid[i] / c - 1) > 0.03 for c in cands):
            cands.append(float(grid[i]))
        if len(cands) == n_candidates:
            break
    return RateScan(float(grid[kc]), float(e[kc]), float(grid[kb]), float(e[kb]), res[kc][1], grid, e, cands)
