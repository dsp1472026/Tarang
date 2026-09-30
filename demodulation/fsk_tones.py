"""
Tone-correlator FSK demodulator over explicit (M, h) hypotheses.

The blind FSK receiver fits the tone model from the instantaneous frequency. At few samples per
symbol a continuous-phase h = 0.5 2FSK (MSK-like) signal - the 'CubeSat UHF' preset of the Space
Downlink Simulator: 96 kHz, 19.2 kBd, 5 samples/symbol - shows smeared, multi-level frequency
tracks and was fitted as 4FSK with h = 0.17, so every bit was wrong. When that happens the file-frame
search calls this demodulator with a handful of standard hypotheses; the one whose decisions contain
the frame's sync word wins.

Per hypothesis: tones f_k = (k - (M - 1) / 2) h Rs around the band centre; each symbol interval is
correlated with every tone (non-coherent energy); the timing offset is the one of SEARCH_OFFSETS
fractions of a symbol that maximises the mean best-tone energy share.
"""

from types import SimpleNamespace
from typing import Iterator, Optional, Tuple

import numpy as np

SEARCH_OFFSETS = 8
H_CANDIDATES = (0.5, 1.0)
M_CANDIDATES = (2, 4, 8)
MAX_FIT_SAMPLES = 1 << 18


def _at(z: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Running sums z (tones x samples+1) at fractional positions (linear interpolation). Rounding the
    window edges to whole samples made the result flip with the 10th decimal of sps: at a half-sample
    offset np.round alternates (round half to even), so 5.0 decoded and 4.9999999988 did not."""
    pos = np.clip(pos, 0, z.shape[1] - 1.000001)
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    return z[:, i] * (1 - f) + z[:, i + 1] * f


def tone_decisions(x: np.ndarray, sps: float, M: int, h: float, center: float) -> Tuple[np.ndarray, float]:
    """(tone positions 0..M-1 per symbol, mean best-tone energy share) for one hypothesis."""
    x = np.asarray(x, dtype=np.complex128)
    n = np.arange(len(x))
    y = x * np.exp(-2j * np.pi * center * n)
    rs = 1.0 / sps
    tones = (np.arange(M) - (M - 1) / 2.0) * h * rs
    z = np.stack([np.concatenate([[0], np.cumsum(y * np.exp(-2j * np.pi * f * n))]) for f in tones])
    best = None
    fit = z[:, : MAX_FIT_SAMPLES + 1]
    for k in range(SEARCH_OFFSETS):
        off = k * sps / SEARCH_OFFSETS
        starts = off + np.arange(int((min(len(x), MAX_FIT_SAMPLES) - off - 1) // sps)) * sps
        e = np.abs(_at(fit, starts + sps) - _at(fit, starts)) ** 2
        share = float(np.mean(e.max(axis=0) / (e.sum(axis=0) + 1e-30)))
        if best is None or share > best[0]:
            best = (share, off)
    share, off = best
    starts = off + np.arange(int((len(x) - off - 1) // sps)) * sps
    e = np.abs(_at(z, starts + sps) - _at(z, starts)) ** 2
    return e.argmax(axis=0).astype(np.int64), share


def hypotheses(x: np.ndarray, sps: float, center: float, fitted_m: Optional[int] = None,
               fitted_h: Optional[float] = None) -> Iterator[Tuple[str, SimpleNamespace]]:
    """(name, receiver-like result with .positions / .llrs / .hard_bits) per (M, h) hypothesis."""
    hs = list(H_CANDIDATES) + ([float(fitted_h)] if fitted_h and all(abs(fitted_h - v) > 0.05 for v in H_CANDIDATES)
                               else [])
    ms = sorted(set(M_CANDIDATES) | ({int(fitted_m)} if fitted_m else set()))
    for M in ms:
        for h in hs:
            pos, share = tone_decisions(x, sps, M, h, center)
            k = int(np.log2(M))
            bits = ((pos[:, None] >> np.arange(k - 1, -1, -1)[None, :]) & 1).astype(np.uint8).ravel()
            yield (f"tone correlator {M}FSK h={h:.2f} (energy share {share:.2f})",
                   SimpleNamespace(positions=pos, hard_bits=bits, llrs=1.0 - 2.0 * bits, modulation=f"{M}FSK",
                                   variant=f"{M}FSK h={h:.2f}", share=share, M=M, h=h))
