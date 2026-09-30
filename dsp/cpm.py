"""
Continuous-phase FSK (CPM) signal model shared by the FSK generator and receiver.

Conventions (time t in SYMBOL units, symbol k centred at c_k = k + 1/2):
  * Tone levels   a = 2k - M + 1  in {-(M-1), ..., -1, +1, ..., M-1},  k = tone position 0..M-1
  * Label mapping "gray": position k carries label gray(k); "natural": label k.
    Bits are MSB first. For M = 2: bit 0 -> lower tone, bit 1 -> upper tone.
  * Frequency     f(t) = (h/2) * sum_k a_k p(t - c_k)            [cycles/symbol]
                  so adjacent tones are h*Rs apart and MSK (h = 1/2) uses +-Rs/4.
  * Phase         phi(t) = pi*h * sum_k a_k P(t - c_k),  P(t) = integral of p, P(+inf) = 1
                  -> each symbol advances the phase by pi*h*a_k (continuous phase).
  * p(t) is a unit-height rectangle of width 1 ("rect", full response) or that
    rectangle convolved with a Gaussian of time-bandwidth product BT ("gaussian",
    GFSK/GMSK, partial response).
"""

import numpy as np
from scipy.special import ndtr

from dsp.constellations import gray

SQRT_LN2 = np.sqrt(np.log(2.0))


def gaussian_sigma(bt: float) -> float:
    """Std-dev (in symbols) of the Gaussian filter with 3 dB bandwidth B = bt / T."""
    return SQRT_LN2 / (2.0 * np.pi * bt)


def pulse_span(pulse: str, bt: float = None, tol: float = 1e-4) -> int:
    """Smallest odd L (symbols) such that p(t) < tol outside |t| <= L/2."""
    if pulse == "rect":
        return 1
    L = 1
    while freq_pulse(np.array([L / 2.0]), pulse, bt)[0] >= tol:
        L += 2
    return L


def fsk_variant_name(order: int, h: float, pulse: str, bt: float = None, continuous: bool = True) -> str:
    """Canonical name: MSK / GMSK / GFSK / Sunde 2FSK / M-FSK (with CP tag where relevant)."""
    near = lambda a, b, tol=0.03: abs(a - b) <= tol
    if order == 2 and continuous and near(h, 0.5):
        return "GMSK" if pulse == "gaussian" else "MSK"
    if pulse == "gaussian":
        return "GFSK" if order == 2 else f"{order}GFSK"
    return f"{order}FSK"


def freq_pulse(t: np.ndarray, pulse: str, bt: float = None) -> np.ndarray:
    """p(t): unit-height frequency pulse centred at 0 (t in symbols)."""
    t = np.asarray(t, dtype=np.float64)
    if pulse == "rect":
        return ((t >= -0.5) & (t < 0.5)).astype(np.float64)
    s = gaussian_sigma(bt)
    return ndtr((t + 0.5) / s) - ndtr((t - 0.5) / s)


def phase_pulse(t: np.ndarray, pulse: str, bt: float = None) -> np.ndarray:
    """P(t) = integral_{-inf}^{t} p: 0 before the pulse, 1 after it."""
    t = np.asarray(t, dtype=np.float64)
    if pulse == "rect":
        return np.clip(t + 0.5, 0.0, 1.0)
    s = gaussian_sigma(bt)

    def G(u):  # integral of the standard normal CDF: u*Phi(u) + phi(u)
        return u * ndtr(u) + np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)

    return s * (G((t + 0.5) / s) - G((t - 0.5) / s))


def levels_from_labels(labels: np.ndarray, order: int, mapping: str = "gray") -> np.ndarray:
    """Symbol labels -> tone levels a = 2k - M + 1."""
    labels = np.asarray(labels, dtype=np.int64)
    if mapping == "gray":
        pos = np.empty(order, dtype=np.int64)
        pos[gray(np.arange(order))] = np.arange(order)     # label -> position
        k = pos[labels]
    elif mapping == "natural":
        k = labels
    else:
        raise ValueError(mapping)
    return 2 * k - order + 1


def labels_from_positions(k: np.ndarray, mapping: str = "gray") -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    return gray(k) if mapping == "gray" else k


def cpm_phase(levels: np.ndarray, t_sym: np.ndarray, h: float, pulse: str, bt: float = None) -> np.ndarray:
    """
    phi(t) = pi*h * sum_k a_k P(t - c_k) at arbitrary symbol-unit times t_sym (sorted).

    Symbols whose pulse has fully passed contribute pi*h*a_k; those still inside
    their pulse contribute pi*h*a_k*P(.). Exact for any sampling grid, so
    fractional samples-per-symbol, timing offsets and clock drift need no resampling.
    """
    levels = np.asarray(levels, dtype=np.float64)
    if pulse == "rect":
        # Full response: phase = pi*h*(sum of completed symbols + a_k * elapsed fraction of symbol k).
        csum = np.concatenate([[0.0], np.cumsum(levels)])
        k = np.clip(np.floor(t_sym).astype(np.int64), 0, len(levels))
        frac = np.clip(t_sym - k, 0.0, 1.0)
        cur = np.where(k < len(levels), levels[np.minimum(k, len(levels) - 1)], 0.0)
        return np.pi * h * (csum[k] + cur * frac * (t_sym >= 0))
    # Partial response: only symbols within +-H of floor(t) are inside their pulse;
    # older ones are complete (P = 1), newer ones have not started (P = 0).
    L = pulse_span(pulse, bt)
    H = L // 2 + 1
    N = len(levels)
    csum = np.concatenate([[0.0], np.cumsum(levels)])
    k0 = np.floor(t_sym).astype(np.int64)
    J = k0[:, None] + np.arange(-H, H + 1)[None, :]                   # (n, 2H+1) symbol indices
    valid = (J >= 0) & (J < N)
    a = np.where(valid, levels[np.clip(J, 0, N - 1)], 0.0)
    Pv = phase_pulse(t_sym[:, None] - J - 0.5, pulse, bt)
    done = csum[np.clip(k0 - H, 0, N)]                                 # symbols before the window
    return np.pi * h * (done + np.sum(a * Pv, axis=1))


def cpm_frequency(levels: np.ndarray, t_sym: np.ndarray, h: float, pulse: str, bt: float = None) -> np.ndarray:
    """Instantaneous frequency (cycles/symbol) f(t) = (h/2) sum_k a_k p(t - c_k)."""
    levels = np.asarray(levels, dtype=np.float64)
    L = pulse_span(pulse, bt)
    half = L / 2.0
    out = np.zeros(len(t_sym))
    for k, a in enumerate(levels):
        c = k + 0.5
        lo = np.searchsorted(t_sym, c - half)
        hi = np.searchsorted(t_sym, c + half, side="right")
        if hi > lo:
            out[lo:hi] += a * freq_pulse(t_sym[lo:hi] - c, pulse, bt)
    return 0.5 * h * out
