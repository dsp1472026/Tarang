"""
Pulse shaping and interpolation helpers.

All pulse functions use *symbol-normalised time* (t in units of the symbol
period T), so they work for integer and fractional samples-per-symbol alike.
"""

import numpy as np


def rrc_pulse(t: np.ndarray, beta: float) -> np.ndarray:
    """
    Continuous root-raised-cosine impulse response evaluated at t (in symbols).

    Unit-energy in continuous time (integral of h^2 dt == 1 for T == 1), so when
    sampled at `sps` samples/symbol, sum(h^2) ~= sps.
    """
    t = np.asarray(t, dtype=np.float64)
    h = np.empty_like(t)
    eps = 1e-8

    at_zero = np.abs(t) < eps
    h[at_zero] = 1.0 - beta + 4.0 * beta / np.pi

    if beta > 0:
        at_sing = np.abs(np.abs(t) - 1.0 / (4.0 * beta)) < eps
        h[at_sing] = (beta / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
    else:
        at_sing = np.zeros_like(at_zero)

    rest = ~(at_zero | at_sing)
    tr = t[rest]
    num = np.sin(np.pi * tr * (1.0 - beta)) + 4.0 * beta * tr * np.cos(np.pi * tr * (1.0 + beta))
    den = np.pi * tr * (1.0 - (4.0 * beta * tr) ** 2)
    h[rest] = num / den
    return h


def rrc_taps(beta: float, sps: float, span: int = 10) -> np.ndarray:
    """
    Unit-energy RRC FIR taps for (possibly fractional) samples-per-symbol.

    The filter is symmetric with an odd number of taps; group delay is
    (len(taps) - 1) / 2 samples.
    """
    half = int(np.ceil(span * sps / 2.0))
    n = np.arange(-half, half + 1)
    h = rrc_pulse(n / sps, beta)
    return h / np.sqrt(np.sum(h ** 2))


def matched_filter(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """
    Apply a symmetric FIR with group-delay compensation.

    Uses full convolution and trims the delay so output[n] aligns with input[n];
    unlike mode='same' on even-length inputs, this is exact for odd-length taps.
    """
    delay = (len(taps) - 1) // 2
    y = np.convolve(x, taps, mode="full")
    return y[delay:delay + len(x)]


def cubic_interp(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    4-tap cubic Lagrange (Farrow-structure) interpolation of x at fractional indices t.

    Indices outside [1, len(x) - 3] are clamped to the valid range, which is
    acceptable because callers only sample well inside the burst.
    """
    t = np.asarray(t, dtype=np.float64)
    n0 = np.floor(t).astype(np.int64)
    mu = t - n0
    n0 = np.clip(n0, 1, len(x) - 3)
    xm1, x0, x1, x2 = x[n0 - 1], x[n0], x[n0 + 1], x[n0 + 2]
    # Lagrange basis on nodes {-1, 0, 1, 2}
    c_m1 = -mu * (mu - 1.0) * (mu - 2.0) / 6.0
    c_0 = (mu + 1.0) * (mu - 1.0) * (mu - 2.0) / 2.0
    c_1 = -(mu + 1.0) * mu * (mu - 2.0) / 2.0
    c_2 = (mu + 1.0) * mu * (mu - 1.0) / 6.0
    return c_m1 * xm1 + c_0 * x0 + c_1 * x1 + c_2 * x2
