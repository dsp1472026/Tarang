"""
L5 symbol-timing recovery for linear modulations.

Two stages:
 1. Oerder & Meyr feedforward estimator: the phase of the symbol-rate spectral
    line of |mf|^2 gives the timing phase directly. It is non-data-aided and
    carrier-blind, and it works on bursts too short for a loop to converge.
    The symbol rate is refined on the matched-filter output at the same time.
 2. Gardner timing-error-detector loop (2 samples/symbol, cubic Farrow
    interpolator), initialised from stage 1, which tracks residual clock drift
    on long signals. With a good initialisation it holds lock on short bursts.
"""

from dataclasses import dataclass

import numpy as np

from dsp.filters import cubic_interp
from estimation.linear_mod_params import cyclic_line


@dataclass
class TimingEstimate:
    sps: float           # refined samples/symbol
    tau: float           # sample index of a symbol centre (any one; others at tau + k*sps)
    line_strength: float # |line| / sum|mf|^2, a timing-SNR proxy in [0, 1]


def oerder_meyr(mf: np.ndarray, sps_init: float, refine: bool = True) -> TimingEstimate:
    y = np.abs(mf) ** 2
    alpha0 = 1.0 / sps_init
    if refine:
        # +-0.5% search: covers the coarse estimator's error and several hundred ppm of clock offset.
        grid = alpha0 * (1.0 + np.linspace(-5e-3, 5e-3, 401))
        alpha = float(grid[np.argmax(cyclic_line(y - y.mean(), grid))])
    else:
        alpha = alpha0
    n = np.arange(len(y))
    c = np.sum(y * np.exp(-2j * np.pi * alpha * n))
    # |mf|^2 peaks at n = tau + k/alpha  ->  c ~ exp(-j 2 pi alpha tau)
    tau = (-np.angle(c) / (2 * np.pi)) / alpha
    return TimingEstimate(1.0 / alpha, float(tau % (1.0 / alpha)), float(np.abs(c) / np.sum(y)))


@dataclass
class TimingResult:
    symbols: np.ndarray       # 1 sample/symbol, complex
    strobes: np.ndarray       # fractional sample index of every symbol
    sps_track: np.ndarray     # loop's instantaneous samples/symbol per symbol
    ted_error: np.ndarray     # Gardner error per symbol (diagnostic)
    method: str


def symbol_strobes(tau: float, sps: float, start: float, stop: float) -> np.ndarray:
    """All instants tau + k*sps with start - sps/2 <= t < stop - sps/2 (one per symbol period)."""
    k0 = int(np.ceil((start - sps / 2 - tau) / sps))
    k1 = int(np.floor((stop - sps / 2 - tau) / sps))
    return tau + np.arange(k0, k1 + 1) * sps


def gardner_loop(mf: np.ndarray, est: TimingEstimate, start: float, stop: float,
                 loop_bw: float = 0.004, damping: float = 0.707) -> TimingResult:
    """
    Gardner TED: e_k = Re{ conj(y(t_k - T/2)) * (y(t_k) - y(t_{k-1})) } / P

    e > 0 means the strobe is late. PI loop filter in normalised units
    (Bn*T = loop_bw), TED gain normalised by the running symbol power P so the
    loop gain does not depend on the input level.
    """
    theta = loop_bw / (damping + 1.0 / (4.0 * damping))
    d = 1.0 + 2.0 * damping * theta + theta ** 2
    k_ted = 2.0  # approximate Gardner S-curve slope for unit-power RRC signals
    kp = 4.0 * damping * theta / d / k_ted
    ki = 4.0 * theta ** 2 / d / k_ted

    strobes0 = symbol_strobes(est.tau, est.sps, start, stop)
    t = float(strobes0[0])
    sps_nom = est.sps
    integ = 0.0
    p_avg = float(np.mean(np.abs(mf[int(max(start, 0)):int(stop)]) ** 2)) + 1e-12

    out, strobes, sps_track, errs = [], [], [], []
    prev = cubic_interp(mf, np.array([t - sps_nom]))[0]
    last = stop - sps_nom / 2
    while t < last:
        cur = cubic_interp(mf, np.array([t, t - sps_nom / 2]))
        y_k, y_mid = cur[0], cur[1]
        e = float(np.real(np.conj(y_mid) * (y_k - prev))) / p_avg
        integ += ki * e
        step = sps_nom * (1.0 - (kp * e + integ))
        out.append(y_k)
        strobes.append(t)
        sps_track.append(step)
        errs.append(e)
        prev = y_k
        t += step
    return TimingResult(np.array(out), np.array(strobes), np.array(sps_track), np.array(errs), "gardner")


def feedforward_symbols(mf: np.ndarray, est: TimingEstimate, start: float, stop: float) -> TimingResult:
    strobes = symbol_strobes(est.tau, est.sps, start, stop)
    return TimingResult(cubic_interp(mf, strobes), strobes, np.full(len(strobes), est.sps),
                        np.zeros(len(strobes)), "oerder-meyr")
