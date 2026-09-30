"""
L1/L2 signal conditioning: power normalisation (AGC), burst detection and DC removal.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import firwin

from dsp.filters import matched_filter


def normalize_power(x: np.ndarray, target: float = 1.0) -> tuple[np.ndarray, float]:
    """Block AGC: scale so mean |x|^2 == target. Returns (scaled, applied_gain)."""
    p = float(np.mean(np.abs(x) ** 2))
    g = np.sqrt(target / p) if p > 0 else 1.0
    return x * g, g


@dataclass
class Burst:
    start: int              # first sample of the active region (inclusive)
    stop: int               # last sample (exclusive)
    noise_power: float      # in-band mean power of the inactive region (nan if none)
    signal_power: float     # in-band mean power of the active region
    snr_db: float           # in-band SNR estimate from the power split (nan if no noise-only region)
    whole_file: bool        # True if no inactive region was found


def channel_filter(x: np.ndarray, center: float, bw: float, scale: float = 0.7) -> np.ndarray:
    """Band-isolate [center - scale*bw, center + scale*bw] (cycles/sample) with a linear-phase FIR."""
    cutoff = float(np.clip(scale * bw, 0.01, 0.49))
    ntaps = int(min(255, 2 * int(4.0 / cutoff) + 1))
    h = firwin(ntaps, cutoff, fs=1.0)
    n = np.arange(len(x))
    lo = np.exp(-2j * np.pi * center * n)
    return matched_filter(x * lo, h) * np.conj(lo)


def detect_burst(x: np.ndarray, center: Optional[float] = None, bw: Optional[float] = None,
                 threshold_db: float = 6.0, contrast_db: float = 8.0) -> Burst:
    """
    Energy detector for a single burst.

    When the carrier centre / bandwidth are known (from the PSD), the signal is
    channel-filtered first, which raises the in-band SNR by roughly the
    oversampling ratio and makes low per-sample-SNR bursts detectable.

    Noise floor = 2nd percentile of the smoothed power. If the median is less
    than `contrast_db` above that floor, the signal is assumed to fill the whole
    capture (e.g. the legacy Test-Data files).
    """
    if center is not None and bw is not None and bw > 0:
        xf = channel_filter(x, center, bw)
        window = int(np.clip(4.0 / bw, 8, 256))     # ~4 symbols
    else:
        xf, window = x, 32
    p = np.abs(xf) ** 2
    ps = uniform_filter1d(p, size=min(window, len(p)), mode="nearest")
    floor = float(np.percentile(ps, 2))
    median = float(np.median(ps))

    if floor <= 0 or 10 * np.log10(median / floor) < contrast_db:
        return Burst(0, len(x), float("nan"), float(np.mean(p)), float("nan"), True)

    active = ps > floor * 10 ** (threshold_db / 10.0)
    idx = np.flatnonzero(active)
    start, stop = int(idx[0]), int(idx[-1]) + 1
    # The smoothing window blurs edges by ~window/2; widen so no symbol is lost.
    start = max(0, start - window // 2)
    stop = min(len(x), stop + window // 2)

    inactive = np.ones(len(x), dtype=bool)
    inactive[start:stop] = False
    if inactive.sum() < window:
        return Burst(0, len(x), float("nan"), float(np.mean(p)), float("nan"), True)
    noise_power = float(np.mean(p[inactive]))
    signal_power = float(np.mean(p[start:stop]))
    snr = 10 * np.log10(max(signal_power - noise_power, 1e-12) / noise_power)
    return Burst(start, stop, noise_power, signal_power, float(snr), False)


def estimate_dc(x: np.ndarray, burst: Burst) -> tuple[complex, str]:
    """
    DC (LO leakage) estimate.

    A plain mean is biased whenever the payload itself has a non-zero mean
    (ASCII text in BPSK/QPSK has a mean of 0.2-0.5 of the RMS amplitude), so
    only noise-only samples outside the burst are used. Without them the raw
    DC is left alone and the demodulator estimates it decision-directed in the
    symbol domain instead (see demodulation.psk_receiver).
    """
    if not burst.whole_file:
        outside = np.r_[x[:burst.start], x[burst.stop:]]
        return complex(np.mean(outside)), "mean of noise-only samples outside the burst"
    return 0j, "none in raw domain (no noise-only region; deferred to decision-directed symbol-domain estimate)"
