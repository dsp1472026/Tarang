"""
Phase and amplitude histogram features for the modulation family / order split.

    instantaneous phase       phi[n] = arg x[n]               in [-pi, pi]
    instantaneous amplitude   A[n]   = |x[n]|
    instantaneous frequency   f[n]   = d unwrap(phi) / dt / (2 pi)   (Hz)

    phase histogram     PSK: one cluster per constellation phase (BPSK 2, QPSK 4, 8PSK 8)
    amplitude histogram PSK/FSK: one level (constant envelope); QAM: one level per ring
                        (16QAM 3, 64QAM up to 9)

The IF histogram is NOT re-implemented here: it reuses
modulation.signal_features.if_histogram_peaks (the one the classifier already uses)
and only renormalises it to unit sum.

Both counts are only meaningful on a CFO-free, symbol-rate (or rectangular-pulse)
signal: a residual carrier offset rotates the phase histogram flat, and pulse
shaping between symbols fills the amplitude histogram in. On a raw capture they
enter the classifier only as weak evidence.

Peak counting (both histograms): smooth with a Gaussian kernel (sigma = 2 bins),
scale to a maximum of 1, then scipy.signal.find_peaks with a minimum spacing and a
minimum prominence RELATIVE to the largest peak (a sum-normalised 8PSK peak is only
~0.02 high, so an absolute prominence would not scale with the order). The phase
histogram is circular: it is smoothed with wrap-around and rolled so its lowest bin
sits at the edge, so a cluster at +-pi (BPSK) is counted once, not twice.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from modulation.signal_features import HIST_BINS, if_histogram_peaks

SMOOTH_SIGMA_BINS = 2.0
PHASE_BINS = 64
AMPLITUDE_BINS = 128
PSK_CLUSTER_COUNTS = (2, 4, 8)      # phase cluster counts that look like a PSK constellation


def _validate(iq: np.ndarray) -> np.ndarray:
    """1-D complex copy of iq; ValueError if empty or not finite."""
    x = np.asarray(iq)
    if x.size == 0:
        raise ValueError("iq is empty")
    x = x.astype(np.complex128).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("iq contains non-finite values (NaN or Inf)")
    return x


def _centres(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def _unit_sum(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    s = h.sum()
    return h / s if s > 0 else np.zeros_like(h)


def _count_peaks(h: np.ndarray, distance_bins: int, min_prominence: float, circular: bool) -> int:
    """Peaks of the smoothed histogram h, prominence relative to its maximum."""
    hs = gaussian_filter1d(np.asarray(h, dtype=float), SMOOTH_SIGMA_BINS, mode="wrap" if circular else "constant")
    if hs.max() <= 0:
        return 0
    hs = hs / hs.max()
    if circular:
        hs = np.roll(hs, -int(np.argmin(hs)))           # lowest bin at the edge: no peak straddles the wrap
    pk, _ = find_peaks(np.r_[0.0, hs, 0.0], distance=max(1, int(distance_bins)), prominence=min_prominence)
    return int(len(pk))


def instantaneous_frequency(iq: np.ndarray, fs: float) -> np.ndarray:
    """
    Instantaneous frequency in Hz, (1 / 2 pi) * d unwrap(arg iq) / dt, with dt = 1 / fs.

    Returns len(iq) - 1 values (first difference of the unwrapped phase).
    """
    x = _validate(iq)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a positive, finite sample rate")
    return np.diff(np.unwrap(np.angle(x))) * fs / (2.0 * np.pi)


def instantaneous_phase(iq: np.ndarray) -> np.ndarray:
    """Wrapped instantaneous phase arg(iq) in [-pi, pi]."""
    return np.angle(_validate(iq))


def instantaneous_amplitude(iq: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude |iq|."""
    return np.abs(_validate(iq))


def phase_histogram(iq: np.ndarray, n_bins: int = PHASE_BINS,
                    range_rad: Tuple[float, float] = (-np.pi, np.pi)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram of the instantaneous phase, normalised to sum 1.

    BPSK gives two clusters (0 and +-pi), QPSK four (+-pi/4, +-3pi/4), 8PSK eight.
    Returns (hist, bin_centers) with bin centres in radians.
    """
    hist, edges = np.histogram(instantaneous_phase(iq), bins=n_bins, range=range_rad)
    return _unit_sum(hist), _centres(edges)


def amplitude_histogram(iq: np.ndarray, n_bins: int = AMPLITUDE_BINS,
                        range_amp: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram of the instantaneous amplitude, normalised to sum 1.

    Constant-envelope signals (PSK, FSK) give one level, QAM one level per ring.
    range_amp defaults to (0, 1.05 x max amplitude). Returns (hist, bin_centers).
    """
    a = instantaneous_amplitude(iq)
    if range_amp is None:
        hi = 1.05 * float(a.max())
        range_amp = (0.0, hi if hi > 0 else 1.0)
    hist, edges = np.histogram(a, bins=n_bins, range=range_amp)
    return _unit_sum(hist), _centres(edges)


def count_phase_clusters(iq: np.ndarray, min_separation_rad: float = 0.3, min_prominence: float = 0.05) -> int:
    """
    Number of distinct clusters in the (circular) phase histogram.

    The histogram is smoothed (Gaussian, sigma = 2 bins, wrap-around) and scaled to a
    maximum of 1; peaks at least min_separation_rad apart with a prominence of at
    least min_prominence (relative to the largest peak) are counted.
    """
    hist, centres = phase_histogram(iq)
    width = 2.0 * np.pi / len(hist)
    return _count_peaks(hist, int(np.ceil(min_separation_rad / width)), min_prominence, circular=True)


def count_amplitude_levels(iq: np.ndarray, min_separation_ratio: float = 0.1, min_prominence: float = 0.03) -> int:
    """
    Number of distinct levels in the amplitude histogram.

    Levels must be at least min_separation_ratio x RMS amplitude apart and have a
    prominence of at least min_prominence relative to the largest level. PSK/FSK
    give 1, 16QAM 3, 64QAM up to 9 (two of its rings are only 0.08 x RMS apart).
    """
    hist, centres = amplitude_histogram(iq)
    a = instantaneous_amplitude(iq)
    rms = float(np.sqrt(np.mean(a ** 2)))
    width = centres[1] - centres[0] if len(centres) > 1 else 1.0
    return _count_peaks(hist, int(np.ceil(min_separation_ratio * rms / width)), min_prominence, circular=False)


def histogram_features(iq: np.ndarray, fs: float) -> Dict[str, Any]:
    """
    IF, phase and amplitude histograms plus the phase-cluster and amplitude-level counts.

    The IF histogram is modulation.signal_features.if_histogram_peaks (64 bins over the
    1-99 % IF range, lightly smoothed) renormalised to sum 1; with fewer than two
    samples it is all zeros. Keys: if_hist, if_bins (Hz), phase_hist, phase_bins (rad),
    amplitude_hist, amplitude_bins, phase_cluster_count, amplitude_level_count.
    """
    x = _validate(iq)
    f = instantaneous_frequency(x, fs)
    if len(f) > 0:
        hs, edges, _, _ = if_histogram_peaks(f, HIST_BINS)
        if_hist, if_bins = _unit_sum(hs), _centres(edges)
    else:
        if_hist, if_bins = np.zeros(HIST_BINS), np.zeros(HIST_BINS)
    ph, pb = phase_histogram(x)
    ah, ab = amplitude_histogram(x)
    return {"if_hist": if_hist, "if_bins": if_bins,
            "phase_hist": ph, "phase_bins": pb,
            "amplitude_hist": ah, "amplitude_bins": ab,
            "phase_cluster_count": int(count_phase_clusters(x)),
            "amplitude_level_count": int(count_amplitude_levels(x))}


def histogram_evidence(iq: np.ndarray, fs: float) -> List[Dict[str, Any]]:
    """
    Two evidence entries for the unified classifier, FAM_PHASE_HIST and FAM_AMP_HIST.

    Each dict has pillar, description, score in [0, 1] and details, plus source /
    metadata aliases so it maps one-to-one onto core.evidence.Evidence.
      FAM_PHASE_HIST  score 0.9 when the phase-cluster count is 2, 4 or 8 (PSK-like), else 0.1
      FAM_AMP_HIST    score 0.9 for one amplitude level (constant envelope, PSK/FSK),
                      0.5 for two, 0.2 for three or more (QAM), 0.0 when none was found
    """
    hf = histogram_features(iq, fs)
    nc, na = hf["phase_cluster_count"], hf["amplitude_level_count"]
    psk_like = nc in PSK_CLUSTER_COUNTS
    phase = {"pillar": "FAM_PHASE_HIST",
             "description": f"Phase histogram ({len(hf['phase_hist'])} bins): {nc} cluster(s) -> "
                            + ("power-of-two phase set (PSK-like)" if psk_like else "no PSK phase set"),
             "score": 0.9 if psk_like else 0.1,
             "details": {"phase_cluster_count": nc, "power_of_two": psk_like}}
    amp_score = {0: 0.0, 1: 0.9, 2: 0.5}.get(na, 0.2)
    amp = {"pillar": "FAM_AMP_HIST",
           "description": f"Amplitude histogram ({len(hf['amplitude_hist'])} bins): {na} level(s) -> "
                          + ("constant envelope (PSK/FSK)" if na == 1 else
                             "multi-level envelope (QAM-like)" if na >= 3 else "inconclusive"),
           "score": amp_score,
           "details": {"amplitude_level_count": na}}
    for e in (phase, amp):
        e["source"], e["metadata"] = e["pillar"], e["details"]
    return [phase, amp]
