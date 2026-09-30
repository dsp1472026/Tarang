"""
Timing-free DSP features for the FSK / PSK / QAM family split.

All features are computed on the burst after band isolation: coarse CFO removed
(occupied-band centre, so FSK tones are symmetric about 0) and a channel
filter over the occupied band.

    instantaneous amplitude   A[n] = |x[n]|
    instantaneous frequency   f[n] = arg(x[n] x*[n-1]) / (2 pi)   (cycles/sample)
    IF histogram peaks        FSK: one mode per tone; PSK/QAM: one mode at 0
    IF kurtosis               E[(f-mu)^4] / E[(f-mu)^2]^2
    cumulants C40, C42        normalised by (E|x|^2)^2
    envelope CV               std(A) / mean(A)
    cyclic tone pair          symmetric line pair in x (k=1) or x^2 (k=2): FSK tones

Two corrections to the textbook rules, both measured on this data set:
  * The IF of a pulse-shaped PSK/QAM signal has huge spikes where the envelope
    passes near zero (arg() of a tiny vector), so its RAW kurtosis is large
    (often > 10), while a multi-tone FSK IF is bounded and platykurtic
    (2-FSK ~ 1-2). So low kurtosis indicates FSK, not high. The histogram is
    built from samples whose amplitude is above 0.5 x RMS, which removes those
    spikes without affecting constant-envelope FSK.
  * Sample-level cumulants of RRC-shaped symbols differ from the symbol-level
    table (pulse overlap makes them more Gaussian); they are reported here but
    the PSK order is decided at the symbol level (psk_classifier).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from dsp.conditioning import channel_filter, detect_burst, normalize_power
from estimation.fsk_estimator import occupied_band

HIST_BINS = 64


@dataclass
class SignalFeatures:
    n: int
    band_center: float                 # cycles/sample
    bw: float                          # 97 % occupied bandwidth, cycles/sample
    snr_db: float
    env_cv: float
    if_kurtosis_raw: float
    if_kurtosis_gated: float
    if_peaks: int
    if_peak_freqs: List[float]         # cycles/sample, relative to band centre
    if_peak_heights: List[float]       # relative to the largest
    if_hist: np.ndarray = field(repr=False)
    if_edges: np.ndarray = field(repr=False)
    c40: float = 0.0                   # |C40| normalised
    c42: float = 0.0                   # C42 normalised (real)
    c20: float = 0.0                   # |E x^2| / E|x|^2
    tone_pairs: Dict[int, bool] = field(default_factory=dict)      # k -> symmetric cyclic line pair in x^k
    burst: tuple = (0, 0)
    if_center_density: float = 0.0    # smoothed IF histogram at 0 Hz relative to its maximum
    power_line: Dict[int, float] = field(default_factory=dict)     # k -> strongest single line of x^k (rho)

    def as_dict(self) -> Dict[str, float]:
        return {"env_cv": self.env_cv, "if_kurtosis_raw": self.if_kurtosis_raw,
                "if_kurtosis_gated": self.if_kurtosis_gated, "if_peaks": self.if_peaks,
                "|C40|": self.c40, "C42": self.c42, "|C20|": self.c20,
                "tone_pair_x1": float(self.tone_pairs.get(1, False)),
                "tone_pair_x2": float(self.tone_pairs.get(2, False)), "if_center_density": self.if_center_density,
                "power_line_x2": self.power_line.get(2, 0.0), "power_line_x4": self.power_line.get(4, 0.0),
                "snr_db": self.snr_db}


def _kurtosis(v: np.ndarray) -> float:
    v = v - v.mean()
    m2 = np.mean(v ** 2)
    return float(np.mean(v ** 4) / (m2 ** 2 + 1e-30))


def if_histogram_peaks(f: np.ndarray, bins: int = HIST_BINS, rel_height: float = 0.2):
    """Histogram of the IF (64 bins over the 1-99 % range), lightly smoothed; peaks >= 20 % of max."""
    lo, hi = np.percentile(f, [1, 99])
    pad = 0.1 * (hi - lo + 1e-12)
    hist, edges = np.histogram(f, bins=bins, range=(lo - pad, hi + pad))
    hs = gaussian_filter1d(hist.astype(float), 1.0, mode="constant")
    hs = hs / (hs.max() + 1e-30)
    pk, _ = find_peaks(np.r_[0, hs, 0], height=rel_height, prominence=0.1)
    pk = pk - 1
    centres = 0.5 * (edges[:-1] + edges[1:])
    return hs, edges, centres[pk].tolist(), hs[pk].tolist()


def cyclic_tone_pair(y: np.ndarray, k: int):
    """
    First-order (k=1) / higher-order (k=2) cyclic lines: the two strongest lines of
    z = |y| exp(j k arg y). FSK puts a SYMMETRIC PAIR of lines at +-k*tone (h = 1 FSK
    at k=1, MSK/h = 1/2 CPFSK at k=2); linear modulations give at most one line (at
    k*CFO, BPSK at k=2). Line strength rho = |Z(f)| / sum|z| (1 = pure tone).
    Returns (is_pair, (f1, rho1), (f2, rho2)) with frequencies in units of the y frequency.
    """
    z = np.abs(y) * np.exp(1j * k * np.angle(y))
    n = len(z)
    nfft = int(2 ** np.ceil(np.log2(max(8 * n, 1024))))
    S = np.abs(np.fft.fft(z, nfft)) / (np.sum(np.abs(z)) + 1e-30)
    fr = np.fft.fftfreq(nfft)
    k1 = int(np.argmax(S))
    guard = max(2, int(3 * nfft / n))                       # exclude the main lobe of line 1
    S2 = S.copy()
    idx = (np.arange(k1 - guard, k1 + guard + 1)) % nfft
    S2[idx] = 0
    k2 = int(np.argmax(S2))
    (f1, r1), (f2, r2) = (fr[k1] / k, S[k1]), (fr[k2] / k, S[k2])
    floor = max(0.2, 3.0 / np.sqrt(n))
    sym = abs(f1 + f2) < 0.15 * abs(f1 - f2) + 1e-12
    return bool(r2 >= floor and r2 >= 0.5 * r1 and sym), (float(f1), float(r1)), (float(f2), float(r2))


def compute_features(samples: np.ndarray) -> SignalFeatures:
    x, _ = normalize_power(np.asarray(samples, dtype=np.complex128))
    band0 = occupied_band(x)
    burst = detect_burst(x, band0.center, band0.bw99)
    xb = x[burst.start:burst.stop]  # DC deliberately kept: a biased payload mean is signal
    band = occupied_band(xb)
    n = np.arange(len(xb))
    y = channel_filter(xb * np.exp(-2j * np.pi * band.center * n), 0.0, band.bw99, 1.0)
    e = max(1, len(y) // 20)
    yc = y[e:-e] if len(y) > 4 * e else y
    amp = np.abs(yc)
    env_cv = float(np.std(amp) / (np.mean(amp) + 1e-20))

    f = np.angle(yc[1:] * np.conj(yc[:-1])) / (2 * np.pi)
    rms = np.sqrt(np.mean(amp ** 2))
    gate = np.minimum(amp[1:], amp[:-1]) > 0.5 * rms
    fg = f[gate] if gate.sum() > 32 else f
    hs, edges, pf, ph = if_histogram_peaks(fg)

    z = yc / (rms + 1e-30)
    m20 = np.mean(z ** 2)
    m21 = np.mean(np.abs(z) ** 2)
    m40 = np.mean(z ** 4)
    m42 = np.mean(np.abs(z) ** 4)
    c40 = (m40 - 3 * m20 ** 2) / m21 ** 2
    c42 = (m42 - np.abs(m20) ** 2 - 2 * m21 ** 2) / m21 ** 2

    pairs = {k: cyclic_tone_pair(yc, k)[0] for k in (1, 2)}
    power = {}
    for k in (2, 4):
        pair, (f1, r1), _ = cyclic_tone_pair(yc, k)
        power[k] = 0.0 if pair else r1
    centres = 0.5 * (edges[:-1] + edges[1:])
    k0 = int(np.argmin(np.abs(centres)))
    return SignalFeatures(len(xb), band.center, band.bw99, band.snr_db, env_cv, _kurtosis(f), _kurtosis(fg),
                          len(pf), pf, ph, hs, edges, float(np.abs(c40)), float(np.real(c42)),
                          float(np.abs(m20) / m21), pairs, (burst.start, burst.stop), float(hs[k0]), power)
