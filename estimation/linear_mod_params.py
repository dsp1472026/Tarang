"""
L3 parameter estimation for single-carrier linear modulations (PSK now, QAM later).

All estimators work in normalised units (cycles/sample, samples/symbol) so they
never need an absolute sample rate; the caller converts to Hz only when the
sample rate is actually known.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Spectral occupancy: carrier centre and -3 dB bandwidth
# ---------------------------------------------------------------------------

@dataclass
class SpectrumEstimate:
    cfo_norm: float          # carrier centre, cycles/sample (midpoint of the -3 dB band)
    bw3_norm: float          # -3 dB bandwidth, cycles/sample
    snr_db: float            # plateau-to-floor ratio of the PSD
    confidence: float


def estimate_spectrum(x: np.ndarray) -> SpectrumEstimate:
    """
    Carrier centre and -3 dB bandwidth from a Welch PSD.

    Key property used later: the PSD of RRC-shaped symbols is a raised cosine,
    which is exactly half its plateau at |f| = Rs/2 for ANY roll-off. So the
    -3 dB bandwidth is a roll-off-independent, data-independent estimate of the
    symbol rate, and its midpoint an estimate of the carrier offset.
    Periodic or biased payloads (e.g. repeated ASCII) turn the PSD into a line
    spectrum with several dB of ripple once the Welch segment is longer than the
    data period. So a first low-resolution pass (64-sample segments) measures
    the bandwidth, and the second pass uses segments of ~8 symbols: fine enough
    for accurate band edges, coarse enough to average the data lines out.
    """
    first = _half_power_band(x, 64)
    sps1 = 1.0 / max(first.bw3_norm, 1.0 / 256)
    nseg = int(2 ** np.round(np.log2(np.clip(8 * sps1, 32, 1024))))
    return _half_power_band(x, nseg)


def _half_power_band(x: np.ndarray, nperseg: int) -> SpectrumEstimate:
    from scipy.signal import medfilt

    nfft = int(min(nperseg, 2 ** np.floor(np.log2(max(len(x) // 4, 16)))))
    f, p = welch(x, nperseg=nfft, noverlap=nfft // 2, return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    ps = medfilt(p, 3)
    floor = float(np.percentile(ps, 10))
    k_pk = int(np.argmax(ps))
    ref = float(ps[k_pk])
    half = floor + 0.5 * (ref - floor)

    # Contiguous region around the peak that stays above the half-power level.
    lo = k_pk
    while lo > 0 and ps[lo - 1] >= half:
        lo -= 1
    hi = k_pk
    while hi < len(ps) - 1 and ps[hi + 1] >= half:
        hi += 1

    def cross(k_in, k_out):  # linear interpolation of the half-power crossing
        if k_out < 0 or k_out >= len(ps):
            return f[k_in]
        a, b = ps[k_in], ps[k_out]
        w = (a - half) / (a - b) if a != b else 0.5
        return f[k_in] + w * (f[k_out] - f[k_in])

    f_lo, f_hi = cross(lo, lo - 1), cross(hi, hi + 1)
    snr = 10 * np.log10(max(ref / max(floor, 1e-20) - 1, 1e-3))
    return SpectrumEstimate(float((f_lo + f_hi) / 2), float(f_hi - f_lo), float(snr),
                            float(np.clip(snr / 20.0, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Symbol rate (-3 dB bandwidth + cyclostationary square-law line)
# ---------------------------------------------------------------------------

@dataclass
class SymbolRateEstimate:
    sps: float               # samples per symbol
    rate_norm: float         # symbols per sample (= 1 / sps)
    bw3_rate_norm: float     # coarse estimate from the -3 dB bandwidth
    line_snr_db: float       # cyclic line height above the cyclic-spectrum median
    confidence: float


def cyclic_line(y: np.ndarray, alpha: np.ndarray, chunk: int = 2 ** 21) -> np.ndarray:
    """|sum_n y[n] exp(-j 2 pi alpha n)| on a grid of cycle frequencies (memory-bounded)."""
    n = np.arange(len(y))
    alpha = np.asarray(alpha, dtype=np.float64)
    rows = max(1, chunk // max(len(y), 1))
    out = np.empty(len(alpha))
    for i in range(0, len(alpha), rows):
        out[i:i + rows] = np.abs(np.exp(-2j * np.pi * np.outer(alpha[i:i + rows], n)) @ y)
    return out


def estimate_symbol_rate(x: np.ndarray, bw3_norm: float, search: float = 0.15) -> SymbolRateEstimate:
    """
    Two-step symbol-rate estimator.

    1. Coarse: Rs ~ -3 dB bandwidth (see estimate_spectrum).
    2. Fine: square-law cyclostationary line of |x|^2 searched only within
       +-`search` of the coarse value. Searching the whole cycle-frequency axis
       is unreliable: periodic/biased payloads create data-induced cyclic lines
       (observed at 3/4 and 1/2 of Rs for repeated ASCII) that can exceed the
       true symbol-rate line. The bandwidth prior removes that failure mode.
    """
    y = np.abs(x) ** 2
    y = y - y.mean()
    # Zero-padded FFT (>= 4x oversampled): the grid must be finer than the line
    # width (~1/N), otherwise the true line can fall between grid points.
    nfft = int(2 ** np.ceil(np.log2(4 * len(y))))
    spec = np.abs(np.fft.rfft(y, nfft))
    freqs = np.arange(len(spec)) / nfft
    win = np.flatnonzero((freqs >= bw3_norm * (1 - search)) & (freqs <= bw3_norm * (1 + search))
                         & (freqs > 1.0 / 128) & (freqs < 0.5))
    if len(win) == 0:
        return SymbolRateEstimate(1.0 / bw3_norm, bw3_norm, bw3_norm, 0.0, 0.0)
    k = win[np.argmax(spec[win])]
    # Dense DTFT refinement within +-1 FFT bin.
    fine = freqs[k] + np.linspace(-1.0, 1.0, 41) / nfft
    fl = cyclic_line(y, fine)
    alpha = float(fine[np.argmax(fl)])

    # Line SNR against the cyclic spectrum over the whole search window.
    line_snr = 20 * np.log10(spec[k] / (np.median(spec[win]) + 1e-20))
    conf = float(np.clip((line_snr - 3.0) / 12.0, 0.0, 1.0))
    return SymbolRateEstimate(1.0 / alpha, alpha, bw3_norm, float(line_snr), conf)


def cyclic_peaks(x: np.ndarray, k: int = 6, sps_min: float = 2.2, sps_max: float = 64.0) -> list:
    """Cycle frequencies (symbols/sample) of the k strongest, mutually distinct |x|^2 lines."""
    y = np.abs(x) ** 2
    y = y - y.mean()
    nfft = int(2 ** np.ceil(np.log2(8 * len(y))))
    spec = np.abs(np.fft.rfft(y, nfft))
    freqs = np.arange(len(spec)) / nfft
    idx = np.flatnonzero((freqs >= 1.0 / sps_max) & (freqs <= 1.0 / sps_min))
    order = idx[np.argsort(spec[idx])[::-1]]
    picks = []
    for i in order:
        if all(abs(freqs[i] - p) > 0.02 * p for p in picks):
            picks.append(float(freqs[i]))
        if len(picks) == k:
            break
    return picks


@dataclass
class RateHypothesis:
    sps: float
    quality: float     # lower is better (e.g. amplitude CV of strobed PSK symbols)
    source: str


def select_symbol_rate(x: np.ndarray, bw3_norm: float, quality_fn, rel_tol: float = 1.1,
                       abs_tol: float = 0.01, extra_sources=()) -> tuple[SymbolRateEstimate, list]:
    """
    Hypothesis test over symbol-rate candidates.

    Candidates: the -3 dB/cyclic estimate plus the strongest cyclic lines over
    the whole range (the -3 dB estimate is wrong when the payload is far from
    white, e.g. differentially encoded ASCII with long constant-phase runs).
    Each candidate is refined locally and scored by `quality_fn(sps)` (strobe
    the matched-filter output and measure constellation compactness).

    `extra_sources` are further signals (e.g. the unfiltered burst) whose cyclic
    lines are added as candidates.

    Rule: keep the spectral (-3 dB + cyclic) estimate if its score is within
    tolerance of the best. Otherwise take the HIGHEST rate within tolerance of
    the best, because integer sub-rates (sps*2, *3, ...) strobe valid symbols
    too and score as well as the truth.
    """
    base = estimate_symbol_rate(x, bw3_norm)
    cands = [(base.rate_norm, "-3dB+cyclic")]
    # Strong lines may sit at a sub-multiple of the true rate (few transitions),
    # so each line also proposes its 2x and 3x rates.
    # Each candidate is refined on the signal whose cyclic spectrum proposed it.
    raw = [(a * h, sig, "cyclic line" if h == 1 else f"{h}x cyclic line")
           for sig in (x, *extra_sources) for a in cyclic_peaks(sig, k=8) for h in (1, 2, 3)]
    for a, sig, src in raw:
        if not (1.0 / 64 <= a <= 1.0 / 2.2):
            continue
        refined = estimate_symbol_rate(sig, a, search=0.02).rate_norm
        if all(abs(refined - c) > 0.01 * c for c, _ in cands):
            cands.append((refined, src))
    hyps = [RateHypothesis(1.0 / a, float(quality_fn(1.0 / a)), src) for a, src in cands]
    best_q = min(h.quality for h in hyps)
    tol = best_q * rel_tol + abs_tol
    if hyps[0].quality <= tol:
        # The spectral estimate is consistent with the best hypothesis: keep it. At low
        # SNR all scores are noise-dominated and a double-rate candidate would otherwise
        # slip inside the tolerance.
        chosen = hyps[0]
    else:
        chosen = min((h for h in hyps if h.quality <= tol), key=lambda h: h.sps)
    if chosen is not hyps[0]:
        # Candidates are already refined on their own source signal; re-refining on `x`
        # could drift to a neighbouring line if `x` lacks this one (e.g. filtered away).
        base = SymbolRateEstimate(chosen.sps, 1.0 / chosen.sps, base.bw3_rate_norm, base.line_snr_db,
                                  base.confidence)
    return base, hyps


# ---------------------------------------------------------------------------
# Roll-off (fit of the raised-cosine PSD)
# ---------------------------------------------------------------------------

@dataclass
class RolloffEstimate:
    beta: float
    fit_residual: float
    candidates: dict   # beta -> normalised residual


def raised_cosine_psd(f: np.ndarray, rate: float, beta: float) -> np.ndarray:
    """|H_RC(f)| for a raised-cosine spectrum (the PSD of RRC-shaped symbols)."""
    af = np.abs(f)
    lo, hi = (1 - beta) * rate / 2, (1 + beta) * rate / 2
    out = np.zeros_like(af)
    out[af <= lo] = 1.0
    mid = (af > lo) & (af <= hi)
    if beta > 0:
        out[mid] = 0.5 * (1 + np.cos(np.pi / (beta * rate) * (af[mid] - lo)))
    return out


def estimate_rolloff(x: np.ndarray, sps: float,
                     candidates=(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)) -> RolloffEstimate:
    """
    Least-squares fit P(f) ~ A * RC(f; beta) + N over the candidate roll-offs.

    x must already be frequency-centred. A and N are solved linearly per
    candidate; the beta with the smallest residual wins.
    """
    nfft = int(min(512, 2 ** np.floor(np.log2(max(len(x) // 4, 64)))))
    f, p = welch(x, nperseg=nfft, noverlap=nfft // 2, return_onesided=False, detrend=False)
    p = p / p.max()
    rate = 1.0 / sps
    res = {}
    for b in candidates:
        model = raised_cosine_psd(f, rate, b)
        a_mat = np.column_stack([model, np.ones_like(model)])
        coef, *_ = np.linalg.lstsq(a_mat, p, rcond=None)
        res[b] = float(np.sum((a_mat @ coef - p) ** 2) / np.sum(p ** 2))
    best = min(res, key=res.get)
    return RolloffEstimate(best, res[best], res)
