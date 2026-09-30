"""
L3 parameter estimation for FSK / CPFSK / MSK / GFSK / GMSK.

Everything is in normalised units (cycles/sample, samples/symbol, cycles/symbol)
until the caller converts with a KNOWN sample rate.

Pipeline pieces:
  occupied_band       PSD -> carrier centre and 97 % occupied bandwidth
  iq_balance          blind second-order IQ-imbalance correction (FSK is a proper signal)
  inst_freq           discriminator output f[n] (cycles/sample), located at n + 1/2
  estimate_rate       symbol rate from spectral lines of two transition-sensitive
                      IF transforms (difference, square law),
                      scored jointly; refined and timed by decision-directed
                      re-modulation (dd_match); candidates kept for backtracking
  if_strobes          mean frequency over the central part of each symbol (integrate & dump)
  fit_levels          equally spaced M-level fit (M, tone spacing = h, centre = CFO) with a
                      Gaussian-mixture log-likelihood per M (nested orders pay ln M)
  phase_continuity    IF spikes at symbol boundaries => phase-discontinuous FSK
  model_match         re-modulation test: per-symbol non-coherent correlation between the
                      received signal and a CPM model built from the decisions; ranks
                      pulse-shape (rect / Gaussian BT) and h hypotheses
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks, medfilt, welch

from dsp.cpm import cpm_phase


# ---------------------------------------------------------------------------
# Spectrum / band isolation / IQ balance
# ---------------------------------------------------------------------------

@dataclass
class Band:
    center: float        # cycles/sample
    bw99: float          # two-sided 97 % occupied bandwidth, cycles/sample (name kept for brevity)
    floor: float         # PSD noise floor (per bin)
    snr_db: float        # in-band SNR from the PSD
    psd_f: np.ndarray = field(repr=False, default=None)
    psd: np.ndarray = field(repr=False, default=None)


def occupied_band(x: np.ndarray) -> Band:
    nseg = int(min(1024, 2 ** np.floor(np.log2(max(len(x) // 8, 64)))))
    f, p = welch(x, nperseg=nseg, noverlap=nseg // 2, return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    floor = float(np.percentile(p, 10))
    noise = float(np.median(np.sort(p)[: len(p) // 2]))      # typical noise-bin level
    # Only bins clearly above the noise count, otherwise Welch fluctuations of the
    # floor accumulate over the whole band at low SNR.
    excess = np.clip(medfilt(p, 3) - 2.0 * noise, 0, None)
    c = np.cumsum(excess) / max(excess.sum(), 1e-20)
    # 97 % (not 99 %): rectangular CPFSK side lobes decay slowly and a 99 % band
    # sweeps in most of the noise floor.
    lo = f[np.searchsorted(c, 0.015)]
    hi = f[min(np.searchsorted(c, 0.985), len(f) - 1)]
    inband = (f >= lo) & (f <= hi)
    sig = float(np.sum(p[inband] - floor))
    noise = float(floor * np.count_nonzero(inband))
    snr = 10 * np.log10(max(sig, 1e-20) / max(noise, 1e-20))
    return Band(float((lo + hi) / 2), float(hi - lo + (f[1] - f[0])), floor, snr, f, p)


def psd_tone_peaks(band: Band, min_sep: float) -> List[float]:
    """Frequencies of PSD peaks >= 6 dB above the floor, separated by >= min_sep (evidence only)."""
    ps = medfilt(band.psd, 3)
    df = band.psd_f[1] - band.psd_f[0]
    idx, _ = find_peaks(ps, height=4 * band.floor, distance=max(1, int(min_sep / df)),
                        prominence=0.25 * (ps.max() - band.floor))
    return [float(band.psd_f[i]) for i in idx]


@dataclass
class IQBalance:
    applied: bool
    improper_ratio: float     # |E[x^2]| / E[|x|^2] before correction
    gain_db: float
    phase_deg: float


def iq_balance(x: np.ndarray, threshold: float = 0.01) -> tuple[np.ndarray, IQBalance]:
    """
    Blind IQ-imbalance correction by whitening I/Q (Gram-Schmidt + power equalisation).

    A CPFSK signal whose tones rotate through all phases is proper (E[x^2] = 0);
    receiver IQ imbalance makes it improper. Only valid for proper signals, so
    it is applied inside the FSK receiver, never to BPSK-like (improper) signals.
    """
    xc = x - x.mean()
    ratio = float(np.abs(np.mean(xc ** 2)) / np.mean(np.abs(xc) ** 2))
    i, q = x.real, x.imag
    pii, pqq, piq = np.mean(i * i), np.mean(q * q), np.mean(i * q)
    if pii <= 1e-20 or pqq <= 1e-20:                  # a rail is empty: nothing to balance
        return x, IQBalance(False, ratio, float("nan"), float("nan"))
    rho = piq / np.sqrt(pii * pqq)
    gain_db = 10 * np.log10(pqq / pii)
    phase_deg = float(np.degrees(np.arcsin(np.clip(rho, -1, 1))))
    if ratio < threshold:
        return x, IQBalance(False, ratio, float(gain_db), phase_deg)
    q2 = q - (piq / pii) * i                       # remove the I component leaking into Q
    q2 = q2 * np.sqrt(pii / np.mean(q2 * q2))       # equalise power
    return i + 1j * q2, IQBalance(True, ratio, float(gain_db), phase_deg)


def inst_freq(y: np.ndarray) -> np.ndarray:
    """f[n] = arg(y[n+1] conj(y[n])) / 2pi  (cycles/sample), centred at n + 1/2."""
    return np.angle(y[1:] * np.conj(y[:-1])) / (2 * np.pi)


# ---------------------------------------------------------------------------
# Symbol rate and boundary timing from tone transitions
# ---------------------------------------------------------------------------

@dataclass
class RateEstimate:
    sps: float
    boundary: float                 # sample index (float, relative to f[0]) of one symbol boundary
    match: float                    # decision-directed re-modulation match of the winner (0..1)
    candidates: List[Dict]          # every tested hypothesis (evidence / backtracking)


def _cyclic_lines(d: np.ndarray, k: int, sps_min: float, sps_max: float) -> List[tuple]:
    """(rate, normalised line height) of the k strongest distinct cyclic lines of d."""
    d = d - d.mean()
    nfft = int(2 ** np.ceil(np.log2(4 * len(d))))
    spec = np.abs(np.fft.rfft(d, nfft))
    freqs = np.arange(len(spec)) / nfft
    idx = np.flatnonzero((freqs >= 1.0 / sps_max) & (freqs <= 1.0 / sps_min))
    med = float(np.median(spec[idx])) + 1e-20
    out = []
    for i in idx[np.argsort(spec[idx])[::-1]]:
        if all(abs(freqs[i] - o[0]) > 0.03 * o[0] for o in out):
            # refine within +-1 bin by a dense DTFT
            n = np.arange(len(d))
            grid = freqs[i] + np.linspace(-1, 1, 21) / nfft
            vals = np.abs(np.exp(-2j * np.pi * np.outer(grid, n)) @ d)
            out.append((float(grid[np.argmax(vals)]), float(spec[i] / med)))
        if len(out) == k:
            break
    return out


def rate_candidates(f: np.ndarray, sps_min: float = 2.2, sps_max: float = 64.0) -> List[Dict]:
    """
    Candidate symbol rates from cyclic lines of several transition-sensitive
    transforms of the discriminator output, at several smoothing scales W:
      |fW[n+W] - fW[n-W]|  (smoothed IF difference: impulse train at transitions)
      (fW - mean)^2        (square-law line of the IF viewed as a PAM waveform;
                            strong for Gaussian-filtered GFSK/GMSK)
    Each line also proposes 2x, 3x, 1/2 and 1/3 of its rate (harmonic ambiguity).
    """
    from scipy.ndimage import uniform_filter1d
    lines = []
    for W in (1, 2, 4, 8, 16):
        if 4 * W >= len(f):
            break
        fw = uniform_filter1d(f, W)
        for name, d in ((f"|dIF| W={W}", np.abs(fw[2 * W:] - fw[:-2 * W])),
                        (f"IF^2 W={W}", (fw - fw.mean()) ** 2)):
            lines += [(r, hgt, name) for r, hgt in _cyclic_lines(d, 3, sps_min, sps_max)]
    cands = []
    for r, hgt, name in sorted(lines, key=lambda t: -t[1]):
        for mult, tag in ((1, ""), (2, " x2"), (3, " x3"), (0.5, " /2"), (1 / 3, " /3")):
            a = r * mult
            if 1.0 / sps_max <= a <= 1.0 / sps_min and all(abs(a - c["rate"]) > 0.005 * a for c in cands):
                cands.append(dict(rate=a, sps=1.0 / a, source=name + tag, line=hgt))
    return cands


def caf_lines(y: np.ndarray, k: int = 8, sps_min: float = 2.2, sps_max: float = 64.0,
              lags=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32)) -> List[tuple]:
    """
    Strongest cyclic-autocorrelation lines: for each lag L, z = y[n] conj(y[n-L]) is
    periodic in the symbol period (CPM phase-state lattice), so |FFT(z - mean)|^2,
    normalised and summed over lags, shows lines at multiples of Rs/2.
    Candidate generator only (data periodicities add lines too).
    """
    n = len(y)
    nfft = int(2 ** np.ceil(np.log2(4 * n)))
    tot = np.zeros(nfft)
    for L in lags:
        if L >= n // 4:
            break
        z = y[L:] * np.conj(y[:-L])
        S = np.abs(np.fft.fft(z - z.mean(), nfft)) ** 2
        S = S + S[::-1]
        tot += S / np.median(S)
    freqs = np.arange(nfft) / nfft
    idx = np.flatnonzero((freqs >= 1.0 / sps_max) & (freqs <= 1.0 / sps_min))
    out = []
    for i in idx[np.argsort(tot[idx])[::-1]]:
        if all(abs(freqs[i] - q) > 0.03 * q for q, _ in out):
            out.append((float(freqs[i]), float(tot[i] / np.median(tot[idx]))))
        if len(out) == k:
            break
    return out


def dd_match(y: np.ndarray, Phi: np.ndarray, sps: float, phase: float, pulse: str = "rect",
             bt: Optional[float] = None, max_sym: int = 300, continuous: bool = True):
    """
    Decision-directed re-modulation score of a (rate, timing) hypothesis.
    IF integrate-and-dump -> best M-level grid (M, h = spacing, CFO = centre) ->
    re-modulation match: continuous CPM model over fixed windows, or per-symbol
    tones for phase-discontinuous FSK. Returns (match, LevelFit, starts).
    """
    starts = boundaries_in(phase, sps, 0.0, float(len(y)))
    starts = starts[(starts >= 0) & (starts + sps <= len(y))]
    if len(starts) < 8:
        return -np.inf, None, starts
    if len(starts) > max_sym:
        c0 = (len(starts) - max_sym) // 2
        starts = starts[c0:c0 + max_sym]
    v = if_strobes(Phi, starts, sps)
    fits = [fit_levels(v, M, iters=10) for M in (2, 4, 8) if len(v) >= 3 * M]
    fit = max(fits, key=lambda q: q.loglik)
    lv = 2 * fit.positions - fit.order + 1
    n0, n1 = int(np.ceil(starts[0])), min(int(np.floor(starts[-1] + sps)), len(y))
    t = (np.arange(n0, n1) - starts[0]) / sps
    if continuous:
        # CPM model over fixed windows: phase continuity makes this sharply rate-sensitive.
        m = model_match(y[n0:n1], t, lv, fit.spacing, fit.center, pulse, bt)
    else:
        # Phase jumps: per-symbol non-coherent windows (bias negligible within ~1 % of rate).
        m = model_match_symbols(y[n0:n1], t, lv, fit.spacing, fit.center)
    return m, fit, starts


def _line_spectrum(d: np.ndarray, nfft: int) -> np.ndarray:
    """|FFT(d - mean)| normalised by its median over the whole axis."""
    S = np.abs(np.fft.rfft(d - d.mean(), nfft))
    return S / (np.median(S[1:]) + 1e-20)


def _line_at(S: np.ndarray, a: float, nfft: int) -> float:
    k = int(round(a * nfft))
    if k < 2 or k >= len(S) - 2:
        return 0.0
    return float(S[k - 2:k + 3].max())


def power_line_pair(y: np.ndarray, powers=(1, 2, 4)):
    """
    Power-law line test for CPM: if P*h is an integer, y^P contains discrete
    spectral lines at the (P-scaled) tone frequencies, adjacent ones exactly
    Rs apart (x^2 for MSK/GMSK, x^1 for Sunde FSK). Returns
    (rate, P, line1, line2, dominance) for the smallest P with a strong,
    dominant pair (2nd line >= 30 dB above the median, >= 2x the 3rd), else None.
    """
    from scipy.signal import find_peaks
    u = y / np.maximum(np.abs(y), 1e-9)
    nfft = int(2 ** np.ceil(np.log2(8 * len(u))))
    for P in powers:
        S = np.fft.fftshift(np.abs(np.fft.fft(u ** P, nfft)) ** 2)
        S = S / (np.median(S) + 1e-20)
        fr = np.fft.fftshift(np.fft.fftfreq(nfft))
        idx, _ = find_peaks(S, distance=max(1, nfft // 400))
        if len(idx) < 3:
            continue
        top = idx[np.argsort(S[idx])[::-1][:3]]
        p1, p2, p3 = S[top]
        if p2 >= 1e3 and p2 >= 2 * p3:
            f1, f2 = fr[top[0]], fr[top[1]]
            return float(abs(f1 - f2)), P, float(f1), float(f2), float(p2 / p3)
    return None


def estimate_rate(y: np.ndarray, f: np.ndarray, bw: float, sps_min: float = 2.2, sps_max: float = 64.0,
                  max_samples: int = 16384, continuous: bool = True) -> RateEstimate:
    """
    Symbol-rate estimate for (G)FSK / CPM from spectral lines, with evidence.

    Two transition-sensitive transforms of the IF of the channel-filtered burst,
    each at several smoothing scales W (1..8 samples, capped at ~0.6/bandwidth):
        T1 = |fW[n+W] - fW[n-W]|        tone transitions -> lines at k*Rs
        T2 = (fW - mean)^2              square-law line of the IF as a PAM waveform
    Each candidate line is scored by its mean normalised height over transforms
    and scales: the symbol rate persists across all of them, noise and
    data-induced lines do not. (The cyclic autocorrelation of y was tried and
    rejected: its k*Rs/2 lattice lines and data periodicities dominate.)
    A dominant power-law line pair (power_line_pair: exact for P*h integer, e.g.
    MSK/GMSK, where the IF lines are weakest) is ranked first. Candidates are
    kept in order for backtracking (the receiver retries if the sync word is
    not found).
    The winner is refined, and the boundary phase found, by a local
    decision-directed re-modulation scan (dd_match).
    """
    from scipy.ndimage import uniform_filter1d

    c0 = max(0, (len(y) - max_samples) // 2)
    y, f = y[c0:c0 + max_samples], f[c0:c0 + max_samples - 1]
    nfft = int(2 ** np.ceil(np.log2(4 * len(f))))
    w_max = max(1, int(round(0.6 / max(bw, 1e-3))))
    scales = [W for W in (1, 2, 4, 8) if W <= w_max and 4 * W < len(f)] or [1]
    total = np.zeros(nfft // 2 + 1)
    for W in scales:
        fw = uniform_filter1d(f, W)
        total += _line_spectrum(np.abs(fw[2 * W:] - fw[:-2 * W]), nfft)
        total += _line_spectrum((fw - fw.mean()) ** 2, nfft)
    total /= len(scales)
    freqs = np.arange(nfft // 2 + 1) / nfft
    band = (freqs >= 1.0 / sps_max) & (freqs <= 1.0 / sps_min)
    idx = np.flatnonzero(band)
    cands = []
    for i in idx[np.argsort(total[idx])[::-1]]:
        a = float(freqs[i])
        if all(abs(a - c["rate"]) > 0.03 * a for c in cands):
            cands.append(dict(rate=a, sps=1 / a, score=float(total[i]), scales=scales))
        if len(cands) == 8:
            break
    pair = power_line_pair(y)
    if pair is not None and 1.0 / sps_max <= pair[0] <= 1.0 / sps_min:
        r, P, f1, f2, dom = pair
        cands = [c for c in cands if abs(c["rate"] - r) > 0.01 * r]
        cands.insert(0, dict(rate=r, sps=1 / r, score=float("inf"),
                             source=f"x^{P} line pair at {f1:+.4f}/{f2:+.4f} cyc/sample (dominance {dom:.1f})"))
    best = cands[0]

    # Harmonic relatives of the top lines are interleaved right after them so that
    # backtracking reaches them early (weak-cyclostationarity cases, e.g. GFSK BT 0.3).
    ordered = []
    for c in cands[:4]:
        ordered.append(c)
        for mult, tag in ((2.0, "x2"), (0.5, "/2")):
            a = c["rate"] * mult
            if 1.0 / sps_max <= a <= 1.0 / sps_min:
                ordered.append(dict(rate=a, sps=1 / a, score=0.0, source=f"{tag} of line at {c['sps']:.2f} sps"))
    ordered += cands[4:]
    cands = []
    for c in ordered:
        if all(abs(c["rate"] - d["rate"]) > 0.01 * c["rate"] for d in cands):
            cands.append(c)
    if str(best.get("source", "")).startswith("x^"):
        # A power-law line spacing is already a precise rate measurement (resolution
        # 1/(8N)); the rect-model refinement is biased for Gaussian pulses at low SNR,
        # so only the boundary phase is refined.
        m, sps, ph = dd_refine(y, cumulative_phase(f), best["sps"], span=0.0, steps=5, continuous=continuous)
    else:
        m, sps, ph = dd_refine(y, cumulative_phase(f), best["sps"], continuous=continuous)
    return RateEstimate(sps, ph + c0, m, cands)


def dd_refine(y: np.ndarray, Phi: np.ndarray, sps: float, span: float = 0.008, steps: int = 33,
              continuous: bool = True):
    """
    Local decision-directed refinement of (sps, boundary phase), coarse to fine:
    a coarse sps grid with a 1-sample phase grid, then a fine sps grid with a
    half-sample phase grid around the best point. Returns (match, sps, phase).
    """
    def scan(s_grid, ph_step):
        best = (-np.inf, sps, 0.0)
        for s in s_grid:
            nph = max(8, int(np.ceil(s / ph_step)))
            for ph in np.arange(nph) / nph * s:
                m, _, _ = dd_match(y, Phi, s, ph, continuous=continuous)
                if m > best[0]:
                    best = (m, float(s), float(ph))
        return best

    coarse = scan(sps * (1 + np.linspace(-span, span, max(5, steps // 4))), 1.0)
    step = 2 * span * sps / max(4, steps // 4 - 1)
    fine = scan(coarse[1] + np.linspace(-step, step, 9), 0.5)
    return max(coarse, fine)


def boundaries_in(tau: float, sps: float, start: float, stop: float) -> np.ndarray:
    """Symbol start times tau + k*sps whose symbol centre lies in [start, stop)."""
    k0 = int(np.ceil((start - tau) / sps - 0.5))
    k1 = int(np.ceil((stop - tau) / sps - 0.5)) - 1
    return tau + np.arange(k0, k1 + 1) * sps


# ---------------------------------------------------------------------------
# Symbol-level frequency strobes and M-level fit
# ---------------------------------------------------------------------------

def cumulative_phase(f: np.ndarray) -> np.ndarray:
    """Phi[n] = accumulated phase (cycles) at sample n, Phi[0] = 0."""
    return np.concatenate([[0.0], np.cumsum(f)])


def if_strobes_median(f: np.ndarray, starts: np.ndarray, sps: float, frac: float = 0.6) -> np.ndarray:
    """
    Robust per-symbol frequency (cycles/symbol): median of the discriminator output
    over the central `frac` of each symbol. Immune to the IF spikes that phase
    jumps (discontinuous FSK) or noise clicks leave after channel filtering.
    """
    # Fixed window length (vectorised): N samples centred in each symbol, indices clipped.
    N = max(1, int(np.floor(frac * sps)))
    i1 = np.round(np.asarray(starts) + (sps - N) / 2 - 0.5).astype(np.int64)
    idx = np.clip(i1[:, None] + np.arange(N)[None, :], 0, len(f) - 1)
    return np.median(f[idx], axis=1) * sps


def if_strobes(Phi: np.ndarray, starts: np.ndarray, sps: float, frac: float = 0.6) -> np.ndarray:
    """Mean frequency (cycles/symbol) over the central `frac` of every symbol."""
    n = np.arange(len(Phi))
    t1 = starts + (1 - frac) / 2 * sps
    t2 = starts + (1 + frac) / 2 * sps
    return (np.interp(t2, n, Phi) - np.interp(t1, n, Phi)) / (frac * sps) * sps


@dataclass
class LevelFit:
    order: int
    center: float            # cycles/symbol (residual CFO)
    spacing: float           # adjacent-level spacing, cycles/symbol (= h for rect pulses)
    sigma: float             # RMS residual, cycles/symbol
    loglik: float            # per-symbol Gaussian-mixture log-likelihood (equal priors)
    positions: np.ndarray    # nearest level index 0..M-1 per symbol
    occupancy: float

    def levels(self) -> np.ndarray:
        return self.center + 0.5 * self.spacing * (2 * np.arange(self.order) - self.order + 1)


def fit_levels(v: np.ndarray, order: int, iters: int = 20) -> LevelFit:
    """
    Equally spaced M-level fit. Several initial spans are tried (full range, 1-99 %,
    5-95 %): with short, biased payloads an extreme tone may occur only once and
    a percentile-only start converges to a compressed grid.
    """
    fits = [_fit_levels_from(v, order, iters, lo, hi)
            for lo, hi in (np.percentile(v, [0, 100]), np.percentile(v, [1, 99]), np.percentile(v, [5, 95]))]
    return max(fits, key=lambda q: q.loglik)


def _fit_levels_from(v: np.ndarray, order: int, iters: int, lo: float, hi: float) -> LevelFit:
    c, d = (lo + hi) / 2, max((hi - lo) / (order - 1), 1e-6)
    for _ in range(iters):
        pos = np.clip(np.round((v - c) / d + (order - 1) / 2), 0, order - 1).astype(int)
        a = 2 * pos - order + 1
        A = np.column_stack([np.ones_like(v), 0.5 * a])
        (c_new, d_new), *_ = np.linalg.lstsq(A, v, rcond=None)
        if d_new <= 0:
            break
        if abs(c_new - c) < 1e-9 and abs(d_new - d) < 1e-9:
            break
        c, d = c_new, d_new
    pos = np.clip(np.round((v - c) / d + (order - 1) / 2), 0, order - 1).astype(int)
    mu = c + 0.5 * d * (2 * np.arange(order) - order + 1)
    sigma = float(np.sqrt(np.mean((v - mu[pos]) ** 2))) + 1e-4
    z = -((v[:, None] - mu[None, :]) ** 2) / (2 * sigma ** 2)
    zmax = z.max(axis=1, keepdims=True)
    ll = zmax[:, 0] + np.log(np.mean(np.exp(z - zmax), axis=1)) - np.log(np.sqrt(2 * np.pi) * sigma)
    counts = np.bincount(pos, minlength=order)
    return LevelFit(order, float(c), float(d), sigma, float(np.mean(ll)), pos,
                    float(np.mean(counts >= max(1, 0.1 * len(v) / order))))


# ---------------------------------------------------------------------------
# Phase continuity
# ---------------------------------------------------------------------------

@dataclass
class Continuity:
    continuous: bool
    boundary_excess: float    # mean |IF| excess of the peak symbol-phase bin (cycles/sample)
    std_error: float


def phase_continuity(f_raw: np.ndarray, starts: np.ndarray, sps: float, max_dev: float) -> Continuity:
    """
    Phase-continuity test from the UNFILTERED discriminator output f_raw (a
    channel filter would smear jumps).
    A phase jump at a symbol boundary adds |jump|/2pi (on average ~0.25
    cycles/sample for random jumps) to the IF sample that straddles it, while
    CPFSK transitions stay inside the tone range. The mean |IF| is computed per
    symbol-phase bin (1 sample wide): noise raises every bin equally, jumps pile
    up in one bin. Discontinuous if the peak bin exceeds the median bin by more
    than 0.05 cycles/sample and by 6 standard errors. Needs no prior timing.
    Returns rates as (peak-bin excess, its standard error) for the evidence.
    """
    nbins = max(4, int(np.ceil(sps)))
    pos = (np.arange(len(f_raw)) + 0.5 - starts[0]) % sps / sps
    b = np.minimum((pos * nbins).astype(int), nbins - 1)
    a = np.abs(f_raw)
    cnt = np.bincount(b, minlength=nbins)
    mean = np.bincount(b, a, minlength=nbins) / np.maximum(cnt, 1)
    var = np.bincount(b, a * a, minlength=nbins) / np.maximum(cnt, 1) - mean ** 2
    k = int(np.argmax(mean))
    excess = float(mean[k] - np.median(mean))
    se = float(np.sqrt(max(var[k], 1e-12) / max(cnt[k], 1)))
    continuous = not (excess > 0.05 and excess > 6 * se)
    return Continuity(continuous, excess, se)


# ---------------------------------------------------------------------------
# Re-modulation match (pulse shape / BT / h hypotheses)
# ---------------------------------------------------------------------------

MATCH_WINDOW = 24   # samples; fixed so that scores of different rate hypotheses are comparable


def model_match(r: np.ndarray, t_sym: np.ndarray, levels: np.ndarray, h: float, cfo: float, pulse: str,
                bt: Optional[float], window: int = MATCH_WINDOW) -> float:
    """
    Mean normalised non-coherent correlation |<r, s>|^2 / (|r|^2 |s|^2) over
    consecutive windows of `window` samples between the received samples and a
    CPM model s built from `levels` (t_sym: symbol-unit time of every sample).

    Short windows make the test immune to CPM phase memory and isolated decision
    errors. The window length is FIXED IN SAMPLES: a normalised correlation over
    N samples has a small-sample (overfitting) floor ~ 1/N, so per-symbol windows
    would favour hypotheses with few samples per symbol.
    """
    s = np.exp(1j * (cpm_phase(levels, t_sym, h, pulse, bt) + 2 * np.pi * cfo * t_sym))
    n = (len(r) // window) * window
    if n == 0:
        return -np.inf
    prod = (r[:n] * np.conj(s[:n])).reshape(-1, window)
    er = (np.abs(r[:n]) ** 2).reshape(-1, window).sum(axis=1)
    return float(np.mean(np.abs(prod.sum(axis=1)) ** 2 / (window * er + 1e-20)))


def model_match_symbols(r: np.ndarray, t_sym: np.ndarray, levels: np.ndarray, h: float, cfo: float,
                        frac: float = 0.8) -> float:
    """
    Per-symbol version of model_match for phase-DISCONTINUOUS FSK: each symbol's
    central `frac` is correlated non-coherently with its own tone, so arbitrary
    phase jumps at the boundaries do not matter. Only comparable between
    hypotheses that share the same symbol rate.
    """
    k = np.floor(t_sym).astype(np.int64)
    u = t_sym - k
    sel = (u >= (1 - frac) / 2) & (u < (1 + frac) / 2) & (k >= 0) & (k < len(levels))
    k, tt = k[sel], t_sym[sel]
    s = np.exp(2j * np.pi * (cfo + 0.5 * h * levels[k]) * tt)
    prod = r[sel] * np.conj(s)
    num = np.abs(np.bincount(k, prod.real) + 1j * np.bincount(k, prod.imag)) ** 2
    er = np.bincount(k, np.abs(r[sel]) ** 2)
    cnt = np.bincount(k)
    ok = cnt > 0
    return float(np.mean(num[ok] / (cnt[ok] * er[ok] + 1e-20)))


def track_rate(y: np.ndarray, Phi: np.ndarray, sps: float, boundary: float, continuous: bool = True,
               seg_symbols: int = 250, phases: int = 16, max_slope: float = 0.003):
    """
    Whole-burst timing tracking (feedforward): the boundary phase is estimated
    independently in consecutive segments of `seg_symbols` symbols (decision-directed
    match, sps fixed), unwrapped against the prediction boundary + k*sps, and a line
    is fitted to the offsets. Its slope corrects the rate: over a 2500-symbol frame a
    0.05 % rate error is 1.3 symbols of drift, far beyond what a local refinement on a
    few hundred symbols can resolve. The correction is rejected (n_segments < 0) if
    the offsets do not follow a line (RMS residual > sps/8) or |slope| > max_slope.
    Returns (sps, boundary, n_segments).
    """
    seg = int(seg_symbols * sps)
    if len(y) < 2 * seg:
        return sps, boundary, 1
    centres, offs = [], []
    for a in range(0, len(y) - seg + 1, seg):
        ys, Ps = y[a:a + seg], Phi[a:a + seg + 1] - Phi[a]
        best = (-np.inf, 0.0)
        for ph in np.arange(phases) / phases * sps:
            m, _, _ = dd_match(ys, Ps, sps, ph, continuous=continuous)
            if m > best[0]:
                best = (m, ph)
        pred = (boundary - a) % sps
        d = (best[1] - pred + sps / 2) % sps - sps / 2
        centres.append(a + seg / 2)
        offs.append(d)
    centres, offs = np.array(centres), np.unwrap(np.array(offs) * 2 * np.pi / sps) * sps / (2 * np.pi)
    slope, icpt = np.polyfit(centres, offs, 1)
    resid = offs - (slope * centres + icpt)
    # Accept only a consistent, plausible drift: noisy segment phases (low SNR, heavy
    # ISI) must not "correct" a good estimate into a bad one.
    if abs(slope) > max_slope or np.sqrt(np.mean(resid ** 2)) > sps / 8:
        return sps, boundary, -len(offs)
    # boundaries sit at boundary + icpt + k*sps*(1+slope) (to first order)
    return float(sps * (1 + slope)), float(boundary + icpt), len(offs)
