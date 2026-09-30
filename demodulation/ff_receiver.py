"""
Feed-forward linear receiver (PSK / QAM) for long, unscrambled bursts.

The loop receivers (Gardner timing + Costas / decision-directed carrier) need symbol transitions.
An image sent without a scrambler contains long runs of one byte value (a white slide background
= 0xFF...), i.e. thousands of identical symbols: an unmodulated carrier with no timing
information. The loops then drift and slip (a 2 M-sample QPSK screenshot lost lock ~21 500
symbols in: one 90-degree carrier slip plus whole-symbol timing slips, turning the rest of the
image into colour noise). This receiver has no loops:

  1. coarse centring on the occupied band, fine CFO from the M-th power line (x^2 BPSK, x^4 QPSK /
     QAM, x^8 8PSK, x^16 16PSK), parabolic interpolation;
  2. root-raised-cosine matched filter;
  3. timing: Oerder & Meyr square-law estimate per block of symbols; blocks without transitions
     (small spectral line) carry no weight and are bridged by interpolation between their
     neighbours; the unwrapped estimates are smoothed; cubic (Lagrange) interpolation at the
     estimated instants;
  4. carrier phase: Viterbi & Viterbi, M-th power averaged over a sliding window of symbols,
     unwrapped with the M-fold ambiguity kept continuous (identical symbols are no problem:
     their M-th power is constant);
  5. gain: decision-directed scale against the reference constellation.

The residual M-fold phase ambiguity is resolved downstream (sync word / file-frame search), as
for the loop receivers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import fftconvolve

from core.evidence import Evidence, Parameter
from dsp.constellations import get_constellation

TIMING_BLOCK = 256            # symbols per Oerder & Meyr block
CARRIER_WINDOW = 128          # symbols in the Viterbi & Viterbi average
DD_WINDOW = 256               # symbols in the decision-directed phase refinement
MAX_FFT = 1 << 22


def _power_order(mod: str) -> int:
    return {"BPSK": 2, "QPSK": 4, "4QAM": 4, "8PSK": 8, "16PSK": 16}.get(mod, 4)


def rrc(beta: float, sps: float, span: int = 10) -> np.ndarray:
    n = int(np.ceil(span * sps / 2))
    t = np.arange(-n, n + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                                          (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / \
                   (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def _fine_line(z: np.ndarray) -> float:
    """Frequency (cycles/sample) of the strongest line of z, parabolic interpolation."""
    n = min(len(z), MAX_FFT)
    nfft = 1 << int(np.ceil(np.log2(n)))
    Z = np.abs(np.fft.fft(z[:n] * np.hanning(n), nfft)) ** 2
    k = int(np.argmax(Z))
    a, b, c = Z[(k - 1) % nfft], Z[k], Z[(k + 1) % nfft]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den != 0 else 0.0
    f = (k + d) / nfft
    return f - 1.0 if f > 0.5 else f


def _lagrange(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cubic Lagrange interpolation of y at fractional sample positions t."""
    i = np.floor(t).astype(np.int64)
    mu = t - i
    i = np.clip(i, 1, len(y) - 3)
    ym1, y0, y1, y2 = y[i - 1], y[i], y[i + 1], y[i + 2]
    return (-mu * (mu - 1) * (mu - 2) / 6 * ym1 + (mu + 1) * (mu - 1) * (mu - 2) / 2 * y0
            - (mu + 1) * mu * (mu - 2) / 2 * y1 + (mu + 1) * mu * (mu - 1) / 6 * y2)


def _running_mean(z: np.ndarray, w: int) -> np.ndarray:
    w = int(max(1, min(w, len(z))))
    c = np.concatenate([[0], np.cumsum(z)])
    lo = np.clip(np.arange(len(z)) - w // 2, 0, len(z))
    hi = np.clip(lo + w, 0, len(z))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


@dataclass
class FFResult:
    modulation: str
    symbols: np.ndarray
    llrs: np.ndarray
    hard_bits: np.ndarray
    evm: float
    esn0_est_db: float
    parameters: Dict[str, Parameter]
    evidence: List[Evidence] = field(default_factory=list)
    # same surface as the loop receivers' results (the worker reads these)
    payload_text: str = ""
    decode: object = None

    def __post_init__(self):
        if self.decode is None:
            from types import SimpleNamespace
            self.decode = SimpleNamespace(frame=None, method="feed-forward (no text frame)", payload=b"",
                                          sync_bit_errors=None)


def receive_feedforward(x: np.ndarray, fs: Optional[float], modulation: str, sps: float,
                        beta: float = 0.35, center: Optional[float] = None) -> FFResult:
    """Loop-free receiver; `center` = coarse carrier offset in cycles/sample (None: occupied band)."""
    from estimation.fsk_estimator import occupied_band
    mod = "QPSK" if modulation == "4QAM" else modulation
    const = get_constellation(mod)
    M = _power_order(mod) if mod.endswith("PSK") else 4
    x = np.asarray(x, dtype=np.complex128)
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-30)
    n = np.arange(len(x))
    # 1. carrier offset: coarse (band centre) + fine (M-th power line)
    f0 = occupied_band(x[:MAX_FFT]).center if center is None else float(center)
    y = x * np.exp(-2j * np.pi * f0 * n)
    df = _fine_line(y ** M) / M
    y = y * np.exp(-2j * np.pi * df * n)
    cfo = f0 + df
    # 2. matched filter
    y = fftconvolve(y, rrc(beta, sps), mode="same")
    # 3. Oerder & Meyr timing per block, low-transition blocks bridged
    nsym = int((len(y) - 4 * sps) // sps)
    blk = int(TIMING_BLOCK * sps)
    nb = max(1, len(y) // blk)
    p = np.abs(y[: nb * blk]) ** 2
    # global sample index: with a non-integer sps a per-block index restarts the symbol grid at every
    # block and the estimates jump from block to block (drift then goes untracked)
    e = np.exp(-2j * np.pi * np.arange(nb * blk) / sps)
    c = (p * e).reshape(nb, blk).sum(axis=1)
    w = np.abs(c)
    good = w > 0.2 * np.median(w[w > 0]) if np.any(w > 0) else np.ones(nb, bool)
    if good.sum() < 2:
        good[:] = True
    ph = np.unwrap(np.angle(c[good]))
    centres = (np.arange(nb) + 0.5) * blk
    tau_good = -ph / (2 * np.pi) * sps                        # timing offset (samples) at the good blocks
    # a sample-rate / symbol-rate mismatch is a linear drift: weighted straight-line fit (extends to the
    # burst edges, where holding the first block's value mis-sampled the header), plus the smoothed
    # residual of the good blocks, bridged across the blocks without transitions
    xg, wg = centres[good], w[good]
    if len(xg) >= 2:
        A = np.vstack([np.ones_like(xg), xg]).T * wg[:, None]
        coef = np.linalg.lstsq(A, tau_good * wg, rcond=None)[0]
    else:
        coef = np.array([tau_good[0], 0.0])
    resid = _running_mean(tau_good - (coef[0] + coef[1] * xg), 5)
    ts = np.arange(nsym) * sps
    tau = coef[0] + coef[1] * ts + np.interp(ts, xg, resid)
    t = np.arange(nsym) * sps + tau
    t = t[(t > 2) & (t < len(y) - 3)]
    s = _lagrange(y, t)
    # 4. Viterbi & Viterbi carrier phase (M-th power, sliding window), unwrapped continuously
    ref = np.angle(np.mean(const.points ** M)) / M            # where the reference set's M-th power points
    if mod.endswith("PSK"):
        z = s / (np.abs(s) + 1e-12)
        zm = _running_mean(z ** M, CARRIER_WINDOW)
        g = np.sqrt(np.mean(np.abs(const.points) ** 2) / (np.mean(np.abs(s) ** 2) + 1e-30))
    else:
        # QAM. Image data is far from equiprobable (a white background is one corner symbol), so
        # (a) the scale comes from matching |s| to the constellation's ring radii, not from the mean
        #     power (which shrank the grid and put corners on inner points), and
        # (b) the 4th-power phase uses the outer-corner ring only: the middle-ring points' 4th power
        #     is not at the corners' angle, and with the data mix changing along the image that bias
        #     rotated whole stretches (16QAM screenshot: clean rings, smeared phase, EVM 20 %).
        radii = np.unique(np.round(np.abs(const.points), 6))
        mags = np.abs(s[:: max(1, len(s) // 50000)])
        best = None
        for gg in np.geomspace(0.25, 4.0, 241):
            m = mags * gg
            err = float(np.mean(np.min(np.abs(m[:, None] - radii[None, :]), axis=1) ** 2))
            if best is None or err < best[0]:
                best = (err, gg)
        g = best[1]
        corner = np.abs(s) * g > 0.5 * (radii[-1] + radii[-2])
        wsum = _running_mean(corner.astype(float), 4 * CARRIER_WINDOW)
        zm = _running_mean(np.where(corner, s ** 4, 0), 4 * CARRIER_WINDOW) / np.maximum(wsum, 1e-9)
        have = wsum > 0.5 / (4 * CARRIER_WINDOW)
        if np.any(have) and not np.all(have):            # windows without corner symbols: bridge
            idx = np.arange(len(zm))
            zm = (np.interp(idx, idx[have], zm[have].real) + 1j * np.interp(idx, idx[have], zm[have].imag))
    phi = np.unwrap(np.angle(zm)) / M
    s = s * np.exp(-1j * (phi - ref)) * g
    # 5. gain + phase: decision-directed passes
    for _ in range(3):
        dec = const.points[const.nearest_labels(s)]
        g2 = np.real(np.vdot(s, dec)) / (np.real(np.vdot(s, s)) + 1e-30)
        s = s * g2
        # decision-directed phase refinement: the M-th power estimate is noisy for dense QAM (self-noise
        # of the inner / middle rings); decisions give a clean phase error, smoothed over DD_WINDOW
        err = _running_mean(s * np.conj(dec), DD_WINDOW)
        s = s * np.exp(-1j * np.angle(err))
    dec = const.points[const.nearest_labels(s)]
    evm = float(np.sqrt(np.mean(np.abs(s - dec) ** 2) / np.mean(np.abs(dec) ** 2)))
    esn0 = float(-20 * np.log10(max(evm, 1e-4)))
    llrs = const.demap_soft(s, 10 ** (-esn0 / 10))
    hard = (llrs < 0).astype(np.uint8)
    fsr = fs or 1.0
    unit = "Hz" if fs else "cycles/sample"
    params = {
        "sps": Parameter("Samples/symbol", float(sps), "samples", "ESTIMATED", "feed-forward receiver input"),
        "Symbol rate": Parameter("Symbol rate", fsr / sps, "Bd" if fs else "symbols/sample", "ESTIMATED",
                                 "sample rate / samples per symbol"),
        "cfo": Parameter("Carrier offset", cfo * fsr, unit, "ESTIMATED", f"band centre + x^{M} line"),
        "evm": Parameter("EVM", 100 * evm, "%", "ESTIMATED", "decision-directed"),
    }
    ev = [Evidence(f"Feed-forward receiver ({mod}): x^{M} carrier {cfo * fsr:+.2f} {unit}, Oerder-Meyr timing over "
                   f"{nb} blocks ({int(good.sum())} with transitions), Viterbi-Viterbi phase, EVM {100 * evm:.2f} %",
                   "FF_RECEIVER", float(max(0.0, 1 - evm)))]
    return FFResult(mod, s.astype(np.complex64), llrs, hard, evm, esn0, params, ev)
