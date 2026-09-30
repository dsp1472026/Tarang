"""
L5 carrier recovery for M-PSK at 1 sample/symbol.

 * M-th power feedforward: raising s to the M-th power strips M-PSK data and
   leaves a tone at M*df with phase M*phi. Its peak gives the fine frequency,
   its phase gives the carrier phase modulo 2*pi/M (Viterbi & Viterbi).
   The same line strength (rho_M) is used by the classifier: it is carrier-,
   data- and rotation-invariant.
 * Second-order PLL for tracking: classic Costas detectors for BPSK and QPSK,
   decision-directed phase detector for 8PSK and above.

The remaining 2*pi/M ambiguity is resolved in L6 (sync word / differential
decoding / known payload).
"""

from dataclasses import dataclass

import numpy as np

from dsp.constellations import Constellation


@dataclass
class PowerLine:
    order: int
    rho: float          # |line| / sum|s|, in [0, 1]; ~1 when s is M-PSK at high SNR
    freq: float         # carrier frequency offset in cycles/symbol (unambiguous within +-1/(2M))
    phase: float        # carrier phase in rad, modulo 2*pi/M, relative to the reference constellation


def mth_power_line(s: np.ndarray, order: int, ref_point: complex = 1.0, max_freq: float = None) -> PowerLine:
    """
    Locate the M-th power spectral line of 1-sps symbols.

    Uses z = |s| * exp(j*M*arg s) (Viterbi & Viterbi with amplitude exponent 1),
    which limits noise enhancement compared with s**M.
    ref_point: any point of the reference constellation; its M-th-power phase is
    removed so a returned phase of 0 means "aligned with the reference".
    """
    z = np.abs(s) * np.exp(1j * order * np.angle(s))
    n = len(z)
    nfft = int(2 ** np.ceil(np.log2(max(16 * n, 1024))))
    spec = np.fft.fft(z, nfft)
    freqs = np.fft.fftfreq(nfft)
    mag = np.abs(spec)
    if max_freq is not None:
        mag = np.where(np.abs(freqs) <= max_freq * order, mag, 0.0)
    k = int(np.argmax(mag))
    # Parabolic refinement of the peak position.
    ym1, y0, yp1 = mag[k - 1], mag[k], mag[(k + 1) % nfft]
    den = ym1 - 2 * y0 + yp1
    delta = 0.0 if den == 0 else float(np.clip(0.5 * (ym1 - yp1) / den, -0.5, 0.5))
    f_line = freqs[k] + delta / nfft
    # Phase at the refined frequency, referenced to the middle of the block.
    nn = np.arange(n) - (n - 1) / 2.0
    c = np.sum(z * np.exp(-2j * np.pi * f_line * nn))
    ref_phase = np.angle(ref_point ** order)
    phase = float(np.angle(c * np.exp(-1j * ref_phase)) / order)
    return PowerLine(order, float(np.abs(c) / (np.sum(np.abs(s)) + 1e-12)), float(f_line / order), phase)


def apply_carrier(s: np.ndarray, freq: float, phase: float) -> np.ndarray:
    """Remove a carrier (cycles/symbol, rad) referenced to the block centre."""
    nn = np.arange(len(s)) - (len(s) - 1) / 2.0
    return s * np.exp(-1j * (2 * np.pi * freq * nn + phase))


@dataclass
class PLLResult:
    symbols: np.ndarray
    phase_track: np.ndarray   # rad, per symbol (applied correction)
    freq_final: float         # rad/symbol, residual frequency the loop settled on
    detector: str


def carrier_pll(s: np.ndarray, const: Constellation, loop_bw: float = 0.01, damping: float = 0.707) -> PLLResult:
    """
    Second-order carrier PLL at 1 sample/symbol.

    BPSK: Costas  e = Re(y) * Im(y)                     (after phi0 alignment)
    QPSK: Costas  e = sgn(Re y) Im(y) - sgn(Im y) Re(y)  (on the 45-deg-rotated grid)
    else: decision-directed  e = Im(y * conj(d)) with d the nearest point.
    Input amplitude is normalised to unit RMS so the loop gain is level-independent.
    """
    order = const.order
    s = s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-12)
    # Rotate so the constellation sits on the axes (BPSK) / diagonals -> axes (QPSK).
    # Detector formulas below assume BPSK points at +-1 and QPSK points at (+-1 +-j)/sqrt2 rotated to axes.
    if order == 2:
        align = -np.angle(const.points[1])      # put label-1 point on +1 (real axis)
        detector = "costas-bpsk"
    elif order == 4:
        align = -np.angle(const.points[0]) + np.pi / 4  # QPSK points on the diagonals
        detector = "costas-qpsk"
    else:
        align = 0.0
        detector = "decision-directed"

    theta = loop_bw / (damping + 1.0 / (4.0 * damping))
    d = 1.0 + 2.0 * damping * theta + theta ** 2
    kp = 4.0 * damping * theta / d
    ki = 4.0 * theta ** 2 / d

    out = np.empty_like(s)
    phases = np.empty(len(s))
    phi, freq = 0.0, 0.0
    rot = np.exp(1j * align)
    for i, x in enumerate(s):
        y = x * np.exp(-1j * phi)
        yr = y * rot
        if order == 2:
            e = yr.real * yr.imag
        elif order == 4:
            e = (np.sign(yr.real) * yr.imag - np.sign(yr.imag) * yr.real) / np.sqrt(2.0)
        else:
            dec = const.points[np.argmin(np.abs(y - const.points))]
            e = (y * np.conj(dec)).imag
        out[i] = y
        phases[i] = phi
        freq += ki * e
        phi += kp * e + freq
    return PLLResult(out, phases, float(freq), detector)
