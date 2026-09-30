"""
Multi-signal captures: find every signal in the spectrum and extract each one as its own capture.

A capture can hold several transmissions on different carriers (e.g. three AM stations at
-13 kHz, -1 kHz and +11 kHz in a 48 kHz I/Q file). The receivers analyse one signal at a time,
so the occupied bands are located on the averaged spectrum and each is mixed to 0 Hz, low-pass
filtered to its band and decimated to about OVERSAMPLE x its bandwidth.

Band finding: Welch PSD (median-smoothed), noise floor = median bin level (valid while signals
occupy less than half of the band), bins more than THRESH_DB above the floor are occupied, gaps
narrower than MERGE_GAP of the sample rate or than half the wider neighbour are bridged (one
signal's side islands, e.g. 2FSK tone lines), and bands narrower than MIN_BINS bins or
carrying less than MIN_POWER_FRACTION of the in-band power are dropped. A capture is split only
when at least two bands survive; otherwise it is analysed whole, exactly as before.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.signal import firwin, medfilt, oaconvolve, resample_poly, welch

from ingestion.capture import OVERSAMPLE, Capture

THRESH_DB = 10.0              # occupied: smoothed PSD this far above the noise floor
MERGE_GAP = 0.01              # bridge gaps narrower than this fraction of the sample rate
MIN_BINS = 8                  # a band must span at least this many PSD bins
MIN_POWER_FRACTION = 0.01     # and carry this share of the total occupied power
MAX_SIGNALS = 16
EDGE_MARGIN = 0.15            # extraction filter: band edges widened by this fraction of the bandwidth


@dataclass
class SignalBand:
    index: int
    center_hz: float          # relative to the capture centre (Hz, or cycles/sample x fs=1 when unknown)
    bandwidth_hz: float
    peak_db: float            # smoothed PSD peak above the noise floor
    snr_db: float             # in-band power over the noise in the same band
    power_fraction: float     # share of all occupied power

    def label(self, fs_known: bool = True) -> str:
        u = "Hz" if fs_known else "cyc/sample"
        return (f"Signal {self.index + 1}: {self.center_hz:+,.1f} {u}, BW {self.bandwidth_hz:,.1f} {u}, "
                f"SNR {self.snr_db:.1f} dB")


def find_signals(x: np.ndarray, fs: Optional[float]) -> List[SignalBand]:
    """Occupied bands of a capture, strongest first. A single band means 'one signal'."""
    x = np.asarray(x)
    fsr = float(fs) if fs else 1.0
    nper = int(min(8192, 2 ** np.floor(np.log2(max(len(x) // 16, 256)))))
    f, p = welch(x, fs=fsr, nperseg=nper, return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    pdb = 10 * np.log10(p + 1e-30)
    sm = medfilt(pdb, 9)
    floor = float(np.median(sm))
    occ = sm > floor + THRESH_DB
    df = f[1] - f[0]
    gap = max(1, int(round(MERGE_GAP * fsr / df)))
    # bridge short gaps
    idx = np.flatnonzero(occ)
    if not len(idx):
        return []
    runs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    # One signal can show up as several islands (the tone lines of a 2FSK signal next to its main
    # lobe): neighbours whose gap is under half the wider one's width are the same signal.
    # Separate stations (e.g. 8 kHz-wide AM channels 5 kHz apart) stay apart.
    # Repeated until stable: a noisy, lumpy spectrum (unscrambled image data at 14 dB) breaks one
    # signal into many pieces, and a small piece between a skirt island and the main body blocked
    # the single left-to-right pass.
    while True:
        merged = [runs[0]]
        for a, b in runs[1:]:
            pa, pb = merged[-1]
            if a - pb - 1 < 0.5 * max(pb - pa + 1, b - a + 1):
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))
        if len(merged) == len(runs):
            break
        runs = merged
    runs = merged
    lin = 10 ** (sm / 10)
    noise = 10 ** (floor / 10)
    bands = []
    for a, b in runs:
        if b - a + 1 < MIN_BINS:
            continue
        pw = float(np.sum(np.clip(lin[a:b + 1] - noise, 0, None)))
        snr = 10 * np.log10(max(pw, 1e-30) / (noise * (b - a + 1)))
        bands.append([(f[a] + f[b]) / 2, f[b] - f[a] + df, float(sm[a:b + 1].max() - floor), snr, pw])
    total = sum(bd[4] for bd in bands) or 1.0
    bands = [bd for bd in bands if bd[4] / total >= MIN_POWER_FRACTION]
    bands.sort(key=lambda bd: -bd[4])
    return [SignalBand(i, float(c), float(w), float(pk), float(s), float(pw / total))
            for i, (c, w, pk, s, pw) in enumerate(bands[:MAX_SIGNALS])]


def extract(cap: Capture, band: SignalBand) -> Capture:
    """Mix `band` to 0 Hz, low-pass it and decimate to ~OVERSAMPLE x its bandwidth."""
    x = np.asarray(cap.samples, dtype=np.complex128)
    fsr = float(cap.sample_rate) if cap.sample_rate else 1.0
    fc = band.center_hz / fsr                                  # cycles/sample
    y = x * np.exp(-2j * np.pi * fc * np.arange(len(x)))
    half = 0.5 * band.bandwidth_hz * (1 + 2 * EDGE_MARGIN) / fsr
    half = min(max(half, 4.0 / len(x)), 0.45)
    dec = int(max(1, np.floor(1.0 / (OVERSAMPLE * 2 * half))))
    ntap = int(min(2001, max(101, 8 / (2 * half)))) | 1
    h = firwin(ntap, 2 * half)
    y = oaconvolve(y, h, mode="same")
    if dec > 1:
        y = resample_poly(y, 1, dec, window=("kaiser", 5.0))
    fs2 = cap.sample_rate / dec if cap.sample_rate else None
    carrier = (cap.carrier_hz or 0.0) + band.center_hz if cap.sample_rate else None
    note = (f"channelised {band.label(bool(cap.sample_rate))}: mixed to 0 Hz, low-pass +-{half * fsr:,.1f}, "
            f"decimated x{dec}" + (f" -> {fs2:g} Hz" if fs2 else ""))
    return Capture(y.astype(np.complex64), fs2, cap.source, cap.path, carrier, cap.decimation * dec,
                   cap.file_sample_rate, list(cap.notes) + [note])
