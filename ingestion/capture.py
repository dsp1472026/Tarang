"""
Unified capture loader: .iq (interleaved or planar) and .wav (stereo IQ or
real passband) -> complex baseband + sample rate + a record of what was done.

    .iq  interleaved   I,Q,I,Q,...            x = I + jQ   (CU8, CS8, CS16, CF32, CF64;
                                                            see ingestion/iq_formats.py)
    .iq  planar        I,I,...,Q,Q,...        de-interleaved (only on request)
    .wav stereo        L = I, R = Q           x = L + jR (unless L == R: then mono)
    .wav mono (real)   passband audio         analytic signal (FFT Hilbert),
                                              carrier estimated from the PSD,
                                              mixed to 0 Hz, low-pass, decimated

A real passband signal s(t) = I cos(wt) - Q sin(wt) has the analytic signal
(I + jQ) exp(jwt), so after mixing down the constellation keeps its
orientation (no spectral inversion; Gray labels and sync words stay valid).
The carrier is only estimated coarsely here (occupied-band centre); the
receivers remove the residual offset themselves, exactly as for IQ files.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly

from synthetic.iq_io import read_iq

# The decimated rate is chosen as >= OVERSAMPLE x the occupied bandwidth.
OVERSAMPLE = 6.0


@dataclass
class Capture:
    samples: np.ndarray              # complex baseband
    sample_rate: Optional[float]     # of `samples` (after any decimation); None = unknown
    source: str                      # iq-cu8 | iq-cs8 | iq-cs16 | iq-cf32 | iq-cf64 | wav-stereo-iq | wav-real-passband
    path: str
    carrier_hz: Optional[float] = None       # estimated passband carrier (real WAV only)
    decimation: int = 1
    file_sample_rate: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    @property
    def sample_rate_source(self) -> str:
        return "KNOWN" if self.sample_rate else "UNKNOWN"


def load_capture(path: str, dtype: Optional[str] = None, sample_rate: Optional[float] = None,
                 layout: str = "interleaved", wav_mode: str = "auto", byte_offset: int = 0,
                 endianness: Optional[str] = None, iq_order: str = "IQ") -> Capture:
    """
    dtype/sample_rate: .iq overrides (else sidecar / '_f32' naming). dtype accepts CU8, CS8/CI8,
      CS16/CI16/SC16, CF32/FC32, CF64/FC64 and the numpy-style names (uint8, int16, complex64...).
    layout: 'interleaved' | 'planar' (.iq only). byte_offset: header bytes to skip (.iq only).
    endianness: 'little' | 'big' (.iq multi-byte types; default from the dtype suffix, else little).
    iq_order: 'IQ' | 'QI' (.iq only: the file stores Q first).
    wav_mode: 'auto' | 'iq' (stereo L=I, R=Q) | 'real' (passband, first channel).
    """
    if path.lower().endswith(".wav"):
        return _load_wav(path, wav_mode, sample_rate)
    f = read_iq(path, datatype=dtype, sample_rate=sample_rate, byte_offset=byte_offset, endianness=endianness)
    x = f.samples
    notes = [f"{f.datatype.upper()} interleaved I/Q" + (f", {byte_offset} header bytes skipped" if byte_offset else "")]
    if layout == "planar":
        raw = np.column_stack((x.real, x.imag)).ravel()          # undo the interleaved split
        h = len(raw) // 2
        x = (raw[:h] + 1j * raw[h:2 * h]).astype(np.complex64)
        notes = [f"{f.datatype.upper()} planar I-block then Q-block"]
    if str(iq_order).upper() == "QI":
        x = (x.imag + 1j * x.real).astype(np.complex64)
        notes.append("I/Q order swapped (file stores Q first)")
    return Capture(x, f.sample_rate, "iq-" + f.datatype.split("_")[0], path, notes=notes,
                   file_sample_rate=f.sample_rate)


MIN_BAND_SNR_DB = 3.0        # below this the occupied band is noise and says nothing about the rate
MIN_DECIMATED_SAMPLES = 4096


def reduce_oversampling(cap: Capture) -> Capture:
    """
    Decimate a heavily oversampled capture to >= OVERSAMPLE x its occupied band (offset included).
    At ~40 samples/symbol (1200 Bd in a 48 kHz WAV / IQ file) the FSK receiver's rate estimate
    locked onto twice the symbol rate and no frame was found; at 8-20 samples/symbol the same
    capture decoded, and the classifier and receivers run several times faster. Captures already
    near OVERSAMPLE x their band (e.g. 4 samples/symbol) are returned unchanged.
    """
    from estimation.fsk_estimator import occupied_band

    x = np.asarray(cap.samples, dtype=np.complex128)
    if len(x) < 2 * MIN_DECIMATED_SAMPLES:
        return cap
    band = occupied_band(x)
    if band.snr_db < MIN_BAND_SNR_DB:
        return cap
    span = band.bw99 + 2.0 * abs(band.center)            # keep a residual carrier offset inside the band
    dec = int(np.floor(1.0 / (OVERSAMPLE * max(span, 1e-6))))
    dec = min(dec, len(x) // MIN_DECIMATED_SAMPLES)
    if dec < 2:
        return cap
    y = resample_poly(x, 1, dec)
    fs = cap.sample_rate / dec if cap.sample_rate else None
    note = (f"oversampled: occupied band {band.bw99 * (cap.sample_rate or 1):.4g} "
            f"{'Hz' if cap.sample_rate else 'cycles/sample'} -> decimated x{dec}"
            + (f" to {fs:g} Hz" if fs else ""))
    return Capture(y.astype(np.complex64), fs, cap.source, cap.path, cap.carrier_hz, cap.decimation * dec,
                   cap.file_sample_rate, cap.notes + [note])


def _wav_is_iq_by_metadata(path: str) -> Optional[bool]:
    """Sidecar hint: True = IQ, False = real audio, None = no information."""
    import json
    stem = os.path.splitext(path)[0]
    for sc in (stem + ".sigmf-meta", stem + ".json", stem + "_metadata.json", stem + "_ground_truth.json"):
        if not os.path.exists(sc):
            continue
        try:
            with open(sc) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        text = json.dumps(meta).lower()
        if '"wav_channels": 1' in text or "passband" in text:
            return False
        if "sigmf" in sc or '"iq"' in text or "wav_iq" in text:
            return True
    return None


def _load_wav(path: str, wav_mode: str, fs_override: Optional[float]) -> Capture:
    fs, data = wavfile.read(path)
    fs = float(fs_override or fs)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)
    notes = [f"WAV {data.shape[1] if data.ndim > 1 else 1} ch @ {fs:g} Hz"]

    if data.ndim > 1 and data.shape[1] >= 2:
        left, right = data[:, 0], data[:, 1]
        same = np.allclose(left, right, atol=1e-4 * (np.max(np.abs(left)) + 1e-12))
        meta_iq = _wav_is_iq_by_metadata(path)
        if wav_mode == "iq" or (wav_mode == "auto" and not same and meta_iq is not False):
            notes.append("stereo -> L = I, R = Q" + ("" if meta_iq else " (no metadata: assumed IQ; "
                                                     "use wav_mode='real' if it is two-channel audio)"))
            return Capture((left + 1j * right).astype(np.complex64), fs, "wav-stereo-iq", path,
                           file_sample_rate=fs, notes=notes)
        notes.append("stereo channels identical -> treated as mono real" if same else
                     "stereo treated as real audio (first channel)")
        data = left
    elif wav_mode == "iq":
        raise ValueError("wav_mode='iq' needs a stereo WAV")

    return _real_passband_to_baseband(data, fs, path, notes)


def _real_passband_to_baseband(s: np.ndarray, fs: float, path: str, notes: List[str]) -> Capture:
    from estimation.fsk_estimator import occupied_band

    s = s - np.mean(s)
    z = hilbert(s)                         # FFT Hilbert: X(f<0)=0, X(f>0)*2, DC/Nyquist *1
    # The analytic signal's negative half is empty; occupied_band takes its noise floor from the
    # quietest bins, so on z itself it read that empty half as the floor and counted all the white
    # noise of the positive half as signal (a 2.4 kHz FSK signal at 9 kHz came out as 22 kHz wide at
    # 12 kHz). The positive half [0, fs/2) is therefore first shifted to [-fs/4, fs/4) and decimated
    # by 2, which leaves no empty region, and the band is measured there.
    n = np.arange(len(z))
    h = resample_poly(z * np.exp(-2j * np.pi * 0.25 * n), 1, 2)
    fs_h = fs / 2.0
    band = occupied_band(h)
    fc = 0.25 + band.center / 2.0          # carrier in cycles/sample of the file rate
    m = np.arange(len(h))
    y = h * np.exp(-2j * np.pi * band.center * m)
    # No channel filter here: resample_poly's anti-alias low-pass (at the new Nyquist,
    # >= 3 x the occupied bandwidth) is enough, and the out-of-band noise it keeps is
    # what the receivers use to measure the noise floor / SNR.
    d2 = int(max(1, np.floor(1.0 / (OVERSAMPLE * max(band.bw99, 1e-6)))))
    if d2 > 1:
        y = resample_poly(y, 1, d2)
    dec = 2 * d2
    notes.append(f"real passband: analytic signal (FFT Hilbert), carrier ~{fc * fs:.1f} Hz "
                 f"(occupied-band centre), 97% bandwidth {band.bw99 * fs_h:.0f} Hz, mixed to 0 Hz, "
                 f"decimated x{dec} -> {fs / dec:g} Hz")
    return Capture(y.astype(np.complex64), fs / dec, "wav-real-passband", path, carrier_hz=float(fc * fs),
                   decimation=dec, file_sample_rate=fs, notes=notes)
