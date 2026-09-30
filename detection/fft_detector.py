from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from scipy import signal as dsp_signal

from core.signal_buffer import SignalBuffer
from core.evidence import Evidence


MIN_SOI_POWER_FRACTION = 0.005   # an SOI must carry this share of the detected power ...
MAX_SOI_DYNAMIC_DB = 40.0        # ... or peak within this many dB of the strongest SOI


@dataclass
class DetectedSOI:
    """
    Represents a Detected Signal of Interest (SOI).
    """
    soi_id: int
    center_freq_hz: float
    bandwidth_hz: float
    power_db: float
    snr_db: float
    start_sample: int
    end_sample: int
    evidence: List[Evidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soi_id": self.soi_id,
            "center_freq_hz": self.center_freq_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "power_db": float(self.power_db),
            "snr_db": float(self.snr_db),
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "evidence": [ev.to_dict() for ev in self.evidence]
        }


class FFTDetector:
    """
    Welch PSD-based energy signal detector with dynamic noise floor estimation
    (Median + MAD) for detecting Signals of Interest (SOIs).
    """

    @classmethod
    def compute_psd(
        cls,
        buffer: SignalBuffer,
        nperseg: int = 1024,
        noverlap: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes Welch Power Spectral Density (PSD) for complex or real signal buffer.
        Returns (frequencies_hz, psd_linear).
        """
        if not buffer.is_valid():
            raise ValueError("Buffer is invalid or empty.")

        fs = buffer.sample_rate if buffer.sample_rate and buffer.sample_rate > 0 else 1.0
        nperseg = max(16, min(nperseg, len(buffer.samples)))
        if noverlap is None or noverlap >= nperseg:
            noverlap = nperseg // 2
        else:
            noverlap = min(noverlap, max(0, nperseg - 1))

        freqs, psd = dsp_signal.welch(
            buffer.samples,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False,
            scaling="density"
        )

        # Shift frequencies and PSD so center freq is 0 (or centered around buffer.center_frequency)
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)

        freqs += buffer.center_frequency
        return freqs, psd

    @classmethod
    def estimate_noise_floor(cls, psd_db: np.ndarray) -> tuple[float, float]:
        """
        Estimates noise floor (in dB) using Median and Median Absolute Deviation (MAD).
        Returns (noise_floor_db, mad_db).
        """
        median_db = float(np.median(psd_db))
        mad_db = float(np.median(np.abs(psd_db - median_db)))
        return median_db, mad_db

    @classmethod
    def detect_sois(
        cls,
        buffer: SignalBuffer,
        nperseg: int = 1024,
        threshold_db_above_noise: float = 6.0,
        min_bandwidth_hz: float = 0.0
    ) -> List[DetectedSOI]:
        """
        Detect candidate Signals of Interest (SOIs) using Welch PSD and adaptive thresholding.
        """
        freqs, psd = cls.compute_psd(buffer, nperseg=nperseg)
        # Avoid log of zero
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-18))
        # Light smoothing: removes single-bin Welch fluctuations, keeps a tone's 3-4 bin main lobe
        sm_db = dsp_signal.medfilt(psd_db, 3)

        noise_floor_db, mad_db = cls.estimate_noise_floor(sm_db)
        # 3 x MAD guards against PSD fluctuation, but in a busy spectrum (several signals, deep empty
        # regions) the MAD is the spread of the signals themselves (~30 dB) and hid all but carrier lines
        detection_threshold_db = noise_floor_db + max(threshold_db_above_noise,
                                                      min(3.0 * mad_db, threshold_db_above_noise + 4.0))

        above = np.flatnonzero(sm_db > detection_threshold_db)
        freq_step = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
        if not len(above):
            return []
        runs = []
        s = p = above[0]
        for i in above[1:]:
            if i - p > 1:
                runs.append((s, p))
                s = i
            p = i
        runs.append((s, p))
        # Islands of one signal (a 2FSK tone line next to its main lobe, a PSD dip at an AM
        # carrier) are merged: a gap narrower than half the wider neighbour is the same signal.
        merged = [runs[0]]
        for a, b in runs[1:]:
            pa, pb = merged[-1]
            if a - pb - 1 < max(2, 0.5 * max(pb - pa + 1, b - a + 1)):
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))
        lin = 10 ** (sm_db / 10)
        noise_lin = 10 ** (noise_floor_db / 10)
        cand = []
        for a, b in merged:
            pw = float(np.sum(np.clip(lin[a:b + 1] - noise_lin, 0, None)))
            cand.append((a, b, pw, float(psd_db[a:b + 1].max())))
        total = sum(c[2] for c in cand) or 1.0
        top_peak = max(c[3] for c in cand)

        sois: List[DetectedSOI] = []
        for a, b, pw, peak_power in cand:
            bw = (b - a + 1) * freq_step
            # Not a signal: negligible share of the detected power AND far below the strongest one
            # (filter skirts, stop-band ripple, spectral leakage of a neighbour)
            if pw / total < MIN_SOI_POWER_FRACTION and peak_power < top_peak - MAX_SOI_DYNAMIC_DB:
                continue
            if bw < min_bandwidth_hz:
                continue
            w = lin[a:b + 1] - noise_lin
            w = np.clip(w, 0, None)
            center_f = float(np.sum(freqs[a:b + 1] * w) / np.sum(w)) if np.sum(w) > 0 else float(freqs[a:b + 1].mean())
            snr_db = peak_power - noise_floor_db
            ev = Evidence(
                description=f"FFT peak detected at {center_f/1e6:.6f} MHz (BW: {bw/1e3:.2f} kHz, SNR: {snr_db:.1f} dB, "
                            f"{100 * pw / total:.1f} % of the detected power)",
                source="FFT_DETECTOR",
                score=min(1.0, max(0.1, snr_db / 30.0)),
                metadata={
                    "noise_floor_db": noise_floor_db,
                    "detection_threshold_db": detection_threshold_db,
                    "peak_power_db": peak_power,
                    "power_fraction": pw / total,
                }
            )
            sois.append(DetectedSOI(
                soi_id=len(sois) + 1,
                center_freq_hz=center_f,
                bandwidth_hz=bw,
                power_db=peak_power,
                snr_db=snr_db,
                start_sample=0,
                end_sample=len(buffer),
                evidence=[ev]
            ))
        return sois
