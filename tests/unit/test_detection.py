import pytest
import numpy as np

from core.signal_buffer import SignalBuffer
from detection.fft_detector import FFTDetector


def test_fft_detector_single_tone():
    fs = 1e6  # 1 MHz sample rate
    f_tone = 100e3  # 100 kHz tone offset
    t = np.arange(10000) / fs
    # Clean tone + mild white noise
    tone = np.exp(2j * np.pi * f_tone * t)
    noise = 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    samples = (tone + noise).astype(np.complex64)

    buf = SignalBuffer(samples=samples, sample_rate=fs, center_frequency=0.0)

    sois = FFTDetector.detect_sois(buf, nperseg=1024, threshold_db_above_noise=10.0)

    assert len(sois) >= 1
    # Check if one of the SOIs is centered near 100 kHz
    detected_freqs = [soi.center_freq_hz for soi in sois]
    closest_freq = min(detected_freqs, key=lambda f: abs(f - f_tone))
    assert abs(closest_freq - f_tone) < 15e3  # Within FFT bin resolution
    assert sois[0].evidence[0].source == "FFT_DETECTOR"


def test_fft_detector_noise_floor():
    psd_db = np.array([-80.0, -81.0, -79.0, -80.5, -80.0])
    median_db, mad_db = FFTDetector.estimate_noise_floor(psd_db)

    assert median_db == pytest.approx(-80.0)
    assert mad_db < 2.0
