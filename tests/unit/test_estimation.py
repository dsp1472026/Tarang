import pytest
import numpy as np

from core.signal_buffer import SignalBuffer
from estimation.sample_rate import SampleRateEstimator
from estimation.center_frequency import CenterFrequencyEstimator
from estimation.bandwidth import BandwidthEstimator
from estimation.snr import SNREstimator
from estimation.symbol_rate import SymbolRateEstimator


def test_sample_rate_estimator():
    buf_known = SignalBuffer(samples=np.ones(100, dtype=np.complex64), sample_rate=1e6)
    param_known = SampleRateEstimator.estimate(buf_known)
    assert param_known.source == "KNOWN"
    assert param_known.value == 1e6

    buf_unknown = SignalBuffer(samples=np.ones(100, dtype=np.complex64), sample_rate=None)
    param_unknown = SampleRateEstimator.estimate(buf_unknown)
    assert param_unknown.source == "UNKNOWN"
    assert param_unknown.value is None


def test_center_frequency_estimator():
    fs = 1e6
    f_offset = 50e3
    t = np.arange(2048) / fs
    samples = np.exp(2j * np.pi * f_offset * t).astype(np.complex64)
    buf = SignalBuffer(samples=samples, sample_rate=fs, center_frequency=100e6)

    param = CenterFrequencyEstimator.estimate(buf)
    assert param.source == "ESTIMATED"
    # Should be close to 100 MHz + 50 kHz = 100.05 MHz
    assert abs(param.value - 100.05e6) < 10e3


def test_bandwidth_estimator():
    fs = 1e6
    t = np.arange(2048) / fs
    # BPSK-like signal
    symbols = np.random.choice([-1, 1], size=2048)
    samples = (symbols + 0j).astype(np.complex64)
    buf = SignalBuffer(samples=samples, sample_rate=fs)

    param = BandwidthEstimator.estimate(buf)
    assert param.source == "ESTIMATED"
    assert param.value > 0


def test_snr_estimator():
    # Clean constant modulus signal vs noisy signal
    clean = np.exp(1j * np.linspace(0, 100, 5000)).astype(np.complex64)
    buf_clean = SignalBuffer(samples=clean, sample_rate=1e6)
    param_clean = SNREstimator.estimate(buf_clean)
    assert param_clean.value > 10.0  # High SNR

    noisy = clean + 0.5 * (np.random.randn(5000) + 1j * np.random.randn(5000))
    buf_noisy = SignalBuffer(samples=noisy.astype(np.complex64), sample_rate=1e6)
    param_noisy = SNREstimator.estimate(buf_noisy)
    assert param_noisy.value < param_clean.value


def test_symbol_rate_estimator():
    fs = 1e6
    sps = 10
    sym_rate = fs / sps  # 100 kBaud
    num_syms = 200
    bits = np.random.choice([-1, 1], size=num_syms)
    samples = np.repeat(bits, sps).astype(np.complex64)

    buf = SignalBuffer(samples=samples, sample_rate=fs)
    param = SymbolRateEstimator.estimate(buf)

    assert param.source == "ESTIMATED"
    assert abs(param.value - sym_rate) < 20e3
