import os
import tempfile
import pytest
import numpy as np

from ingestion.format_detector import FormatDetector
from detection.fft_detector import FFTDetector
from estimation.sample_rate import SampleRateEstimator
from estimation.center_frequency import CenterFrequencyEstimator
from estimation.bandwidth import BandwidthEstimator
from estimation.snr import SNREstimator
from estimation.symbol_rate import SymbolRateEstimator
from gui.workers.pipeline_worker import Phase1PipelineWorker


def test_phase1_end_to_end_synthetic_signal():
    """
    Phase 1 Synthetic End-to-End Integration Test:
    Generates a synthetic IQ file with a BPSK tone + noise, runs through ingestion,
    detection, estimation, and verifies all parameters and provenance tags.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        iq_file_path = os.path.join(tmp_dir, "phase1_test.iq")

        fs = 2.0e6           # 2 MHz sample rate
        f_tone = 250.0e3     # 250 kHz tone offset
        sps = 10             # 10 samples per symbol -> 200 kBaud symbol rate
        num_symbols = 500
        total_samples = num_symbols * sps

        # Generate synthetic BPSK signal at f_tone
        t = np.arange(total_samples) / fs
        bits = np.random.choice([-1, 1], size=num_symbols)
        symbol_seq = np.repeat(bits, sps)
        carrier = np.exp(2j * np.pi * f_tone * t)

        clean_signal = symbol_seq * carrier
        noise = 0.05 * (np.random.randn(total_samples) + 1j * np.random.randn(total_samples))
        iq_samples = (clean_signal + noise).astype(np.complex64)

        iq_samples.tofile(iq_file_path)

        # 1. Pipeline Worker Execution
        config = {
            "file_path": iq_file_path,
            "dtype": "complex64",
            "sample_rate": fs,
            "center_frequency": 100.0e6,
            "threshold_db": 8.0
        }

        worker = Phase1PipelineWorker(config)
        results = worker.work()

        assert "buffer" in results
        assert "sois" in results
        assert "parameters" in results

        buffer = results["buffer"]
        sois = results["sois"]
        params = results["parameters"]

        # 2. Assert Signal Buffer
        assert len(buffer) == total_samples
        assert buffer.is_valid()
        assert buffer.sample_rate == fs
        assert buffer.center_frequency == 100.0e6

        # 3. Assert SOI Detection
        assert len(sois) >= 1
        soi_center_freqs = [soi.center_freq_hz for soi in sois]
        # Should detect SOI near 100 MHz + 250 kHz = 100.25 MHz (within 50 kHz for 200 kBaud bandwidth)
        closest = min(soi_center_freqs, key=lambda f: abs(f - 100.25e6))
        assert abs(closest - 100.25e6) < 50.0e3

        # 4. Assert Parameter Provenance & Estimation
        assert params["sample_rate"].source == "KNOWN"
        assert params["sample_rate"].value == fs

        assert params["center_frequency"].source == "ESTIMATED"
        assert params["center_frequency"].confidence > 0.5

        assert params["bandwidth"].source == "ESTIMATED"
        assert params["bandwidth"].value > 0

        assert params["snr"].source == "ESTIMATED"
        assert params["snr"].value > 10.0  # Clean signal should have >10 dB SNR

        assert params["symbol_rate"].source == "ESTIMATED"
        # 200 kBaud symbol rate
        assert abs(params["symbol_rate"].value - (fs / sps)) < 40.0e3
