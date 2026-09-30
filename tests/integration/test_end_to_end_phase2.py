"""
Phase 2 End-to-End Integration Test.
Generates synthetic IQ file, executes Phase2PipelineWorker, and validates
Layers 1 to 6 (Ingestion, Detection, Estimation, Hybrid Modulation Classification, Sync, Demodulation).
"""

import os
import tempfile
import numpy as np
import pytest

from gui.workers.phase2_worker import Phase2PipelineWorker


def test_phase2_end_to_end_synthetic_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        iq_file_path = os.path.join(tmp_dir, "phase2_test.iq")

        fs = 2.0e6           # 2 MHz sample rate
        f_tone = 100.0e3     # 100 kHz tone offset
        sps = 10             # 10 samples per symbol
        num_symbols = 400
        total_samples = num_symbols * sps

        # Generate synthetic QPSK signal
        t = np.arange(total_samples) / fs
        symbols_idx = np.random.choice([0, 1, 2, 3], size=num_symbols)
        constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        symbol_seq = np.repeat(constellation[symbols_idx], sps)
        carrier = np.exp(2j * np.pi * f_tone * t)

        clean_signal = symbol_seq * carrier
        noise = 0.02 * (np.random.randn(total_samples) + 1j * np.random.randn(total_samples))
        iq_samples = (clean_signal + noise).astype(np.complex64)

        iq_samples.tofile(iq_file_path)

        config = {
            "file_path": iq_file_path,
            "dtype": "complex64",
            "sample_rate": fs,
            "center_frequency": 100.0e6,
            "threshold_db": 6.0
        }

        worker = Phase2PipelineWorker(config)
        results = worker.work()

        assert "buffer" in results
        assert "sois" in results
        assert "parameters" in results
        assert "agc_result" in results
        assert "costas_result" in results
        assert "timing_result" in results
        assert "demod_result" in results

        params = results["parameters"]

        # Validate Parameters & Provenance Tags
        assert params["sample_rate"].source == "KNOWN"
        assert params["center_frequency"].source == "ESTIMATED"
        assert params["bandwidth"].source == "ESTIMATED"
        assert params["symbol_rate"].source == "ESTIMATED"
        assert params["snr"].source == "ESTIMATED"
        assert params["modulation"].source == "ESTIMATED"

        # Validate Demodulation Output
        demod_res = results["demod_result"]
        assert len(demod_res.bits) > 0
        assert demod_res.evm_pct < 30.0
