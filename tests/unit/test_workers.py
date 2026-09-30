import os
import tempfile
import pytest
import numpy as np

from gui.workers.pipeline_worker import Phase1PipelineWorker


def test_pipeline_worker_execution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        iq_file = os.path.join(tmp_dir, "synth.iq")
        samples = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j] * 100, dtype=np.complex64)
        samples.tofile(iq_file)

        config = {
            "file_path": iq_file,
            "dtype": "complex64",
            "sample_rate": 1e6,
            "center_frequency": 100e6
        }

        worker = Phase1PipelineWorker(config)
        results = worker.work()

        assert "buffer" in results
        assert "sois" in results
        assert "parameters" in results
        assert results["buffer"].sample_rate == 1e6
        assert results["parameters"]["sample_rate"].source == "KNOWN"


def test_pipeline_worker_cancellation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        iq_file = os.path.join(tmp_dir, "synth.iq")
        np.array([1 + 1j] * 50, dtype=np.complex64).tofile(iq_file)

        config = {"file_path": iq_file, "dtype": "complex64"}
        worker = Phase1PipelineWorker(config)
        worker.cancel()

        results = worker.work()
        assert results == {}
