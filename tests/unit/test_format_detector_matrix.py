import os
import time
import tempfile
import pytest
import numpy as np
from ingestion.format_detector import FormatDetector


def test_format_detector_existing_small_files():
    # 1. Test hello_bpsk.iq (int16)
    if os.path.exists("hello_bpsk.iq"):
        res = FormatDetector.auto_detect_iq_format("hello_bpsk.iq")
        assert res.format_name == "int16"
        assert res.confidence >= 80.0

    # 2. Test hello_bpsk_f32.iq (float32)
    if os.path.exists("hello_bpsk_f32.iq"):
        res = FormatDetector.auto_detect_iq_format("hello_bpsk_f32.iq")
        assert res.format_name in ("float32", "complex64")
        assert res.confidence >= 80.0


def test_format_detector_synthetic_10mb_int16():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "test_10mb_int16.iq")
        # 10 MB = 2,500,000 int16 pairs
        num_samples = 2_500_000
        data = (np.random.randn(num_samples * 2) * 5000).astype(np.int16)
        data.tofile(file_path)

        t0 = time.time()
        res = FormatDetector.auto_detect_iq_format(file_path)
        t_elapsed = time.time() - t0

        assert res.format_name == "int16"
        assert res.confidence >= 80.0
        assert t_elapsed < 5.0


def test_format_detector_header_skipping():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "test_header_int8.iq")
        # 4 KB ASCII metadata header
        header_text = ("# SigMF Metadata Header\n" + ("X" * 100) + "\n\n").encode("utf-8") * 35
        # Followed by 2 MB int8 IQ samples
        iq_data = (np.random.randn(2_000_000) * 40).astype(np.int8)

        with open(file_path, "wb") as f:
            f.write(header_text)
            f.write(iq_data.tobytes())

        res = FormatDetector.auto_detect_iq_format(file_path)
        assert res.format_name == "int8"
        assert res.confidence >= 80.0


def test_format_detector_leading_silence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "test_silence_int16.iq")
        # 1 MB leading silence (zeros)
        silence = np.zeros(250_000 * 2, dtype=np.int16)
        # Followed by active signal
        active = (np.random.randn(500_000 * 2) * 8000).astype(np.int16)

        with open(file_path, "wb") as f:
            f.write(silence.tobytes())
            f.write(active.tobytes())

        res = FormatDetector.auto_detect_iq_format(file_path)
        assert res.format_name == "int16"
        assert res.confidence >= 80.0


def test_format_detector_trailing_partial_sample():
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "recording_with_partial_bytes.iq")
        data = (np.random.randn(100_000) + 1j * np.random.randn(100_000)).astype(np.complex64)
        data = data / np.std(data)

        with open(file_path, "wb") as f:
            f.write(data.tobytes())
            f.write(b"\x01\x02\x03")  # 3 partial bytes

        res = FormatDetector.auto_detect_iq_format(file_path)
        assert res.format_name in ("float32", "complex64")
        assert res.confidence >= 80.0
        assert any("Trailing" in w for w in res.warnings)
