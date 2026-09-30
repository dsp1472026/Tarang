import os
import tempfile
import json
import struct
import pytest
import numpy as np

from ingestion.iq_reader import IQReader
from ingestion.wav_reader import WAVReader
from ingestion.sigmf_reader import SigMFReader
from ingestion.format_detector import FormatDetector


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_iq_reader_complex64(temp_dir):
    file_path = os.path.join(temp_dir, "test.iq")
    expected_samples = np.array([1 + 2j, 3 - 4j, -0.5 + 0.5j], dtype=np.complex64)
    expected_samples.tofile(file_path)

    buf = IQReader.read(file_path, dtype="complex64", sample_rate=1e6)
    assert buf.is_valid()
    assert len(buf) == 3
    assert buf.sample_rate == 1e6
    assert np.allclose(buf.samples, expected_samples)


def test_iq_reader_int16(temp_dir):
    file_path = os.path.join(temp_dir, "test_int16.iq")
    # 2 complex samples -> 4 int16 values: I0, Q0, I1, Q1
    raw_vals = np.array([16384, -16384, 32767, 0], dtype=np.int16)
    raw_vals.tofile(file_path)

    buf = IQReader.read(file_path, dtype="int16", iq_order="IQ")
    assert buf.is_valid()
    assert len(buf) == 2
    assert pytest.approx(buf.samples[0].real, rel=1e-3) == 0.5
    assert pytest.approx(buf.samples[0].imag, rel=1e-3) == -0.5


def test_wav_reader_stereo(temp_dir):
    file_path = os.path.join(temp_dir, "test.wav")

    # Construct synthetic 16-bit PCM stereo WAV file
    num_frames = 100
    sample_rate = 44100
    t = np.linspace(0, 1, num_frames, endpoint=False)
    left = (np.sin(2 * np.pi * 5 * t) * 32767).astype(np.int16)
    right = (np.cos(2 * np.pi * 5 * t) * 32767).astype(np.int16)

    interleaved = np.empty((num_frames * 2,), dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    pcm_bytes = interleaved.tobytes()
    data_size = len(pcm_bytes)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 2, sample_rate, sample_rate * 4, 4, 16,
        b"data", data_size
    )

    with open(file_path, "wb") as f:
        f.write(header)
        f.write(pcm_bytes)

    buf = WAVReader.read(file_path, mode="STEREO_IQ")
    assert buf.is_valid()
    assert len(buf) == num_frames
    assert buf.sample_rate == 44100.0
    assert buf.source_format == "WAV"


def test_sigmf_reader(temp_dir):
    meta_path = os.path.join(temp_dir, "test_sigmf.sigmf-meta")
    data_path = os.path.join(temp_dir, "test_sigmf.sigmf-data")

    meta_content = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": 2000000.0,
            "core:version": "1.0.0"
        },
        "captures": [{"core:frequency": 433920000.0}]
    }
    with open(meta_path, "w") as f:
        json.dump(meta_content, f)

    data = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex64)
    data.tofile(data_path)

    buf = SigMFReader.read(meta_path)
    assert buf.is_valid()
    assert len(buf) == 3
    assert buf.sample_rate == 2000000.0
    assert buf.center_frequency == 433920000.0
    assert buf.source_format == "SigMF"


def test_format_detector(temp_dir):
    iq_path = os.path.join(temp_dir, "sample.iq")
    np.array([1 + 1j], dtype=np.complex64).tofile(iq_path)

    assert FormatDetector.detect_format(iq_path) == "IQ"
    buf = FormatDetector.load_signal(iq_path, sample_rate=1e5)
    assert buf.is_valid()
    assert buf.sample_rate == 1e5


def test_auto_detect_iq_format_statistical(temp_dir):
    file_path = os.path.join(temp_dir, "recording_int16.iq")
    raw = (np.random.randn(2000) * 10000).astype(np.int16)
    raw.tofile(file_path)

    res = FormatDetector.auto_detect_iq_format(file_path)
    assert res.format_name == "int16"
    assert res.confidence >= 80.0
    assert res.bytes_per_sample == 4


def test_auto_detect_iq_format_filename_hint(temp_dir):
    file_path = os.path.join(temp_dir, "test_signal_cf32.iq")
    raw = (np.random.randn(100) + 1j * np.random.randn(100)).astype(np.complex64)
    raw.tofile(file_path)

    res = FormatDetector.auto_detect_iq_format(file_path)
    assert res.format_name in ("float32", "complex64")
    assert res.confidence >= 90.0

