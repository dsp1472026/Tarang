import os
import tempfile
import struct
import pytest
import numpy as np

from core.signal_buffer import SignalBuffer, SignalConfig
from ingestion.format_detector import FormatDetector, IQDetectionResult
from gui.workers.phase2_worker import Phase2PipelineWorker


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def test_wav_header_sample_rate_resolution(temp_dir):
    wav_path = os.path.join(temp_dir, "known_48k.wav")

    # Create synthetic 48 kHz stereo WAV file
    num_frames = 4800  # 0.1 sec
    sample_rate = 48000
    t = np.linspace(0, 0.1, num_frames, endpoint=False)
    left = (np.sin(2 * np.pi * 1000 * t) * 10000).astype(np.int16)
    right = (np.cos(2 * np.pi * 1000 * t) * 10000).astype(np.int16)

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

    with open(wav_path, "wb") as f:
        f.write(header)
        f.write(pcm_bytes)

    # Validate header format detection & fs resolution
    fmt = FormatDetector.detect_format(wav_path)
    assert fmt == "WAV"

    buf = FormatDetector.load_signal(wav_path)
    assert buf.sample_rate == 48000.0


def test_iq_duration_computed_sample_rate(temp_dir):
    iq_path = os.path.join(temp_dir, "test_duration.iq")

    # Create synthetic int16 IQ file (4 bytes per sample)
    # Target: fs = 2.0e6 Hz, duration = 0.005 s -> 10,000 samples -> 40,000 bytes
    num_samples = 10000
    file_size = num_samples * 4
    raw_vals = (np.random.randn(num_samples * 2) * 5000).astype(np.int16)
    raw_vals.tofile(iq_path)

    det = FormatDetector.auto_detect_iq_format(iq_path)
    assert det.format_name == "int16"
    assert det.bytes_per_sample == 4

    # Compute fs from user duration: 0.005 s
    duration_sec = 0.005
    total_complex = file_size / det.bytes_per_sample
    computed_fs = total_complex / duration_sec

    expected_fs = 2.0e6
    assert abs(computed_fs - expected_fs) / expected_fs < 0.001


def test_iq_format_auto_detection_all_dtypes(temp_dir):
    dtypes = {
        "int8": (np.int8, 2),
        "int16": (np.int16, 4),
        "float32": (np.float32, 8),
        "complex64": (np.complex64, 8),
        "complex128": (np.complex128, 16)
    }

    for fmt_name, (np_type, b_sample) in dtypes.items():
        fname = os.path.join(temp_dir, f"test_signal_{fmt_name}.iq")
        if fmt_name in ("complex64", "complex128"):
            data = (np.random.randn(2000) + 1j * np.random.randn(2000)).astype(np_type)
        else:
            data = (np.random.randn(4000) * 10).astype(np_type)
        data.tofile(fname)

        det = FormatDetector.auto_detect_iq_format(fname)
        assert det.format_name in (fmt_name, "float32", "complex64")
        assert det.bytes_per_sample == b_sample


def test_end_to_end_bpsk_demodulation_pipeline(temp_dir):
    iq_path = os.path.join(temp_dir, "bpsk_synth.iq")

    fs = 1.0e6
    f_tone = 50.0e3
    sps = 8
    num_symbols = 300
    total_samples = num_symbols * sps

    t = np.arange(total_samples) / fs
    np.random.seed(42)
    symbols_idx = np.random.choice([0, 1], size=num_symbols)
    constellation = np.array([1+0j, -1+0j])
    symbol_seq = np.repeat(constellation[symbols_idx], sps)
    carrier = np.exp(2j * np.pi * f_tone * t)

    clean_signal = symbol_seq * carrier
    noise = 0.01 * (np.random.randn(total_samples) + 1j * np.random.randn(total_samples))
    iq_samples = (clean_signal + noise).astype(np.complex64)
    iq_samples.tofile(iq_path)

    worker = Phase2PipelineWorker({
        "file_path": iq_path,
        "dtype": "complex64",
        "sample_rate": fs,
        "center_frequency": 0.0,
        "threshold_db": 6.0
    })

    res = worker.work()
    assert res["parameters"]["modulation"].value in ("BPSK", "QPSK")
    assert res["demod_result"].evm_pct < 30.0
    assert len(res["demod_result"].bits) > 0
