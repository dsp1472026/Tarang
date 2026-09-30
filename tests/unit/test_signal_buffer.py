import pytest
import numpy as np
from core.signal_buffer import SignalBuffer


def test_signal_buffer_creation():
    samples = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    buf = SignalBuffer(
        samples=samples,
        sample_rate=1e6,
        center_frequency=100e6,
        source_format="IQ",
        metadata={"gain": 10}
    )

    assert len(buf) == 3
    assert buf.is_valid()
    assert buf.sample_rate == 1e6
    assert buf.center_frequency == 100e6
    assert buf.source_format == "IQ"
    assert buf.metadata["gain"] == 10


def test_signal_buffer_slice():
    samples = np.array([1, 2, 3, 4, 5], dtype=np.complex64)
    buf = SignalBuffer(samples=samples, sample_rate=1000.0, start_time=0.0)

    sliced = buf.slice(1, 4)
    assert len(sliced) == 3
    assert np.array_equal(sliced.samples, samples[1:4])
    assert sliced.start_time == pytest.approx(0.001)


def test_signal_buffer_invalid():
    invalid_buf = SignalBuffer(samples=np.array([1, np.nan, 3]))
    assert not invalid_buf.is_valid()

    empty_buf = SignalBuffer(samples=np.array([]))
    assert not empty_buf.is_valid()
    assert len(empty_buf) == 0
