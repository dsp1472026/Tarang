import numpy as np
import pytest
from conditioning.differential import DifferentialDecoder


def test_differential_candidate_generator():
    candidates = DifferentialDecoder.candidate_generator()
    assert "DISABLED" in candidates
    assert "DIFF_DECODE" in candidates


def test_differential_encode_decode_roundtrip():
    # Original data bits
    original_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0], dtype=np.uint8)

    # Encode with init_bit = 0
    encoded = DifferentialDecoder.process(original_bits, mode="DIFF_ENCODE", init_bit=0)

    # Decode with same init_bit = 0
    decoded = DifferentialDecoder.process(encoded, mode="DIFF_DECODE", init_bit=0)

    assert np.array_equal(decoded, original_bits)


def test_differential_disabled_mode():
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    passthrough = DifferentialDecoder.process(bits, mode="DISABLED")
    assert np.array_equal(passthrough, bits)


def test_differential_empty_and_invalid():
    empty_bits = DifferentialDecoder.process(np.array([], dtype=np.uint8), mode="DIFF_DECODE")
    assert len(empty_bits) == 0

    with pytest.raises(ValueError):
        DifferentialDecoder.process(np.array([1, 0]), mode="UNKNOWN_MODE")
