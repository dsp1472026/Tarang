import numpy as np
import pytest
from conditioning.descrambler import LFSRDescrambler


def test_descrambler_candidate_generator():
    candidates = LFSRDescrambler.candidate_generator()
    assert "DISABLED" in candidates
    assert "PN9" in candidates
    assert "PN15" in candidates
    assert "CCSDS" in candidates


def test_descrambler_scramble_descramble_roundtrip():
    data_bits = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8)

    for preset in ["PN9", "PN11", "PN15", "DVB-S", "CCSDS"]:
        scrambled = LFSRDescrambler.process(data_bits, preset=preset)
        descrambled = LFSRDescrambler.process(scrambled, preset=preset)
        assert np.array_equal(descrambled, data_bits), f"Roundtrip failed for preset {preset}"


def test_descrambler_disabled_mode():
    bits = np.array([1, 0, 1, 1], dtype=np.uint8)
    out = LFSRDescrambler.process(bits, preset="DISABLED")
    assert np.array_equal(out, bits)


def test_descrambler_empty_and_invalid():
    empty_bits = LFSRDescrambler.process(np.array([], dtype=np.uint8), preset="PN9")
    assert len(empty_bits) == 0

    with pytest.raises(ValueError):
        LFSRDescrambler.process(np.array([1, 0]), preset="UNKNOWN_PRESET")
