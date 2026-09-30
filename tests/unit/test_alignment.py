import numpy as np
import pytest
from conditioning.alignment import BitAlignment


def test_alignment_candidate_generator():
    candidates = BitAlignment.candidate_generator()
    assert candidates == [0, 1, 2, 3, 4, 5, 6, 7]


def test_alignment_shift_bits():
    bits = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    shifted = BitAlignment.shift_bits(bits, shift=3)
    assert np.array_equal(shifted, np.array([0, 1, 0, 1, 0], dtype=np.uint8))


def test_alignment_pattern_search():
    pattern = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
    prefix_garbage = np.array([0, 1, 1], dtype=np.uint8)
    stream = np.concatenate([prefix_garbage, pattern, np.array([0, 0, 1], dtype=np.uint8)])

    aligned, offset = BitAlignment.align_to_pattern(stream, pattern)
    assert offset == 3
    assert np.array_equal(aligned[: len(pattern)], pattern)


def test_alignment_empty():
    empty_bits = BitAlignment.shift_bits(np.array([], dtype=np.uint8), shift=2)
    assert len(empty_bits) == 0

    aligned, offset = BitAlignment.align_to_pattern(np.array([], dtype=np.uint8), np.array([1, 0]))
    assert len(aligned) == 0
    assert offset == 0
