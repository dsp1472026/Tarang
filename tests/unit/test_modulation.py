"""
Unit Tests for Modulation Classification and DSP Feature Extraction.
"""

import numpy as np
import pytest

from core.signal_buffer import SignalBuffer
from modulation.dsp_features import DSPFeatureExtractor
from modulation.ml_classifier import HybridModulationClassifier


def test_dsp_feature_extraction():
    # Synthetic QPSK signal
    bits = np.random.choice([0, 1, 2, 3], size=200)
    constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    syms = constellation[bits]
    samples = np.repeat(syms, 8).astype(np.complex64)

    feats = DSPFeatureExtractor.extract(samples)

    assert feats.c40_c42_ratio >= 0.0
    assert feats.ring_count in [1, 2, 3, 4]
    assert feats.phase_symmetry_fold in [2, 4, 8]
    assert "c40_mag" in feats.metadata


def test_hybrid_modulation_classifier():
    classifier = HybridModulationClassifier()

    # Synthetic BPSK signal
    bits = np.random.choice([-1, 1], size=100)
    samples = np.repeat(bits, 8).astype(np.complex64)
    buf = SignalBuffer(samples=samples, sample_rate=1.0e6)

    param = classifier.classify(buf)

    assert param.name == "modulation"
    assert param.source == "ESTIMATED"
    assert param.confidence > 0.0
    assert len(param.evidence) == 3
