"""
Unit Tests for Synchronization Modules (AGC, Costas Loop, Gardner Timing Recovery).
"""

import numpy as np
import pytest

from core.signal_buffer import SignalBuffer
from synchronization.agc import AutomaticGainControl
from synchronization.costas_loop import CostasLoop
from synchronization.timing_recovery import SymbolTimingRecovery


def test_automatic_gain_control():
    # Low power signal
    samples = (0.1 * (np.random.randn(1000) + 1j * np.random.randn(1000))).astype(np.complex64)
    buf = SignalBuffer(samples=samples, sample_rate=1.0e6)

    res = AutomaticGainControl.process(buf, target_power=1.0)

    assert res.gain_applied > 1.0
    p_norm = np.mean(np.abs(res.buffer.samples)**2)
    assert abs(p_norm - 1.0) < 0.1
    assert res.parameter.source == "ESTIMATED"


def test_costas_loop():
    # Signal with 100 Hz CFO
    fs = 1.0e6
    t = np.arange(1000) / fs
    carrier = np.exp(2j * np.pi * 100.0 * t)
    samples = (np.repeat(np.random.choice([-1, 1], size=100), 10) * carrier).astype(np.complex64)
    buf = SignalBuffer(samples=samples, sample_rate=fs)

    res = CostasLoop.process(buf, order=2)

    assert res.parameter.source == "ESTIMATED"
    assert len(res.buffer.samples) == len(samples)


def test_symbol_timing_recovery():
    # 8 SPS signal
    bits = np.random.choice([-1, 1], size=50)
    samples = np.repeat(bits, 8).astype(np.complex64)
    buf = SignalBuffer(samples=samples, sample_rate=1.0e6)

    res = SymbolTimingRecovery.process(buf, samples_per_symbol=8.0)

    assert len(res.buffer.samples) > 0
    assert len(res.buffer.samples) <= len(samples) // 4
    assert res.parameter.source == "ESTIMATED"
