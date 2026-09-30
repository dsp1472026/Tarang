"""
Unit Tests for Symbol Demodulator.
"""

import numpy as np
import pytest

from core.signal_buffer import SignalBuffer
from demodulation.demodulator import SymbolDemodulator


def test_symbol_demodulator_bpsk():
    syms = np.array([-1.0+0j, 1.0+0j, 1.0+0j, -1.0+0j], dtype=np.complex64)
    buf = SignalBuffer(samples=syms, sample_rate=100.0e3)

    res = SymbolDemodulator.demodulate(buf, modulation="BPSK")

    assert len(res.bits) == 4
    assert np.array_equal(res.bits, np.array([0, 1, 1, 0]))
    assert res.evm_pct < 1.0
    assert res.parameter.source == "ESTIMATED"


def test_symbol_demodulator_qpsk():
    constellation = SymbolDemodulator.CONSTELLATIONS["QPSK"]
    syms = np.tile(constellation, 10)
    buf = SignalBuffer(samples=syms, sample_rate=200.0e3)

    res = SymbolDemodulator.demodulate(buf, modulation="QPSK")

    assert len(res.bits) == 40 * 2  # 80 bits
    assert res.evm_pct < 1.0
    assert res.parameter.confidence > 0.90
