"""
Thin, importable modulator wrappers over the project's existing modulation code.

The generators (synthetic/generator.py, synthetic/fsk_generator.py) build their own
PN127 text frame internally, so they cannot carry an arbitrary bit stream. These
wrappers expose the same building blocks for arbitrary bits, without duplicating them:

    map_bpsk / map_qpsk / map_qam   dsp.constellations (Gray labels, unit average power)
    rrc_shape                       synthetic.generator.shape (continuous-time RRC)
    map_cpfsk                       dsp.cpm.cpm_phase (continuous-phase FSK), with the
                                    Gray label -> tone-position rule of fsk_generator
"""

import numpy as np

from dsp.constellations import get_constellation, gray, labels_of
from dsp.cpm import cpm_phase
from synthetic.generator import shape


def _pad(bits: np.ndarray, bps: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8) & 1
    return np.concatenate([bits, np.zeros((-len(bits)) % bps, dtype=np.uint8)])


def _map(bits: np.ndarray, name: str) -> np.ndarray:
    const = get_constellation(name)
    return const.points[labels_of(_pad(bits, const.bits_per_symbol), const.bits_per_symbol)]


def map_bpsk(bits: np.ndarray) -> np.ndarray:
    """Bits -> BPSK symbols (project convention: bit 0 -> -1, bit 1 -> +1)."""
    return _map(bits, "BPSK")


def map_qpsk(bits: np.ndarray) -> np.ndarray:
    """Bits -> Gray QPSK symbols at 45/135/225/315 degrees (zero-padded to 2 bits/symbol)."""
    return _map(bits, "QPSK")


def map_qam(bits: np.ndarray, M: int = 16) -> np.ndarray:
    """Bits -> square M-QAM symbols, Gray per axis, unit average power (zero-padded)."""
    return _map(bits, f"{M}QAM")


def rrc_shape(symbols: np.ndarray, sps: int, beta: float = 0.35, span: int = 10,
              guard_symbols: int = 24) -> np.ndarray:
    """RRC pulse shaping with silent guard symbols on both sides (unit-energy pulse)."""
    syms = np.concatenate([np.zeros(guard_symbols), np.asarray(symbols), np.zeros(guard_symbols)])
    return shape(syms, float(sps), beta, span, 0.0, 0.0, int(np.ceil(len(syms) * sps)))


def map_cpfsk(bits: np.ndarray, M: int = 4, h: float = 0.5, sps: int = 8, guard_symbols: int = 12) -> np.ndarray:
    """
    Bits -> continuous-phase M-FSK baseband (rect frequency pulse, Gray labels), unit amplitude,
    with silent guard symbols on both sides. Tone levels a = 2k - M + 1, f = a h Rs / 2.
    """
    bps = int(np.log2(M))
    labels = labels_of(_pad(bits, bps), bps)
    pos_of_label = np.empty(M, dtype=np.int64)
    pos_of_label[gray(np.arange(M))] = np.arange(M)             # same rule as synthetic.fsk_generator
    levels = 2 * pos_of_label[labels] - M + 1
    n = int(np.ceil((len(levels) + 2 * guard_symbols) * sps))
    t_sym = np.arange(n) / sps - guard_symbols
    active = (t_sym >= 0) & (t_sym < len(levels))
    return np.where(active, np.exp(1j * cpm_phase(levels, t_sym, h, "rect")), 0.0)
