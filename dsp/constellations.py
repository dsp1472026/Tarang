"""
Constellation definitions with Gray labelling, plus hard/soft demapping.

Conventions (shared by the generator and every demodulator):
  * A symbol *label* is the integer formed by its bits, MSB first.
  * For M-PSK, the point at angular position k (angle = phi0 + 2*pi*k/M)
    carries label gray(k) = k ^ (k >> 1).
  * Default phi0 matches the existing Test-Data ground truth:
        BPSK  phi0 = pi     (bit 0 -> -1, bit 1 -> +1)
        QPSK  phi0 = pi/4   (00 -> 45 deg, 01 -> 135, 11 -> 225, 10 -> 315)
        8PSK  phi0 = 0      (000 -> 0 deg, 001 -> 45, 011 -> 90, ...)
  * LLR convention: llr = ln P(b=0 | y) / P(b=1 | y); hard bit = (llr < 0).
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

CHUNK = 1 << 16    # symbols per block of (symbols x points) distances: a 4.6 M-symbol capture against
                   # 256 points needed 8.7 GB at once


def gray(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k)
    return k ^ (k >> 1)


def inverse_gray(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g).copy()
    shift = g >> 1
    while np.any(shift):
        g ^= shift
        shift >>= 1
    return g


def bits_of(labels: np.ndarray, bps: int) -> np.ndarray:
    """labels (N,) -> bits (N*bps,) MSB first."""
    labels = np.asarray(labels, dtype=np.int64)
    shifts = np.arange(bps - 1, -1, -1)
    return ((labels[:, None] >> shifts[None, :]) & 1).astype(np.uint8).ravel()


def labels_of(bits: np.ndarray, bps: int) -> np.ndarray:
    """bits (N*bps,) MSB first -> labels (N,)."""
    b = np.asarray(bits, dtype=np.int64).reshape(-1, bps)
    weights = 1 << np.arange(bps - 1, -1, -1)
    return b @ weights


@dataclass(frozen=True)
class Constellation:
    name: str
    family: str
    points: np.ndarray        # points[label] -> complex, unit average power
    bits_per_symbol: int
    phase_symmetry: int       # rotations by 2*pi/phase_symmetry map the set onto itself

    @property
    def order(self) -> int:
        return len(self.points)

    @property
    def bit_table(self) -> np.ndarray:
        """(M, bps) uint8 table, row = label."""
        return bits_of(np.arange(self.order), self.bits_per_symbol).reshape(self.order, self.bits_per_symbol)

    def map_bits(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        if len(bits) % self.bits_per_symbol:
            raise ValueError(f"{len(bits)} bits is not a multiple of {self.bits_per_symbol}")
        return self.points[labels_of(bits, self.bits_per_symbol)]

    def nearest_labels(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        out = np.empty(len(y), dtype=np.int64)
        for a in range(0, len(y), CHUNK):             # (N, M) distances in blocks: bounded memory
            out[a:a + CHUNK] = np.argmin(np.abs(y[a:a + CHUNK, None] - self.points[None, :]), axis=1)
        return out

    def demap_hard(self, y: np.ndarray) -> np.ndarray:
        return bits_of(self.nearest_labels(y), self.bits_per_symbol)

    def demap_soft(self, y: np.ndarray, noise_var: float) -> np.ndarray:
        """
        Max-log LLRs: llr_b = (min_{s: b=1} |y-s|^2 - min_{s: b=0} |y-s|^2) / noise_var.

        noise_var is the complex noise variance per symbol (E|n|^2).
        """
        noise_var = max(float(noise_var), 1e-9)
        y = np.asarray(y)
        table = self.bit_table                                         # (M, bps)
        llr = np.empty((len(y), self.bits_per_symbol), dtype=np.float64)
        for a in range(0, len(y), CHUNK):                              # (N, M) in blocks: bounded memory
            d2 = np.abs(y[a:a + CHUNK, None] - self.points[None, :]) ** 2
            for b in range(self.bits_per_symbol):
                ones = table[:, b] == 1
                llr[a:a + CHUNK, b] = (d2[:, ones].min(axis=1) - d2[:, ~ones].min(axis=1)) / noise_var
        return llr.ravel().astype(np.float32)

    def evm(self, y: np.ndarray) -> float:
        """RMS EVM (fraction, not %) relative to the RMS constellation power (=1)."""
        if len(y) == 0:
            return float("nan")
        ref = self.points[self.nearest_labels(y)]
        return float(np.sqrt(np.mean(np.abs(y - ref) ** 2) / np.mean(np.abs(self.points) ** 2)))


_DEFAULT_PSK_PHI0 = {2: np.pi, 4: np.pi / 4.0}


def psk(order: int, phi0: float = None) -> Constellation:
    """Gray-labelled M-PSK constellation (see module docstring for conventions)."""
    if order < 2 or order & (order - 1):
        raise ValueError("PSK order must be a power of two >= 2")
    if phi0 is None:
        phi0 = _DEFAULT_PSK_PHI0.get(order, 0.0)
    k = np.arange(order)
    pts = np.empty(order, dtype=np.complex128)
    pts[gray(k)] = np.exp(1j * (phi0 + 2.0 * np.pi * k / order))
    name = {2: "BPSK", 4: "QPSK"}.get(order, f"{order}PSK")
    return Constellation(name=name, family="PSK", points=pts,
                         bits_per_symbol=int(np.log2(order)), phase_symmetry=order)


PSK_ORDERS: Dict[str, int] = {"BPSK": 2, "QPSK": 4, "4QAM": 4, "8PSK": 8, "16PSK": 16}
QAM_ORDERS: Dict[str, int] = {"16QAM": 16, "64QAM": 64, "256QAM": 256}


def qam(order: int) -> Constellation:
    """
    Square M-QAM, unit average power, Gray-coded per axis (label = I bits | Q bits,
    I/Q levels in increasing order). Used for classification; the bit labelling of
    the test captures is validated in Phase 3.
    """
    side = int(round(np.sqrt(order)))
    if side * side != order or side < 2 or side & (side - 1):
        raise ValueError("square QAM order must be 4, 16, 64, 256, ...")
    half = int(np.log2(side))
    levels = 2.0 * np.arange(side) - (side - 1)
    k = np.arange(order)
    gi, gq = k >> half, k & (side - 1)                 # Gray labels per axis
    pos_i, pos_q = inverse_gray(gi), inverse_gray(gq)  # level index of each label
    pts = levels[pos_i] + 1j * levels[pos_q]
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    return Constellation(name=f"{order}QAM", family="QAM", points=pts.astype(np.complex128),
                         bits_per_symbol=int(np.log2(order)), phase_symmetry=4)


def get_constellation(name: str) -> Constellation:
    key = name.upper().replace("-", "")
    if key in PSK_ORDERS:
        return psk(PSK_ORDERS[key])
    if key in QAM_ORDERS:
        return qam(QAM_ORDERS[key])
    raise ValueError(f"Unknown constellation: {name}")


# ---------------------------------------------------------------------------
# Differential M-PSK (phase-increment) coding
# ---------------------------------------------------------------------------

def differential_encode(labels: np.ndarray, order: int) -> np.ndarray:
    """
    Data labels -> transmitted angular positions.

    Each data label L is turned into a phase increment of inverse_gray(L) steps,
    so adjacent increments differ in one bit. A reference symbol (position 0)
    is prepended, so the output is one symbol longer than the input.
    """
    steps = inverse_gray(np.asarray(labels, dtype=np.int64))
    pos = np.concatenate([[0], np.cumsum(steps)]) % order
    return pos


def positions_to_points(pos: np.ndarray, const: Constellation) -> np.ndarray:
    """Angular position index -> constellation point (independent of labels)."""
    return const.points[gray(np.asarray(pos, dtype=np.int64))]


def differential_demap(y: np.ndarray, const: Constellation, noise_var: float):
    """
    Non-coherent differential detection: z_n = y_n * conj(y_{n-1}).

    Returns (hard_bits, llrs). The phase-difference symbol is demapped against a
    phi0 = 0 M-PSK constellation; LLRs use a doubled noise variance to account
    for noise in both symbols. Immune to any constant carrier-phase rotation.
    """
    order = const.order
    z = y[1:] * np.conj(y[:-1])
    z = z / np.maximum(np.abs(y[:-1]), 1e-9)
    ref = psk(order, phi0=0.0)
    return ref.demap_hard(z), ref.demap_soft(z, 2.0 * noise_var)
