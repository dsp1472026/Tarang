"""
Minimal test-harness framing for known-payload vectors.

Frame (bit level, MSB first):

    +-----------------+-------------+------------------+----------+-----+
    | SYNC (96 bits)  | LEN (16 b)  | PAYLOAD (8*LEN)  | CRC-16   | pad |
    +-----------------+-------------+------------------+----------+-----+

* SYNC is the first 96 bits of the PN127 m-sequence (x^7 + x^6 + 1). 96 is a
  multiple of 1, 2, 3, 4, 6 and 8, so the sync word always occupies whole
  symbols for BPSK/QPSK/8PSK/16QAM/64QAM/256QAM. It doubles as a training
  sequence for timing acquisition and as the reference for carrier-phase
  ambiguity resolution.
* CRC-16/CCITT-FALSE covers LEN + PAYLOAD and gives an independent integrity check.
* Zero padding brings the frame to a whole number of symbols.

This is *not* the L10 frame-analysis stage; it only exists so the synthetic
vectors have a known reference for validation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

SYNC_LEN = 96
LEN_BITS = 16
CRC_BITS = 16


def pn_sequence(n: int, taps=(7, 6), seed: int = 0x7F) -> np.ndarray:
    """Fibonacci LFSR m-sequence (default PN127, polynomial x^7 + x^6 + 1)."""
    deg = max(taps)
    state = seed & ((1 << deg) - 1)
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = state & 1
        fb = 0
        for t in taps:
            fb ^= (state >> (deg - t)) & 1
        state = (state >> 1) | (fb << (deg - 1))
    return out


SYNC_BITS = pn_sequence(SYNC_LEN)


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits) // 8 * 8
    return np.packbits(bits[:n]).tobytes()


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


@dataclass
class FrameLayout:
    total_bits: int
    sync_bits: int
    header_bits: int
    payload_bits: int
    crc_bits: int
    pad_bits: int

    def to_dict(self):
        return self.__dict__.copy()


def build_frame(payload: bytes, bits_per_symbol: int) -> tuple[np.ndarray, FrameLayout]:
    if len(payload) >= 1 << LEN_BITS:
        raise ValueError("payload too long for 16-bit length field")
    header = len(payload).to_bytes(2, "big")
    crc = crc16_ccitt(header + payload).to_bytes(2, "big")
    bits = np.concatenate([SYNC_BITS, bytes_to_bits(header + payload + crc)])
    pad = (-len(bits)) % bits_per_symbol
    bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    layout = FrameLayout(len(bits), SYNC_LEN, LEN_BITS, 8 * len(payload), CRC_BITS, pad)
    return bits, layout


@dataclass
class ParsedFrame:
    length: int
    payload: bytes
    crc_received: int
    crc_computed: int

    @property
    def crc_ok(self) -> bool:
        return self.crc_received == self.crc_computed


def parse_frame(bits_after_sync: np.ndarray) -> Optional[ParsedFrame]:
    """Parse LEN + PAYLOAD + CRC from the bits that follow a detected sync word."""
    if len(bits_after_sync) < LEN_BITS + CRC_BITS:
        return None
    length = int(bits_to_bytes(bits_after_sync[:LEN_BITS]).hex(), 16)
    need = LEN_BITS + 8 * length + CRC_BITS
    if len(bits_after_sync) < need:
        # Length field corrupted or frame truncated: return what is present, CRC will fail.
        length = max(0, (len(bits_after_sync) - LEN_BITS - CRC_BITS) // 8)
        need = LEN_BITS + 8 * length + CRC_BITS
    body = bits_to_bytes(bits_after_sync[:LEN_BITS + 8 * length])
    crc_rx = int(bits_to_bytes(bits_after_sync[LEN_BITS + 8 * length:need]).hex() or "0", 16)
    return ParsedFrame(length, body[2:], crc_rx, crc16_ccitt(body))
