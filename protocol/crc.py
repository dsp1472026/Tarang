"""
CRC catalogue (Rocksoft model: width, poly, init, refin, refout, xorout; check = CRC of b"123456789").
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CRCSpec:
    name: str
    width: int
    poly: int
    init: int
    refin: bool
    refout: bool
    xorout: int
    check: int

    @property
    def nbytes(self) -> int:
        return self.width // 8


def _reflect(v: int, width: int) -> int:
    r = 0
    for i in range(width):
        if v >> i & 1:
            r |= 1 << (width - 1 - i)
    return r


_TABLES = {}


def _table(spec: CRCSpec):
    if spec not in _TABLES:
        top = 1 << (spec.width - 1)
        mask = (1 << spec.width) - 1
        t = []
        for b in range(256):
            crc = b << (spec.width - 8)
            for _ in range(8):
                crc = ((crc << 1) ^ spec.poly) if crc & top else (crc << 1)
            t.append(crc & mask)
        _TABLES[spec] = t
    return _TABLES[spec]


def compute(spec: CRCSpec, data: bytes) -> int:
    mask = (1 << spec.width) - 1
    t = _table(spec)
    crc = spec.init
    shift = spec.width - 8
    for b in data:
        if spec.refin:
            b = _reflect(b, 8)
        crc = ((crc << 8) & mask) ^ t[((crc >> shift) ^ b) & 0xFF]
    if spec.refout:
        crc = _reflect(crc, spec.width)
    return crc ^ spec.xorout


CATALOGUE: List[CRCSpec] = [
    CRCSpec("CRC-16/CCITT-FALSE", 16, 0x1021, 0xFFFF, False, False, 0x0000, 0x29B1),
    CRCSpec("CRC-16/XMODEM", 16, 0x1021, 0x0000, False, False, 0x0000, 0x31C3),
    CRCSpec("CRC-16/KERMIT", 16, 0x1021, 0x0000, True, True, 0x0000, 0x2189),
    CRCSpec("CRC-16/X-25", 16, 0x1021, 0xFFFF, True, True, 0xFFFF, 0x906E),
    CRCSpec("CRC-16/ARC", 16, 0x8005, 0x0000, True, True, 0x0000, 0xBB3D),
    CRCSpec("CRC-16/MODBUS", 16, 0x8005, 0xFFFF, True, True, 0x0000, 0x4B37),
    CRCSpec("CRC-32/ISO-HDLC", 32, 0x04C11DB7, 0xFFFFFFFF, True, True, 0xFFFFFFFF, 0xCBF43926),
    CRCSpec("CRC-32C", 32, 0x1EDC6F41, 0xFFFFFFFF, True, True, 0xFFFFFFFF, 0xE3069283),
    CRCSpec("CRC-8/SMBUS", 8, 0x07, 0x00, False, False, 0x00, 0xF4),
    CRCSpec("CRC-8/MAXIM", 8, 0x31, 0x00, True, True, 0x00, 0xA1),
]
BY_NAME = {c.name: c for c in CATALOGUE}


def self_test() -> bool:
    return all(compute(c, b"123456789") == c.check for c in CATALOGUE)
