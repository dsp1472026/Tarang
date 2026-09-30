"""
Byte-level frame for the image round-trip pipeline (CCSDS-style attached sync marker).

    +----------------+----------------+----------------+---------------+
    | SYNC (4 bytes) | LENGTH (4 B)   | PAYLOAD (N B)  | CRC32 (4 B)   |
    +----------------+----------------+----------------+---------------+

    SYNC     0x1ACFFC1D, the CCSDS attached sync marker (ASM)
    LENGTH   payload length in bytes, big-endian uint32
    PAYLOAD  raw file bytes
    CRC32    zlib.crc32 over SYNC + LENGTH + PAYLOAD, big-endian uint32

This lives beside synthetic/framing.py, not in it: that module is the PN127 / CRC-16 text
frame the existing PSK / QAM / FSK receivers, sync search and generators are built on, and
its build_frame / parse_frame have a different, bit-level interface.
"""

import zlib
from typing import Tuple

SYNC_WORD = bytes.fromhex("1ACFFC1D")
HEADER_BYTES = 8                  # SYNC + LENGTH
CRC_BYTES = 4
OVERHEAD_BYTES = HEADER_BYTES + CRC_BYTES


def crc32_of(data: bytes) -> int:
    """zlib CRC-32 of data as an unsigned int."""
    return zlib.crc32(data) & 0xFFFFFFFF


def build_frame(payload_bytes: bytes) -> bytes:
    """SYNC | LENGTH | PAYLOAD | CRC32 for payload_bytes."""
    payload_bytes = bytes(payload_bytes)
    if len(payload_bytes) >= 1 << 32:
        raise ValueError("payload too long for the 32-bit length field")
    body = SYNC_WORD + len(payload_bytes).to_bytes(4, "big") + payload_bytes
    return body + crc32_of(body).to_bytes(4, "big")


def parse_frame(frame_bytes: bytes) -> Tuple[bytes, bool]:
    """
    Parse a frame that starts at its SYNC word.

    Returns (payload, crc_ok). A missing SYNC or a LENGTH that the frame is too short to hold
    (e.g. a truncated frame) gives (b"", False); a CRC mismatch gives (payload, False).
    Bytes after the CRC are ignored.
    """
    frame_bytes = bytes(frame_bytes)
    if len(frame_bytes) < OVERHEAD_BYTES or frame_bytes[:4] != SYNC_WORD:
        return b"", False
    n = int.from_bytes(frame_bytes[4:8], "big")
    if len(frame_bytes) < OVERHEAD_BYTES + n:
        return b"", False
    body = frame_bytes[:HEADER_BYTES + n]
    crc = int.from_bytes(frame_bytes[HEADER_BYTES + n:OVERHEAD_BYTES + n], "big")
    return body[HEADER_BYTES:], crc == crc32_of(body)
