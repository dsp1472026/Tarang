"""
Tests for the image-pipeline frame SYNC | LENGTH | PAYLOAD | CRC32 (synthetic/image_framing.py).
"""

import zlib

import numpy as np

from synthetic.image_framing import SYNC_WORD, build_frame, parse_frame


def _payload(n=100, seed=1):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def test_roundtrip_100_bytes():
    p = _payload()
    f = build_frame(p)
    assert f[:4] == SYNC_WORD == bytes.fromhex("1ACFFC1D")
    assert int.from_bytes(f[4:8], "big") == 100 and len(f) == 4 + 4 + 100 + 4
    assert int.from_bytes(f[-4:], "big") == zlib.crc32(f[:-4]) & 0xFFFFFFFF
    assert parse_frame(f) == (p, True)
    assert parse_frame(f + b"\x00\x00\x00") == (p, True)           # trailing padding ignored


def test_corrupted_crc_detected():
    p = _payload()
    f = bytearray(build_frame(p))
    f[20] ^= 0x01                                                   # one payload bit
    got, ok = parse_frame(bytes(f))
    assert not ok and len(got) == 100
    f2 = bytearray(build_frame(p))
    f2[-1] ^= 0x80                                                  # the CRC itself
    assert parse_frame(bytes(f2))[1] is False


def test_missing_sync_detected():
    f = bytearray(build_frame(_payload()))
    f[0] ^= 0xFF
    assert parse_frame(bytes(f)) == (b"", False)
    assert parse_frame(build_frame(_payload())[4:]) == (b"", False)


def test_truncated_frame_detected():
    f = build_frame(_payload())
    assert parse_frame(f[:-1]) == (b"", False)                      # CRC cut short
    assert parse_frame(f[:50]) == (b"", False)                      # payload cut short
    assert parse_frame(f[:6]) == (b"", False)                       # shorter than the header
    assert parse_frame(b"") == (b"", False)
