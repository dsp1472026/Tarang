"""
Tests for the image round-trip pipeline: framing, bit helpers, transmitter, receiver (through
the project's PSK / QAM / FSK receivers), format detection and SHA-256 verification.

Every test writes only under pytest's tmp_path. Images are small (8x8 BMP, 246 bytes; the
16QAM and 4FSK tests use a 64-byte payload) to keep the receivers fast.
"""

import os

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

from synthetic.bitops import bits_to_bytes, bytes_to_bits
from synthetic.image_framing import SYNC_WORD, build_frame, parse_frame
from synthetic.iq_io import write_iq
from synthetic.rx_image import detect_format, receive_image
from synthetic.tx_image import add_awgn, fec_encode, modulate, transmit_image
from synthetic.verify_roundtrip import verify

SPS, BAUD = 8, 1200


@pytest.fixture
def bmp(tmp_path):
    """8x8 checkerboard BMP (246 bytes)."""
    im = Image.new("RGB", (8, 8))
    for x in range(8):
        for y in range(8):
            im.putpixel((x, y), (255, 255, 255) if (x + y) % 2 else (0, 0, 0))
    path = tmp_path / "check.bmp"
    im.save(path)
    return str(path)


@pytest.fixture
def small_png(tmp_path):
    """A 64-byte-class PNG-headed payload (random body keeps it incompressible)."""
    data = bytes.fromhex("89504E470D0A1A0A") + np.random.default_rng(7).integers(0, 256, 56, dtype=np.uint8).tobytes()
    path = tmp_path / "small.png"
    path.write_bytes(data)
    return str(path)


def _roundtrip(tmp_path, image, modulation, fec=True):
    tx = transmit_image(image, modulation, str(tmp_path / "sig"), fec=fec, sps=SPS, baud=BAUD, snr_db=25.0)
    out = str(tmp_path / f"rx_{modulation.lower()}{os.path.splitext(image)[1]}")
    rx = receive_image(tx["iq_cs16_path"], modulation, out, fec=fec, sps=SPS, baud=BAUD)
    return tx, rx, verify(image, rx["output_image_path"])


def _write_custom_frame(tmp_path, frame: bytes, modulation="BPSK"):
    """Transmit an arbitrary (possibly broken) frame through the same FEC + modulator chain."""
    x = add_awgn(modulate(fec_encode(bytes_to_bits(frame)), modulation, SPS), modulation, SPS, 25.0, 1)
    stem = str(tmp_path / "custom")
    write_iq(stem, x, float(SPS * BAUD), 0.0, {})
    return stem + ".iq"


def test_framing_roundtrip():
    p = np.random.default_rng(3).integers(0, 256, 100, dtype=np.uint8).tobytes()
    assert parse_frame(build_frame(p)) == (p, True)


def test_bytes_bits_roundtrip():
    data = np.random.default_rng(4).integers(0, 256, 257, dtype=np.uint8).tobytes()
    bits = bytes_to_bits(data)
    assert bits.dtype == np.uint8 and len(bits) == 8 * len(data)
    assert bits_to_bytes(bits) == data
    assert bits_to_bytes(np.r_[bits, [1, 0, 1]]) == data                 # partial byte dropped
    assert "".join(map(str, bytes_to_bits(SYNC_WORD))) == "00011010110011111111110000011101"   # MSB first


def test_transmit_bpsk_writes_three_files(tmp_path, bmp):
    tx = transmit_image(bmp, "BPSK", str(tmp_path / "sig"))
    for key in ("iq_cs16_path", "iq_cf32_path", "wav_path"):
        assert os.path.getsize(tx[key]) > 0
    assert os.path.basename(tx["iq_cs16_path"]) == "img_check_bpsk.iq"
    n16, n32 = os.path.getsize(tx["iq_cs16_path"]) // 4, os.path.getsize(tx["iq_cf32_path"]) // 8
    assert n16 == n32 > tx["symbol_count"] * SPS                        # cs16 and cf32 hold the same samples
    rate, pcm = wavfile.read(tx["wav_path"])
    assert rate == 48000 and pcm.dtype == np.int16 and pcm.ndim == 1
    assert tx["fec"] is True and tx["payload_bytes"] == 246 and tx["frame_bytes"] == 258
    assert tx["coded_bits"] == 2 * (8 * 258 + 6)                          # rate 1/2 + K-1 zero tail
    assert tx["sync_word_hex"] == "1ACFFC1D"


def test_receive_bpsk_recovers_payload_without_fec(tmp_path, bmp):
    tx, rx, v = _roundtrip(tmp_path, bmp, "BPSK", fec=False)
    assert tx["coded_bits"] == 8 * 258
    assert rx["crc_ok"] and not rx["fec_decoded"] and v["match"]


def test_receive_bpsk_recovers_payload_with_fec(tmp_path, bmp):
    _, rx, v = _roundtrip(tmp_path, bmp, "BPSK")
    assert rx["crc_ok"] and rx["fec_decoded"] and rx["detected_format"] == "BMP"
    assert v["match"] and v["reconstructed_bytes"] == 246


def test_receive_qpsk_recovers_payload_with_fec(tmp_path, bmp):
    _, rx, v = _roundtrip(tmp_path, bmp, "QPSK")
    assert rx["crc_ok"] and rx["fec_decoded"] and v["match"]


def test_receive_16qam_recovers_payload_with_fec(tmp_path, small_png):
    _, rx, v = _roundtrip(tmp_path, small_png, "16QAM")
    assert rx["crc_ok"] and rx["fec_decoded"] and rx["detected_format"] == "PNG" and v["match"]


def test_receive_4fsk_recovers_payload_with_fec(tmp_path, small_png):
    _, rx, v = _roundtrip(tmp_path, small_png, "4FSK")
    assert rx["crc_ok"] and rx["fec_decoded"] and v["match"]


def test_receive_detects_corrupted_crc(tmp_path):
    frame = bytearray(build_frame(b"BM" + bytes(range(100))))
    frame[-2] ^= 0xFF                                                    # CRC no longer matches
    out = tmp_path / "out.bmp"
    rx = receive_image(_write_custom_frame(tmp_path, bytes(frame)), "BPSK", str(out), sps=SPS, baud=BAUD)
    assert rx["sync_found"] and not rx["crc_ok"]
    assert rx["output_image_path"] is None and not out.exists()


def test_receive_detects_missing_sync(tmp_path):
    frame = b"\x00\x00\x00\x00" + build_frame(b"BM" + bytes(range(100)))[4:]
    out = tmp_path / "out.bmp"
    rx = receive_image(_write_custom_frame(tmp_path, frame), "BPSK", str(out), sps=SPS, baud=BAUD)
    assert not rx["sync_found"] and not rx["crc_ok"] and not out.exists()


def test_detect_format_jpeg():
    assert detect_format(bytes.fromhex("FFD8FFE000104A464946")) == "JPEG"
    assert detect_format(bytes.fromhex("FFD8")) == "UNKNOWN"             # too short for the magic


def test_detect_format_png():
    assert detect_format(bytes.fromhex("89504E470D0A1A0A0000000D")) == "PNG"
    assert detect_format(b"GIF89a") == "GIF"


def test_detect_format_bmp():
    assert detect_format(b"BM\xf6\x00\x00\x00") == "BMP"
    assert detect_format(b"\x00\x01\x02") == "UNKNOWN"


def test_verify_identical_files_returns_match_true(tmp_path, bmp):
    copy = tmp_path / "copy.bmp"
    copy.write_bytes(open(bmp, "rb").read())
    v = verify(bmp, str(copy))
    assert v["match"] and v["original_sha256"] == v["reconstructed_sha256"]
    assert v["original_bytes"] == v["reconstructed_bytes"] == 246
    other = tmp_path / "other.bmp"
    other.write_bytes(b"BM" + bytes(244))
    assert verify(bmp, str(other))["match"] is False
    assert verify(bmp, None)["match"] is False
