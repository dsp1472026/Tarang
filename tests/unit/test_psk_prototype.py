"""
Unit/integration tests for the Phase-1 PSK prototype
(dsp/, synthetic/, estimation/linear_mod_params, synchronization/{symbol_timing,carrier_psk},
modulation/psk_classifier, demodulation/{psk,psk_receiver}).
"""

import json
import os

import numpy as np
import pytest

from demodulation.psk_receiver import PSKReceiverConfig, receive_psk
from dsp.constellations import (bits_of, differential_demap, differential_encode, get_constellation, gray,
                                inverse_gray, labels_of, positions_to_points, psk)
from synthetic.framing import SYNC_BITS, build_frame, bytes_to_bits, crc16_ccitt, parse_frame
from synthetic.generator import Impairments, SignalSpec, generate
from synthetic.iq_io import read_iq, read_truth, write_iq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DATA = os.path.join(ROOT, "Test-Data")


# --------------------------------------------------------------------------- constellations

def test_gray_roundtrip():
    k = np.arange(256)
    assert np.array_equal(inverse_gray(gray(k)), k)
    # adjacent codes differ in exactly one bit
    assert all(bin(int(gray(i) ^ gray(i + 1))).count("1") == 1 for i in range(255))


@pytest.mark.parametrize("name", ["BPSK", "QPSK", "8PSK"])
def test_constellation_matches_legacy_ground_truth(name):
    """Mapping must agree with the gray_mapping table shipped with Test-Data."""
    path = os.path.join(TEST_DATA, f"hello_{name.lower()}_ground_truth.json")
    if not os.path.exists(path):
        pytest.skip("legacy ground truth not present")
    table = json.load(open(path))["gray_mapping"]
    c = get_constellation(name)
    for row in table:
        label = int(labels_of(np.array(row["bits"]), c.bits_per_symbol)[0])
        assert abs(c.points[label] - (row["I"] + 1j * row["Q"])) < 1e-9


@pytest.mark.parametrize("order", [2, 4, 8])
def test_psk_unit_power_and_hard_demap(order):
    c = psk(order)
    assert np.isclose(np.mean(np.abs(c.points) ** 2), 1.0)
    labels = np.arange(order)
    assert np.array_equal(c.demap_hard(c.points), bits_of(labels, c.bits_per_symbol))


@pytest.mark.parametrize("order", [2, 4, 8])
def test_llr_sign_matches_hard_decision(order):
    c = psk(order)
    rng = np.random.default_rng(0)
    y = c.points[rng.integers(order, size=500)] + 0.2 * (rng.standard_normal(500) + 1j * rng.standard_normal(500))
    llr = c.demap_soft(y, noise_var=0.08)
    assert np.array_equal((llr < 0).astype(np.uint8), c.demap_hard(y))


@pytest.mark.parametrize("order", [2, 4, 8])
def test_differential_roundtrip_is_rotation_invariant(order):
    c = psk(order)
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, size=60 * c.bits_per_symbol).astype(np.uint8)
    y = positions_to_points(differential_encode(labels_of(bits, c.bits_per_symbol), order), c)
    y = y * np.exp(1j * 1.234)                  # arbitrary carrier phase
    hard, _ = differential_demap(y, c, 0.01)
    assert np.array_equal(hard, bits)


# --------------------------------------------------------------------------- framing / io

def test_frame_roundtrip_and_crc():
    payload = b"HELLO WORLD"
    bits, layout = build_frame(payload, bits_per_symbol=3)
    assert len(bits) % 3 == 0 and np.array_equal(bits[:96], SYNC_BITS)
    f = parse_frame(bits[96:])
    assert f.payload == payload and f.crc_ok
    bad = bits.copy()
    bad[96 + 16 + 5] ^= 1
    assert not parse_frame(bad[96:]).crc_ok
    assert crc16_ccitt(b"123456789") == 0x29B1       # CRC-16/CCITT-FALSE check value


def test_iq_io_roundtrip(tmp_path):
    x = (np.arange(10) + 1j * np.arange(10)[::-1]).astype(np.complex128) / 10
    write_iq(str(tmp_path / "v"), x, 9600.0, 0.0, {"modulation": "BPSK"})
    a = read_iq(str(tmp_path / "v_f32.iq"))
    b = read_iq(str(tmp_path / "v.iq"))
    assert a.sample_rate == 9600.0 and a.datatype == "cf32_le" and b.datatype == "cs16_le"
    assert np.allclose(a.samples, x, atol=1e-6)
    assert np.corrcoef(np.abs(a.samples), np.abs(b.samples))[0, 1] > 0.9999
    assert read_truth(str(tmp_path / "v.iq"))["modulation"] == "BPSK"


# --------------------------------------------------------------------------- end to end

@pytest.mark.parametrize("mod,esn0", [("BPSK", 12.0), ("QPSK", 15.0), ("8PSK", 20.0)])
@pytest.mark.parametrize("differential", [False, True])
def test_blind_roundtrip_with_impairments(mod, esn0, differential):
    text = "HELLO WORLD " * 3
    spec = SignalSpec(mod, payload_text=text, sample_rate=10000.0, differential=differential,
                      impairments=Impairments(esn0_db=esn0 + (3 if differential else 0), cfo_hz=41.0,
                                              phase_deg=123.0, timing_offset=0.42, clock_ppm=100.0,
                                              dc_offset=0.04 - 0.02j, gain=0.2, seed=99))
    x, truth = generate(spec)
    r = receive_psk(x, spec.sample_rate)
    assert r.modulation == mod
    assert r.decode.frame is not None and r.decode.frame.crc_ok
    assert r.payload_text == text
    assert r.decode.method == ("differential" if differential else "sync-word")
    assert abs(r.parameters["sps"].value - 10000.0 / 1200.0) < 0.02
    assert abs(r.parameters["cfo"].value - 41.0) < 3.0
    # soft bits: signs equal the hard decisions everywhere, and the truth over the
    # CRC-protected length+payload+CRC region (the sync word tolerates a few errors).
    assert np.array_equal((r.llrs < 0).astype(np.uint8), r.hard_bits)
    ref = np.array(list(truth["frame_bits"]), dtype=np.uint8)[96:96 + 16 + 8 * len(text) + 16]
    s0 = r.decode.payload_bit_start - 16
    assert np.array_equal((r.llrs[s0:s0 + len(ref)] < 0).astype(np.uint8), ref)


def test_forced_modulation_and_unknown_sample_rate():
    spec = SignalSpec("QPSK", payload_text="ABC", impairments=Impairments(esn0_db=25.0, seed=3))
    x, _ = generate(spec)
    r = receive_psk(x, None, PSKReceiverConfig(modulation="QPSK"))
    assert r.classification is None and r.payload_text == "ABC"
    assert r.parameters["sample_rate"].source == "UNKNOWN"
    assert r.parameters["symbol_rate"].unit == "symbols/sample"


@pytest.mark.parametrize("name", ["hello_bpsk", "hello_qpsk", "hello_8psk"])
def test_legacy_test_data_classification_and_payload(name):
    path = os.path.join(TEST_DATA, name + ".iq")
    if not os.path.exists(path):
        pytest.skip("legacy Test-Data not present")
    f = read_iq(path)
    truth = read_truth(path)
    r = receive_psk(f.samples, f.sample_rate, PSKReceiverConfig(expected_payload="HELLO WORLD"))
    assert r.modulation == truth["modulation"]
    assert r.payload_text == "HELLO WORLD"
    assert r.decode.method == "known-payload"
