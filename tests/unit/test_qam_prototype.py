"""
Tests for the Phase-3 QAM receiver (demodulation/qam_receiver.py) and its helpers.
"""

import os

import numpy as np
import pytest

from demodulation.qam_receiver import QAMReceiverConfig, mapped_qam, receive_qam
from dsp.constellations import qam
from synthetic.generator import Impairments, SignalSpec, generate
from tools.qam_validate import ser_theory

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize("order", [16, 64, 256])
def test_default_mapping_is_standard_gray(order):
    assert np.allclose(mapped_qam(order, "gray-asc", "gray-asc", False).points, qam(order).points)


def test_mappings_are_distinct_and_unit_power():
    pts = {tuple(np.round(mapped_qam(16, i, q, f).points, 9)) for i in ("gray-asc", "natural-desc")
           for q in ("gray-asc", "natural-asc") for f in (False, True)}
    assert len(pts) == 8
    assert np.isclose(np.mean(np.abs(mapped_qam(64, "natural-asc", "natural-desc", True).points) ** 2), 1.0)


def test_ser_theory_monotonic():
    assert ser_theory(16, 20) < ser_theory(16, 14) < 0.5
    assert ser_theory(256, 30) > ser_theory(16, 30)


def test_16qam_frame_blind():
    x, t = generate(SignalSpec("16QAM", payload_text="HELLO QAM WORLD", impairments=Impairments(
        esn0_db=30.0, cfo_hz=20.0, phase_deg=35.0, timing_offset=0.3)))
    r = receive_qam(x, t["sample_rate_hz"])
    assert r.modulation == "16QAM"
    assert r.decode.frame and r.decode.frame.crc_ok
    assert r.payload_text == "HELLO QAM WORLD"


@pytest.mark.parametrize("mod,es", [("64QAM", 32.0), ("256QAM", 38.0)])
def test_dense_qam_frame_known_rate(mod, es):
    x, t = generate(SignalSpec(mod, impairments=Impairments(esn0_db=es, cfo_hz=15.0, phase_deg=-60.0,
                                                            timing_offset=0.2)))
    r = receive_qam(x, t["sample_rate_hz"], QAMReceiverConfig(modulation=mod, sps=8.0))
    assert r.decode.frame and r.decode.frame.crc_ok
    assert r.payload_text == t["payload_text"]


def test_natural_mapping_legacy_style_capture():
    """Unframed ASCII with natural-binary, Q-descending labels (like all_modulations_output) is recovered."""
    text = "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS"
    const = mapped_qam(16, "natural-asc", "natural-desc", False)
    from dsp.constellations import labels_of
    from dsp.filters import rrc_pulse
    from synthetic.framing import bytes_to_bits
    syms = const.points[labels_of(bytes_to_bits(text.encode()), 4)]
    sps, n = 8, np.arange(len(syms) * 8 + 80)
    x = sum(a * rrc_pulse((n - 40 - 8 * k) / sps, 0.35) for k, a in enumerate(syms)) * np.exp(1j * 0.4)
    x = x + 0.01 * (np.random.default_rng(1).standard_normal(len(x)) + 1j * np.random.default_rng(2).standard_normal(len(x)))
    r = receive_qam(x, 9600.0, QAMReceiverConfig(modulation="16QAM", sps=8.0, expected_payload=text))
    assert r.payload_text == text
    assert any("natural" in e.description for e in r.decode.evidence)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "Test-Data", "qam_pangram_all_files",
                                                    "16qam_bright_vixens.iq")), reason="Test-Data not present")
def test_legacy_16qam_blind():
    from ingestion.capture import load_capture
    c = load_capture(os.path.join(ROOT, "Test-Data", "qam_pangram_all_files", "16qam_bright_vixens.iq"))
    r = receive_qam(c.samples, c.sample_rate)
    assert r.payload_text == "BRIGHT VIXENS JUMP; DOZY FOWL QUACK"
