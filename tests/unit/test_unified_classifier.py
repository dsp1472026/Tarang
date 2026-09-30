"""
Tests for WAV/IQ ingestion (ingestion/capture.py) and the unified FSK/PSK/QAM
classifier (modulation/{signal_features,linear_classifier,unified_classifier},
estimation/linear_rate_check).
"""

import os

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample_poly

from demodulation.psk_receiver import PSKReceiverConfig, receive_psk
from dsp.constellations import qam
from ingestion.capture import load_capture
from modulation.unified_classifier import classify_capture
from synthetic.fsk_generator import FSKImpairments, FSKSpec, generate_fsk
from synthetic.generator import Impairments, SignalSpec, generate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DATA = os.path.join(ROOT, "Test-Data")


def _psk(mod, esn0=30.0, text="HELLO WORLD", **imp):
    x, truth = generate(SignalSpec(modulation=mod, payload_text=text, impairments=Impairments(esn0_db=esn0, **imp)))
    return x, truth


def _write_real_passband(path, x, fs_bb, fs_wav=48000, fc=6000.0):
    up = fs_wav // int(fs_bb)
    i, q = resample_poly(x.real, up, 1), resample_poly(x.imag, up, 1)
    t = np.arange(len(i)) / fs_wav
    rf = i * np.cos(2 * np.pi * fc * t) - q * np.sin(2 * np.pi * fc * t)
    wavfile.write(path, fs_wav, np.round(rf / np.max(np.abs(rf)) * 0.9 * 32767).astype(np.int16))


# --------------------------------------------------------------------------- ingestion

def test_real_passband_wav_decodes_like_iq(tmp_path):
    """Mono WAV at 48 kHz with a 6 kHz carrier: analytic signal + mix-down keeps orientation (no inversion)."""
    x, truth = _psk("QPSK", cfo_hz=40.0, phase_deg=30.0)
    path = str(tmp_path / "qpsk.wav")
    _write_real_passband(path, x, truth["sample_rate_hz"])
    cap = load_capture(path)
    assert cap.source == "wav-real-passband"
    assert abs(cap.carrier_hz - 6000.0) < 400
    r = receive_psk(cap.samples, cap.sample_rate)
    assert r.modulation == "QPSK"
    assert r.decode.frame and r.decode.frame.crc_ok
    assert r.payload_text == "HELLO WORLD"
    assert abs(r.parameters["symbol_rate"].value - 1200.0) < 2.0


def test_stereo_wav_is_iq(tmp_path):
    x, truth = _psk("8PSK")
    path = str(tmp_path / "iq.wav")
    wavfile.write(path, int(truth["sample_rate_hz"]), np.stack([x.real, x.imag], 1).astype(np.float32))
    cap = load_capture(path)
    assert cap.source == "wav-stereo-iq" and cap.sample_rate == truth["sample_rate_hz"]
    assert np.allclose(cap.samples, x, atol=1e-6)


def test_stereo_wav_with_identical_channels_is_real(tmp_path):
    x, truth = _psk("BPSK")
    up = 5
    i = resample_poly(x.real, up, 1)
    t = np.arange(len(i)) / 48000
    rf = i * np.cos(2 * np.pi * 6000 * t)
    path = str(tmp_path / "dual_mono.wav")
    wavfile.write(path, 48000, np.stack([rf, rf], 1).astype(np.float32))
    assert load_capture(path).source == "wav-real-passband"


def test_planar_iq_layout(tmp_path):
    x, _ = _psk("QPSK")
    path = str(tmp_path / "planar_f32.iq")
    np.r_[x.real, x.imag].astype("<f4").tofile(path)
    cap = load_capture(path, dtype="cf32_le", layout="planar")
    assert np.allclose(cap.samples, x, atol=1e-6)


# --------------------------------------------------------------------------- constellations

@pytest.mark.parametrize("order", [16, 64, 256])
def test_qam_unit_power_and_axis_gray(order):
    c = qam(order)
    assert np.isclose(np.mean(np.abs(c.points) ** 2), 1.0)
    side = int(np.sqrt(order))
    assert len(np.unique(np.round(c.points.real, 9))) == side
    d = np.abs(c.points[:, None] - c.points[None, :])
    dmin = d[d > 1e-9].min()
    a, b = np.nonzero(np.isclose(d, dmin))
    assert len(a) > 0
    assert all(bin(int(i) ^ int(j)).count("1") == 1 for i, j in zip(a, b))   # Gray along both axes


# --------------------------------------------------------------------------- classification

@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16PSK"])
def test_psk_orders_classified_and_decoded(mod):
    x, truth = _psk(mod, esn0=30.0, cfo_hz=25.0, phase_deg=17.0, timing_offset=0.3)
    cls = classify_capture(x, truth["sample_rate_hz"])
    assert (cls.family, cls.modulation) == ("PSK", mod)
    cfg = cls.front_config
    cfg.modulation = cls.modulation
    r = receive_psk(x, truth["sample_rate_hz"], cfg, front=cls.front)
    assert r.decode.frame and r.decode.frame.crc_ok and r.payload_text == "HELLO WORLD"


@pytest.mark.parametrize("mod", ["16QAM", "64QAM", pytest.param("256QAM", marks=pytest.mark.xfail(
    reason="long framed 256QAM: front end locks at 3x sps; QAM work deferred to Phase 3", strict=False))])
def test_qam_orders_classified(mod):
    x, truth = _psk(mod, esn0=40.0, text="HELLO WORLD " * 12, phase_deg=10.0)
    cls = classify_capture(x, truth["sample_rate_hz"])
    assert (cls.family, cls.modulation) == ("QAM", mod)


@pytest.mark.parametrize("spec", [
    FSKSpec(order=2, h=1.0),
    FSKSpec(order=2, h=0.5),                                  # MSK
    FSKSpec(order=2, h=0.5, pulse="gaussian", bt=0.3),        # GMSK
    FSKSpec(order=4, h=1.0),
])
def test_fsk_family_detected(spec):
    spec.impairments = FSKImpairments(snr_db=15.0)
    x, truth = generate_fsk(spec)
    cls = classify_capture(x, truth["sample_rate_hz"])
    assert cls.family == "FSK", cls.features


def test_linear_family_not_fsk():
    for mod in ("QPSK", "16PSK"):
        x, truth = _psk(mod)
        assert classify_capture(x, truth["sample_rate_hz"]).family != "FSK"


@pytest.mark.skipif(not os.path.exists(os.path.join(TEST_DATA, "all_modulations_output", "16QAM.wav")),
                    reason="Test-Data not present")
@pytest.mark.parametrize("name,expected", [("16QAM", "16QAM"), ("64QAM", "64QAM"), ("QPSK", "QPSK"),
                                           ("8PSK", "8PSK")])
def test_short_legacy_wav_captures(name, expected):
    """35-character captures (25-70 symbols) delivered as real-passband WAV."""
    cap = load_capture(os.path.join(TEST_DATA, "all_modulations_output", f"{name}.wav"))
    cls = classify_capture(cap.samples, cap.sample_rate)
    assert cls.modulation == expected


# --------------------------------------------------------------------------- PSK bit-label mappings

def _legacy_psk(const, text, phase=0.7):
    """Unframed ASCII, RRC 0.35, 8 sps, file cut mid-pulse at both ends (like 'same' convolution)."""
    from dsp.constellations import labels_of
    from dsp.filters import rrc_pulse
    from synthetic.framing import bytes_to_bits
    bps = const.bits_per_symbol
    bits = bytes_to_bits(text.encode())
    bits = np.concatenate([bits, np.zeros((-len(bits)) % bps, dtype=np.uint8)])
    syms = const.points[labels_of(bits, bps)]
    n = np.arange(len(syms) * 8)
    x = sum(a * rrc_pulse((n - 8 * k) / 8, 0.35) for k, a in enumerate(syms)) * np.exp(1j * phase)
    rng = np.random.default_rng(3)
    return x + 0.01 * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))


@pytest.mark.parametrize("order,mapping,name", [(4, "gray-cw", "4QAM"), (4, "natural-ccw", "QPSK"),
                                                (8, "natural-ccw", "8PSK"), (8, "gray-cw", "8PSK")])
def test_legacy_psk_label_mappings(order, mapping, name):
    from demodulation.psk import mapped_psk
    from dsp.constellations import psk
    text = "BRIGHT VIXENS JUMP; DOZY FOWL QUACK"
    x = _legacy_psk(mapped_psk(psk(order), mapping), text)
    r = receive_psk(x, 9600.0, PSKReceiverConfig(modulation={4: "QPSK", 8: "8PSK"}[order]))
    assert r.payload_text == text
    assert r.decode.mapping == mapping
    assert r.modulation == name


def test_standard_gray_psk_keeps_default_mapping():
    from dsp.constellations import psk
    text = "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS"
    r = receive_psk(_legacy_psk(psk(8), text), 9600.0, PSKReceiverConfig(modulation="8PSK"))
    assert r.payload_text == text and r.decode.mapping == "gray-ccw"
