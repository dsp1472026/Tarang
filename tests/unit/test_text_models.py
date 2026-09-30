"""Unframed text decisions: numeric / English / general message models and short-reading guards."""

import numpy as np
import pytest

from demodulation.psk import best_model, best_text_region, decode_psk, text_decisive
from demodulation.qam_receiver import decode_mapping_hypotheses
from dsp.constellations import get_constellation
from synthetic.framing import bytes_to_bits

PREAMBLE = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
                     1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
MESSAGES = ["PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS.", "9876543210 0123456789 31415926535",
            "20260926123456789"]


def _decode(mod, msg, preamble, seed=1):
    c = get_constellation(mod)
    bits = bytes_to_bits(msg.encode())
    if preamble:
        bits = np.r_[PREAMBLE, bits]
    bits = np.r_[bits, np.zeros((-len(bits)) % c.bits_per_symbol, dtype=np.uint8)]
    rng = np.random.default_rng(seed)
    y = c.map_bits(bits) * np.exp(2j * np.pi * 1 / c.phase_symmetry)          # unknown rotation
    y = y + 0.01 * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    if mod.endswith("QAM"):
        r, _ = decode_mapping_hypotheses(y, c.order, 0.01, None)
        return r
    return decode_psk(y, c, 0.01)


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK", "16PSK", "16QAM", "64QAM", "256QAM"])
@pytest.mark.parametrize("msg", MESSAGES)
@pytest.mark.parametrize("preamble", [False, True])
def test_unframed_text_and_numeric_decode(mod, msg, preamble):
    r = _decode(mod, msg, preamble)
    assert r is not None and r.method == "text-heuristic"
    assert msg in r.text


def test_message_models():
    assert best_model(b"9876543210 0123456789")[1] == "numeric"
    assert best_model(b"THE WEATHER REPORT FOR THE NORTHERN REGION")[1] == "english"
    # digits beat the lowercase junk a wrong rotation produces
    assert best_model(b"9876543210 0123456789")[0] > best_model(b"canlomfdgeEegdfmolnac")[0] + 1.0


def test_short_readings_need_stronger_evidence():
    assert not text_decisive(-3.0, -9.0, 5)                  # below TEXT_MIN_BYTES
    assert not text_decisive(-6.0, -9.0, 13)                 # short and weak
    assert text_decisive(-2.7, -4.2, 17)                     # short digit string, clear lead
    assert text_decisive(-6.0, -6.6, 40)                     # long symbol-heavy text: unchanged rule
    assert best_text_region(bytes_to_bits(b"ok"))[0] == -np.inf


@pytest.mark.parametrize("mod", ["8PSK", "16PSK", "16QAM"])
def test_short_random_streams_rarely_read_as_text(mod):
    c = get_constellation(mod)
    rng = np.random.default_rng(11)
    n_trials = 40 if mod.endswith("QAM") else 100
    hits = 0
    for _ in range(n_trials):
        n = int(rng.integers(8, 26)) * 8 // c.bits_per_symbol
        y = c.points[rng.integers(0, c.order, n)]
        if mod.endswith("QAM"):
            hits += decode_mapping_hypotheses(y, c.order, 0.01, None)[0] is not None
        else:
            hits += decode_psk(y, c, 0.01).method == "text-heuristic"
    assert hits <= 0.03 * n_trials
