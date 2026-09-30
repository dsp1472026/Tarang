"""
Unit/integration tests for the Phase-2 FSK prototype
(dsp/cpm, synthetic/fsk_generator, estimation/fsk_estimator, modulation/fsk_classifier,
demodulation/{fsk_demod,fsk_receiver}).
"""

import json
import os
from fractions import Fraction

import numpy as np
import pytest

from demodulation.fsk_demod import (MapHyp, bcjr_ll, isi_viterbi, strobe_isi_taps, symbol_ll_to_bits)
from demodulation.fsk_receiver import FSKReceiverConfig, receive_fsk
from dsp.constellations import bits_of, gray
from dsp.cpm import cpm_frequency, cpm_phase, fsk_variant_name, levels_from_labels, phase_pulse
from estimation.fsk_estimator import fit_levels, occupied_band, phase_continuity, inst_freq
from modulation.fsk_classifier import classify_family
from synthetic.fsk_generator import FSKImpairments, FSKSpec, generate_fsk
from synthetic.generator import Impairments, SignalSpec, generate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- CPM model

@pytest.mark.parametrize("pulse,bt", [("rect", None), ("gaussian", 0.3), ("gaussian", 0.5)])
def test_cpm_phase_matches_brute_force(pulse, bt):
    rng = np.random.default_rng(0)
    lv = rng.choice([-3.0, -1.0, 1.0, 3.0], 60)
    t = np.linspace(-1, 61, 3000)
    ref = np.pi * 0.7 * np.sum(lv[None, :] * phase_pulse(t[:, None] - np.arange(60)[None, :] - 0.5, pulse, bt), axis=1)
    assert np.allclose(cpm_phase(lv, t, 0.7, pulse, bt), ref, atol=1e-8)


def test_msk_phase_advances_quarter_cycle_per_symbol():
    lv = np.array([1, 1, -1, 1], float)
    ph = cpm_phase(lv, np.arange(5.0), 0.5, "rect")
    assert np.allclose(np.diff(ph), np.pi / 2 * lv)
    f = cpm_frequency(lv, np.array([0.5, 2.5]), 0.5, "rect")
    assert np.allclose(f, [0.25, -0.25])                # +-Rs/4 tones


def test_variant_names():
    assert fsk_variant_name(2, 0.5, "rect") == "MSK"
    assert fsk_variant_name(2, 0.5, "gaussian", 0.3) == "GMSK"
    assert fsk_variant_name(2, 0.35, "gaussian", 0.3) == "GFSK"
    assert fsk_variant_name(2, 1.0, "rect") == "2FSK"
    assert fsk_variant_name(2, 0.5, "rect", continuous=False) == "2FSK"
    assert fsk_variant_name(4, 0.5, "rect") == "4FSK"


def test_gray_level_mapping():
    lv = levels_from_labels(np.arange(4), 4, "gray")      # labels 00,01,10,11 -> positions 0,1,3,2
    assert list(lv) == [-3, -1, 3, 1]
    assert list(levels_from_labels(np.arange(4), 4, "natural")) == [-3, -1, 1, 3]


# --------------------------------------------------------------------------- bit mapping / detectors

@pytest.mark.parametrize("hyp", [MapHyp("gray", False, False), MapHyp("natural", True, False),
                                 MapHyp("gray", False, True), MapHyp("natural", True, True)])
def test_symbol_ll_to_bits_roundtrip(hyp):
    M, rng = 4, np.random.default_rng(1)
    labels = rng.integers(0, M, 50)
    lab_of_pos = gray(np.arange(M)) if hyp.mapping == "gray" else np.arange(M)
    pos_of_lab = np.argsort(lab_of_pos)
    if hyp.differential:
        pos = np.concatenate([[0], np.cumsum(pos_of_lab[labels])]) % M
    else:
        pos = pos_of_lab[labels]
    if hyp.inverted:
        pos = M - 1 - pos
    LL = -10.0 * (np.arange(M)[None, :] != pos[:, None])
    bits, llr = symbol_ll_to_bits(LL, hyp)
    assert np.array_equal(bits, bits_of(labels, 2))
    assert np.array_equal((llr < 0).astype(np.uint8), bits)


def test_isi_viterbi_recovers_gaussian_isi_sequence():
    rng = np.random.default_rng(2)
    pos = rng.integers(0, 2, 200)
    taps = strobe_isi_taps("gaussian", 0.3)
    lv = 2 * pos - 1
    J = len(taps) // 2
    v = 0.1 + 0.25 * np.convolve(np.pad(lv, J, mode="edge"), taps[::-1], mode="valid") + 0.01 * rng.standard_normal(200)
    est = isi_viterbi(v, 2, taps, 0.5, 0.1)
    assert np.mean(est[J:-J] == pos[J:-J]) == 1.0


def test_bcjr_noiseless_msk():
    rng = np.random.default_rng(3)
    pos = rng.integers(0, 2, 80)
    sps = 8
    t = np.arange(80 * sps) / sps
    r = np.exp(1j * cpm_phase(2 * pos - 1, t, 0.5, "rect"))
    LL = bcjr_ll(r, t, 80, 2, Fraction(1, 2), "rect", None, 0.1)
    assert np.array_equal(np.argmax(LL, axis=1), pos)


def test_level_fit_finds_order_and_spacing():
    rng = np.random.default_rng(4)
    v = 0.05 + 0.5 * 0.4 * (2 * rng.integers(0, 4, 300) - 3) + 0.01 * rng.standard_normal(300)
    fits = {M: fit_levels(v, M) for M in (2, 4, 8)}
    assert max(fits, key=lambda m: fits[m].loglik) == 4
    assert abs(fits[4].spacing - 0.4) < 0.01 and abs(fits[4].center - 0.05) < 0.01


def test_continuity_detector():
    for cont in (True, False):
        x, t = generate_fsk(FSKSpec(2, 1.0, continuous_phase=cont, impairments=FSKImpairments(snr_db=10, seed=5)))
        c = phase_continuity(inst_freq(x), np.array([0.0]), t["samples_per_symbol"], 0.05)
        assert c.continuous == cont


def test_family_classifier_separates_fsk_and_psk():
    xf, _ = generate_fsk(FSKSpec(2, 0.5, impairments=FSKImpairments(snr_db=15, seed=6)))
    xp, _ = generate(SignalSpec("QPSK", payload_text="HELLO", impairments=Impairments(esn0_db=25, seed=6)))
    for x, fam in ((xf, "FSK"), (xp, "LINEAR")):
        x = x / np.sqrt(np.mean(np.abs(x) ** 2))
        xb = x[np.abs(x) > 0.05 * np.abs(x).max()]
        assert classify_family(xb, occupied_band(xb)).family == fam


# --------------------------------------------------------------------------- end to end

CASES = [
    ("4FSK", FSKSpec(4, 1.0, payload_text="FSK TEST", impairments=FSKImpairments(cfo_hz=400, timing_offset=0.3, seed=7))),
    ("MSK", FSKSpec(2, 0.5, payload_text="FSK TEST", impairments=FSKImpairments(cfo_hz=-600, timing_offset=0.6, seed=8))),
    ("GMSK", FSKSpec(2, 0.5, pulse="gaussian", bt=0.3, payload_text="FSK TEST",
                     impairments=FSKImpairments(cfo_hz=250, timing_offset=0.1, seed=9))),
    ("2FSK", FSKSpec(2, 1.0, continuous_phase=False, payload_text="FSK TEST",
                     impairments=FSKImpairments(cfo_hz=300, seed=10))),
    ("4FSK", FSKSpec(4, 0.5, mapping="natural", sample_rate=73_000.0, payload_text="FSK TEST",
                     impairments=FSKImpairments(cfo_hz=100, dc_offset=0.1j, iq_gain_db=1.0, iq_phase_deg=4, seed=11))),
]


@pytest.mark.parametrize("variant,spec", CASES, ids=[c[0] + str(i) for i, c in enumerate(CASES)])
def test_blind_roundtrip(variant, spec):
    x, truth = generate_fsk(spec)
    r = receive_fsk(x, spec.sample_rate)
    assert r.variant == variant
    assert r.primary.crc_ok and r.payload_text == spec.payload_text
    assert abs(r.model.h - spec.h) < 0.05 * spec.h
    assert abs(r.parameters["symbol_rate"].value - truth["symbol_rate_hz"]) < 0.002 * truth["symbol_rate_hz"]
    assert abs(r.parameters["cfo"].value - spec.impairments.cfo_hz) < 0.02 * spec.symbol_rate
    assert np.array_equal((r.llrs < 0).astype(np.uint8), r.hard_bits)


def test_legacy_all_modulations_bfsk_blind():
    path = os.path.join(ROOT, "Test-Data", "all_modulations_output", "BFSK.iq")
    if not os.path.exists(path):
        pytest.skip("legacy file not present")
    from synthetic.iq_io import read_iq
    f = read_iq(path)
    r = receive_fsk(f.samples, f.sample_rate)
    meta = json.load(open(path.replace(".iq", "_metadata.json")))
    assert r.variant == "MSK" and meta["input_text"] in r.payload_text
