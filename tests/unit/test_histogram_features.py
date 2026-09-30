"""
Tests for the phase / amplitude histogram features (dsp/histogram_features.py).

Linear-modulation signals are symbol-rate constellation points (dsp.constellations)
with light AWGN, i.e. what the histograms see after matched filtering and timing.
"""

import numpy as np
import pytest

from dsp.constellations import psk, qam
from dsp.histogram_features import (amplitude_histogram, count_amplitude_levels, count_phase_clusters,
                                    histogram_evidence, histogram_features, phase_histogram)

RNG_SEED = 26147


def _symbols(const, n, snr_db=30.0, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    s = const.points[rng.integers(0, const.order, n)]
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)
    return s + sigma * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _fsk2(n_sym=500, rs=1000.0, sps=8, dev=600.0, seed=RNG_SEED):
    """Continuous-phase 2FSK, tones at +-dev Hz."""
    rng = np.random.default_rng(seed)
    fs = rs * sps
    f = np.repeat(np.where(rng.integers(0, 2, n_sym) == 1, dev, -dev), sps)
    return np.exp(2j * np.pi * np.cumsum(f) / fs), fs


def test_if_histogram_shape():
    # the IF histogram is the classifier's own (signal_features.if_histogram_peaks), renormalised
    x, fs = _fsk2()
    hf = histogram_features(x, fs)
    assert hf["if_hist"].shape == (64,)
    assert hf["if_bins"].shape == (64,)
    assert hf["if_hist"].sum() == pytest.approx(1.0, abs=1e-9)
    # the two tones are the two largest modes, at about -600 and +600 Hz
    top = hf["if_bins"][np.argsort(hf["if_hist"])[-2:]]
    assert sorted(np.sign(top)) == [-1, 1]
    assert np.all(np.abs(np.abs(top) - 600.0) < 100.0)


def test_phase_histogram_bpsk():
    x = _symbols(psk(2), 1000)
    h, c = phase_histogram(x)
    assert h.shape == c.shape == (64,)
    assert h.sum() == pytest.approx(1.0)
    assert count_phase_clusters(x) == 2


def test_phase_histogram_qpsk():
    assert count_phase_clusters(_symbols(psk(4), 1000)) == 4


def test_phase_histogram_8psk():
    assert count_phase_clusters(_symbols(psk(8), 2000)) == 8


def test_amplitude_histogram_psk():
    x = _symbols(psk(4), 1000)
    h, c = amplitude_histogram(x)
    assert h.shape == c.shape == (128,)
    assert h.sum() == pytest.approx(1.0)
    assert count_amplitude_levels(x) == 1


def test_amplitude_histogram_16qam():
    x = _symbols(qam(16), 2000, snr_db=35.0)
    assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, abs=0.05)     # unit average power
    assert count_amplitude_levels(x) >= 3


def test_amplitude_histogram_64qam():
    assert count_amplitude_levels(_symbols(qam(64), 8000, snr_db=35.0)) >= 6


def test_histogram_evidence_schema():
    for x, fs in ((_symbols(psk(4), 1000), 1000.0), (_symbols(qam(16), 2000, 35.0), 1000.0), _fsk2()):
        ev = histogram_evidence(x, fs)
        assert isinstance(ev, list) and len(ev) == 2
        assert [e["pillar"] for e in ev] == ["FAM_PHASE_HIST", "FAM_AMP_HIST"]
        for e in ev:
            assert {"pillar", "description", "score", "details"} <= set(e)
            assert isinstance(e["description"], str) and e["description"]
            assert isinstance(e["details"], dict)
            assert 0.0 <= e["score"] <= 1.0
    qpsk = histogram_evidence(_symbols(psk(4), 1000), 1000.0)
    assert qpsk[0]["score"] > 0.5 and qpsk[1]["score"] > 0.5         # 4 clusters, constant envelope


def test_short_burst_handling():
    x = _symbols(psk(4), 8)
    hf = histogram_features(x, 1000.0)
    assert isinstance(hf["phase_cluster_count"], int) and hf["phase_cluster_count"] >= 0
    assert isinstance(hf["amplitude_level_count"], int) and hf["amplitude_level_count"] >= 0
    assert len(histogram_evidence(x, 1000.0)) == 2
    histogram_features(x[:1], 1000.0)                                  # a single sample: no IF at all


def test_nan_rejection():
    x = _symbols(psk(4), 100)
    x[10] = np.nan
    with pytest.raises(ValueError):
        histogram_features(x, 1000.0)
    with pytest.raises(ValueError):
        count_phase_clusters(x)
    with pytest.raises(ValueError):
        phase_histogram(np.array([], dtype=complex))


# ---------------------------------------------------------------------------
# Post-demodulation histogram check (postdemod/analyzer.py): symbols vs the receiver's modulation
# ---------------------------------------------------------------------------

from postdemod.analyzer import BitHypothesis, analyze, histogram_check  # noqa: E402


@pytest.mark.parametrize("mod, const, n", [("QPSK", psk(4), 1000), ("8PSK", psk(8), 2000)])
def test_postdemod_psk_symbols_confirm_modulation(mod, const, n):
    hc = histogram_check(_symbols(const, n), mod)
    assert hc.phase_cluster_count == {"QPSK": 4, "8PSK": 8}[mod] and hc.amplitude_level_count == 1
    assert hc.terms == {"phase": 2.0, "amplitude": 1.5} and hc.p_consistent > 0.95


@pytest.mark.parametrize("mod, const, n", [("16QAM", qam(16), 2000), ("64QAM", qam(64), 8000)])
def test_postdemod_qam_symbols_confirm_modulation(mod, const, n):
    hc = histogram_check(_symbols(const, n, snr_db=35.0), mod)
    assert hc.amplitude_level_count >= (3 if mod == "16QAM" else 6)
    assert hc.terms == {"phase": 2.0, "amplitude": 1.5} and hc.log_odds == pytest.approx(3.5)


def test_postdemod_weights_and_disagreement():
    # 16QAM symbols demodulated as BPSK: both post-demod pillars disagree at full (higher) weight
    hc = histogram_check(_symbols(qam(16), 2000, snr_db=35.0), "BPSK")
    assert hc.terms == {"phase": -2.0, "amplitude": -1.5} and hc.p_consistent < 0.05
    assert [e.source for e in hc.evidence] == ["FAM_PHASE_HIST_POSTDEMOD", "FAM_AMP_HIST_POSTDEMOD"]
    assert all(e.score == 0.0 and "supersedes" in e.description for e in hc.evidence)
    # 16PSK: 64 phase bins cannot resolve 16 phases, so the phase pillar is neutral
    assert histogram_check(_symbols(psk(16), 4000), "16PSK").terms["phase"] == 0.0
    # uneven symbol use (short ASCII payloads): 8PSK with one phase hit once is still on the lattice
    y = _symbols(psk(8), 94)
    far = np.abs(np.angle(y * np.exp(1j * np.pi / 4))) > 0.2          # drop all but one -45 deg symbol
    y = np.r_[y[far], y[~far][:1]]
    assert histogram_check(y, "8PSK").terms["phase"] >= 0.0
    # 8PSK symbols demodulated as QPSK: spread over more phases than the 4-fold lattice
    assert histogram_check(_symbols(psk(8), 2000), "QPSK").terms["phase"] == -2.0
    # BPSK symbols demodulated as QPSK: only the 2-fold sub-lattice is used -> lower order
    assert histogram_check(_symbols(psk(2), 1000), "QPSK").terms["phase"] == -2.0
    # 64QAM symbols demodulated as QPSK: QAM phases fit a 4-fold lattice only statistically, and
    # the amplitude contradicts the family, so phase cannot confirm the order (family guard)
    hq = histogram_check(_symbols(qam(64), 8000, snr_db=35.0), "QPSK")
    assert hq.terms == {"phase": 0.0, "amplitude": -1.5} and hq.log_odds < 0
    # impaired PSK envelope (3+ amplitude levels) with locked phases: PSK-grade lattice keeps the phase vote
    rng = np.random.default_rng(RNG_SEED)
    y = _symbols(psk(4), 1000) * rng.choice([0.6, 1.0, 1.4], 1000)
    hi = histogram_check(y, "QPSK")
    assert hi.terms == {"phase": 2.0, "amplitude": -1.5} and hi.log_odds > 0


def test_postdemod_fallback_and_analyze_integration():
    x = _symbols(psk(4), 1000)
    assert histogram_check(None, "QPSK") is None
    assert histogram_check(x[:10], "QPSK") is None                       # too few symbols
    assert histogram_check(x, "2FSK") is None                            # not PSK/QAM
    bad = x.copy()
    bad[3] = np.nan
    assert histogram_check(bad, "QPSK") is None                          # non-finite: no check, no exception
    bits = np.random.default_rng(RNG_SEED).integers(0, 2, 512).astype(np.uint8)
    hyps = [BitHypothesis("receiver output", bits, 1.0 - 2.0 * bits)]
    assert analyze(hyps, budget_s=5.0).histogram_check is None           # no symbols: behaviour unchanged
    assert analyze(hyps, budget_s=5.0, symbols=x, modulation="QPSK").histogram_check.log_odds == pytest.approx(3.5)
