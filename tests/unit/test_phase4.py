"""Phase 4 unit tests: CRC catalogue, FEC codecs, GF(2) rank, interleavers, blind estimation,
sync search, frame analysis, the joint post-demodulation search and the report."""

import numpy as np
import pytest

from bitstream import sync_search
from deinterleaving import block as blk
from deinterleaving import convolutional as cvi
from deinterleaving.gf2 import random_rank_pmf, rank
from fec import block_codes, convolutional, reed_solomon
from fec.identify import blind_conv_params, classify_family, structure_scan
from postdemod.analyzer import BitHypothesis, analyze, data_span
from protocol import crc
from protocol.frames import analyze as analyze_frames
from synthetic.channel_coding import CodingSpec, encode_frame
from synthetic.framing import SYNC_BITS, build_frame, bytes_to_bits

TEXT = "Frame #7: T=23.5C {ok} [a-z|A-Z] ~@_^`\\ 0123456789"
rng = np.random.default_rng(42)


def test_crc_catalogue_check_values():
    assert crc.self_test()


@pytest.mark.parametrize("code", convolutional.LIBRARY, ids=lambda c: c.name)
def test_convolutional_roundtrip_with_noise(code):
    u = rng.integers(0, 2, 600).astype(np.uint8)
    x = convolutional.encode(u, code)
    llr = (1 - 2.0 * x) * 4 + np.random.default_rng(1).normal(0, 1.0, len(x))
    r = convolutional.viterbi(llr, code)
    assert np.array_equal(r.bits, u)
    assert r.reencode_mismatch < 0.01


def test_syndrome_separates_right_and_wrong_phase_and_inversion():
    code = convolutional.get_code("CC K=5 (23,35) r1/2")
    x = convolutional.encode(rng.integers(0, 2, 800).astype(np.uint8), code)
    assert convolutional.syndrome_weight(x, code, 0) == 0.0
    assert convolutional.syndrome_weight(x, code, 1) > 0.35
    sc = structure_scan(1 - x)                       # complement: even-weight generator flips the checks
    assert sc.conv_code.name == code.name and sc.conv_inverted


@pytest.mark.parametrize("prof", reed_solomon.PROFILES, ids=lambda p: p.name)
def test_reed_solomon_blind_search(prof):
    u = rng.integers(0, 2, 8 * 300).astype(np.uint8)
    x = reed_solomon.encode(u, prof)
    y = x.copy()
    y[np.random.default_rng(3).choice(len(y), prof.t, replace=False)] ^= 1
    y = np.concatenate([np.ones(3, np.uint8), y, np.zeros(5, np.uint8)])
    best_p, d = reed_solomon.search(y)[0]
    assert best_p.nsym == prof.nsym and d.ok and np.array_equal(d.data_bits[:len(u)], u)


@pytest.mark.parametrize("code", block_codes.LIBRARY, ids=lambda c: c.name)
def test_hamming_single_error_correction(code):
    u = rng.integers(0, 2, 440).astype(np.uint8)
    x = block_codes.encode(u, code)
    x[::code.n * 3] ^= 1
    d, corrected, bad = block_codes.decode(x, code)
    assert np.array_equal(d[:len(u)], u) and bad == 0 and corrected > 0


def test_gf2_rank_and_random_law():
    M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    assert rank(M) == 2
    assert rank(np.eye(70, dtype=np.uint8)) == 70
    assert abs(random_rank_pmf(20, 20)[-1] - 0.2888) < 1e-3


def test_interleavers_roundtrip():
    u = rng.integers(0, 2, 1000).astype(np.uint8)
    assert np.array_equal(blk.deinterleave(blk.interleave(u, 7, 9), 7, 9)[:1000], u)
    assert np.array_equal(cvi.deinterleave(cvi.interleave(u, 5, 3), 5, 3)[:1000], u)
    llr = 1.0 - 2.0 * u                              # LLRs keep their dtype through the Forney de-interleaver
    assert np.array_equal(cvi.deinterleave(cvi.interleave(llr, 5, 3), 5, 3)[:1000], llr)


def test_burel_period_and_offset():
    h = block_codes.encode(rng.integers(0, 2, 3000).astype(np.uint8), block_codes.LIBRARY[0])
    s = np.concatenate([rng.integers(0, 2, 11).astype(np.uint8), blk.interleave(h, 4, 7)])
    pe = blk.estimate_period(s, 64)
    assert pe.period == 28 and pe.offset == 11
    assert blk.estimate_period(rng.integers(0, 2, 6000).astype(np.uint8), 64).period is None


@pytest.mark.parametrize("name,rate,K", [("CC K=7 (171,133) r1/2 [NASA/CCSDS]", (1, 2), 7),
                                         ("CC K=5 (23,35) r1/2", (1, 2), 5),
                                         ("CC K=7 (133,171,165) r1/3 [LTE-style]", (1, 3), 7)])
def test_blind_convolutional_parameters(name, rate, K):
    x = convolutional.encode(rng.integers(0, 2, 1500).astype(np.uint8), convolutional.get_code(name))
    e = blind_conv_params(blk.rank_profile(x, 64))
    assert (e.k, e.n, e.K) == (*rate, K)


def test_family_classification_random_is_uncoded():
    assert classify_family(structure_scan(rng.integers(0, 2, 3000).astype(np.uint8))).family == "Uncoded"


def test_sync_search_occurrences_and_inversion():
    f, _ = build_frame(b"HELLO WORLD", 1)
    bits = np.concatenate([rng.integers(0, 2, 77).astype(np.uint8), f, f, rng.integers(0, 2, 40).astype(np.uint8)])
    det = sync_search.best_detection(sync_search.search(bits))
    assert det.name.startswith("SIH") and det.occurrences == [77, 77 + len(f)] and det.period == len(f)
    inv = sync_search.best_detection(sync_search.search(1 - bits))
    assert inv.inverted


def test_frame_layout_discovery_ccsds_style():
    asm = sync_search.SYNC_LIBRARY["CCSDS ASM 0x1ACFFC1D"]
    p = b"Telemetry 123! {x}"
    c = crc.compute(crc.BY_NAME["CRC-32/ISO-HDLC"], p).to_bytes(4, "little")
    bits = np.concatenate([rng.integers(0, 2, 13).astype(np.uint8), asm, bytes_to_bits(bytes([len(p)]) + p + c),
                           rng.integers(0, 2, 20).astype(np.uint8)])
    fa = analyze_frames(bits, sync_search.best_detection(sync_search.search(bits)))
    assert fa.valid_frames == 1 and fa.frames[0].payload == p
    assert fa.layout.length_field == "u8" and fa.layout.crc == "CRC-32/ISO-HDLC" and fa.layout.crc_endian == "le"


def test_data_span_trims_guard():
    llr = np.r_[np.full(8, 2.0), np.full(400, 60.0), np.full(6, 3.0)]
    assert data_span(llr) == (8, 408)


@pytest.mark.parametrize("spec", [
    CodingSpec(None, None),
    CodingSpec("CC K=7 (171,133) r1/2 [NASA/CCSDS]", None),
    CodingSpec("CC K=7 (171,133) r3/4 punct [DVB/CCSDS]", None, sync_outside=True),
    CodingSpec("RS(255,223) [CCSDS-size]", None),
    CodingSpec("Hamming(15,11)", "block:8x15"),
    CodingSpec("CC K=5 (23,35) r1/2", "conv:6,4"),
], ids=lambda s: s.describe())
def test_joint_search_bit_level(spec):
    bits, _ = encode_frame(TEXT.encode(), spec, 2)
    llr = (1 - 2.0 * bits) * 3.0 + np.random.default_rng(5).normal(0, 1.0, len(bits))
    hard = (llr < 0).astype(np.uint8)
    r = analyze([BitHypothesis("h0", hard, llr), BitHypothesis("inverted", 1 - hard, -llr)])
    assert r.verified and r.payload_text == TEXT
    assert r.best.fec_name == (spec.fec or "none")
    if spec.interleaver:
        assert r.best.deint.kind == spec.interleaver.split(":")[0]


def test_report_renders_without_gui_objects():
    from reporting.report import build_report, to_html, to_markdown, to_json
    bits, _ = encode_frame(TEXT.encode(), CodingSpec("CC K=3 (7,5) r1/2", None), 1)
    post = analyze([BitHypothesis("h0", bits, 1 - 2.0 * bits)], budget_s=30)
    rep = build_report({"postdemod": post, "payload_text": post.payload_text, "crc_valid": post.verified,
                        "config": {"file_path": "x.iq", "dtype": "cf32"}})
    assert rep["result"]["status"].startswith("VERIFIED") and rep["fec"]["decision"] == "CC K=3 (7,5) r1/2"
    assert TEXT in to_html(rep).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace(
        "&quot;", '"').replace("&#x27;", "'") and "## FEC (L10)" in to_markdown(rep) and to_json(rep)


# ---------------------------------------------------------------------------
# Diagonal / pseudo-random interleavers, LDPC, concatenated codes, blind sample rate
# ---------------------------------------------------------------------------

from deinterleaving import diagonal as dgi  # noqa: E402
from deinterleaving import pseudorandom as pri  # noqa: E402
from fec import concatenated, ldpc  # noqa: E402
from fec.dual import recover_parity_checks  # noqa: E402


@pytest.mark.parametrize("R,C,d", [(3, 4, 1), (16, 24, 1), (12, 20, -1), (7, 7, -1)])
def test_diagonal_is_a_permutation_and_roundtrips(R, C, d):
    p = dgi.permutation(R, C, d)
    assert sorted(p.tolist()) == list(range(R * C))
    u = rng.integers(0, 2, 3 * R * C + 5).astype(np.uint8)
    assert np.array_equal(dgi.deinterleave(dgi.interleave(u, R, C, d), R, C, d)[:len(u)], u)


def test_quoted_diagonal_formula_is_not_a_permutation():
    # pi(k) = ((k mod R)*C + (k mod R) + floor(k/R)) mod (R*C) collides for every R, C >= 2
    for R in range(2, 12):
        for C in range(2, 12):
            k = np.arange(R * C)
            assert len(set((((k % R) * C + (k % R) + k // R) % (R * C)).tolist())) < R * C


def test_lfsr_taps_are_maximal_and_pr_roundtrips():
    assert all(len(v) == 1 for v in pri.PRIMITIVE_TAPS.values())
    h = pri.hypotheses(1020)[0]
    perm = h.perm()
    assert sorted(perm.tolist()) == list(range(1020))
    u = rng.integers(0, 2, 2040).astype(np.uint8)
    assert np.array_equal(pri.deinterleave(pri.interleave(u, perm), perm), u)
    assert len(pri.qpp_permutation(40, 3, 10)) == 40


@pytest.mark.parametrize("code", ldpc.LIBRARY, ids=lambda c: c.name)
def test_ldpc_encode_align_decode(code):
    u = rng.integers(0, 2, code.k * 4).astype(np.uint8)
    x = ldpc.encode(u, code)
    assert not ((code.H.astype(int) @ x.reshape(-1, code.n).T) % 2).any()
    sig = 0.5                                      # ~4.3 dB Eb/N0 at rate 3/4: above the waterfall
    y = (1 - 2.0 * x) + np.random.default_rng(3).normal(0, sig, len(x))
    llr = np.r_[np.zeros(9), 2 * y / sig ** 2]
    o, f, z = ldpc.align(np.r_[np.zeros(9, np.uint8), (y < 0).astype(np.uint8)], code)
    assert o == 9 and z > 5
    d = ldpc.decode(llr, code, o)
    assert d.converged == d.codewords and np.array_equal(d.bits, u)


def test_blind_parity_check_recovery():
    code = ldpc.regular_gallager("unknown", 96, 3, 6, 99)
    x = ldpc.encode(rng.integers(0, 2, code.k * 130).astype(np.uint8), code).reshape(-1, 96)
    rec = recover_parity_checks(x, w_max=8)
    assert rec["complete"] and set(rec["row_weights"]) == {6}


@pytest.mark.parametrize("name", list(concatenated.PROFILES))
def test_concatenated_roundtrip_and_blind_outer(name):
    spec = concatenated.PROFILES[name]
    u = rng.integers(0, 2, 8 * 300).astype(np.uint8)
    x = concatenated.encode(u, spec)
    y = (1 - 2.0 * x) + np.random.default_rng(4).normal(0, 0.7, len(x))
    d = concatenated.decode(2 * y / 0.49, spec)
    assert d.ok and np.array_equal(d.data_bits[:len(u)], u)
    vb = convolutional.viterbi(1.0 - 2.0 * x, convolutional.get_code(spec.inner)).bits
    best = concatenated.search_outer(vb, spec.inner)[0][0]
    assert best.rs_name == spec.rs_name and best.outer == spec.outer


def test_concatenated_never_claims_a_wrong_decode():
    spec = concatenated.PROFILES["CCSDS structure (I=1)"]
    u = rng.integers(0, 2, 8 * 400).astype(np.uint8)
    x = concatenated.encode(u, spec)
    for seed in range(3):
        y = (1 - 2.0 * x) + np.random.default_rng(seed).normal(0, 1.0, len(x))
        d = concatenated.decode(2 * y, spec)
        if d.ok:
            assert np.array_equal(d.data_bits[:len(u)], u)


def test_blind_sample_rate_hypotheses():
    from estimation.blind_sample_rate import estimate
    from synthetic.generator import Impairments, SignalSpec, generate
    x, truth = generate(SignalSpec("QPSK", symbol_rate=2400.0, sample_rate=19200.0,
                                   impairments=Impairments(esn0_db=20.0, seed=3)))
    e = estimate(x)
    assert abs(e.sps - 8.0) < 0.1
    assert any(h.sample_rate == 19200.0 and h.symbol_rate == 2400.0 for h in e.hypotheses)
