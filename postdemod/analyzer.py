"""
Post-demodulation analysis (L8 bitstream -> L9 de-interleaving -> L10 FEC -> L11 framing).

The layers are solved jointly, as the reference papers recommend (the FEC structure reveals
the interleaver, the right de-interleaver reveals the FEC, and only a sync word + CRC proves
the chain):

    for every demodulator bit hypothesis (carrier-phase rotation / polarity):
      A  raw stream: library sync word (Chiani LRT/GLRT metrics) -> frame layout -> CRC
         (sync found but CRC fails: the body after the sync is a coded-block candidate)
      B  code structure on the stream: convolutional parity checks, Hamming syndromes,
         Reed-Solomon decoding, punctured-code Viterbi -> decode -> sync -> CRC
      C  de-interleaving hypotheses (only if A/B found nothing that validates):
           block R x C: Burel-Gautier period/offset estimate (when N > P^2) and burst-length
                        hypotheses (a burst is a whole number of blocks); K-S score as evidence
           convolutional B, M (Forney)
         each screened by the code-structure tests, the best ones decoded as in B
    candidates are ranked: CRC-valid frames, then structure strength, then simplicity.

Nothing is reported as decoded unless a CRC validates; otherwise the best hypothesis is
reported with its evidence and marked unverified.

Post-demodulation histogram check (PSK/QAM only): the phase / amplitude histograms of the
receiver's 1-sps symbols are compared with the modulation the receiver used (phase: M-fold
lattice fit + cluster count; amplitude: ring count). On symbols these are reliable (unlike the
raw-burst FAM_PHASE_HIST / FAM_AMP_HIST prior of the classifier, which they supersede), so they
get larger weights: +-2.0 (phase) and +-1.5 (amplitude levels) of log-odds that the modulation
is consistent. Checked, not re-decided: a disagreement only lowers the confidence; the
receiver is not re-run.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from bitstream import sync_search
from bitstream.correlation import BitstreamCorrelation
from bitstream.entropy import BitstreamEntropy
from core.evidence import Evidence
from deinterleaving import block as blk
from deinterleaving import convolutional as cvi
from deinterleaving import diagonal as dgi
from deinterleaving import pseudorandom as pri
from dsp.histogram_features import histogram_evidence
from fec import block_codes, concatenated, convolutional, ldpc, reed_solomon
from fec.dual import recover_parity_checks
from fec.identify import (ConvBlindEstimate, FamilyDecision, StructureScan, blind_conv_params, classify_family,
                          structure_scan)
from protocol.frames import FrameAnalysis, analyze as analyze_frames

STRONG = 0.9                      # structure strength that stops the de-interleaver screening
LDPC_Z_MIN = 5.0                  # codeword-alignment significance (random data: z <= ~3.5); CRC decides
NO_STRUCTURE = StructureScan(None, 0, 0.5, None, 0, 1.0)     # "run only the RS decoder"


@dataclass
class BitHypothesis:
    name: str
    bits: np.ndarray
    llrs: np.ndarray
    span: Optional[Tuple[int, int]] = None     # [start, end) of real data bits (guard symbols excluded)


@dataclass
class DeintCandidate:
    kind: str                     # "none" | "block" | "conv"
    params: str                   # "16 x 32 (write rows, read columns)" / "B=8, M=3"
    offset: int
    strength: float
    detail: str
    validated: bool = False
    stream: str = ""              # demodulator bit hypothesis it was screened on

    def as_row(self):
        return [self.stream, self.kind, self.params, self.offset, round(self.strength, 3), self.detail,
                "CRC valid" if self.validated else ""]


@dataclass
class FECCandidate:
    code: str
    family: str
    setting: str                  # phase / offset / alignment
    metric: str                   # syndrome / re-encoding / corrected symbols
    result: str
    crc_frames: int = 0
    stream: str = ""

    def as_row(self):
        return [self.stream, self.family, self.code, self.setting, self.metric, self.result, self.crc_frames]


@dataclass
class ChainResult:
    hypothesis: str
    deint: DeintCandidate
    fec_name: str                 # "none" or code name
    fec_family: str
    fec_detail: str
    decoded_bits: np.ndarray
    decoded_llrs: Optional[np.ndarray]
    frames: FrameAnalysis
    strength: float
    evidence: List[Evidence] = field(default_factory=list)

    @property
    def crc_frames(self) -> int:
        return self.frames.valid_frames

    def key(self):
        simple = (self.deint.kind == "none") + (self.fec_name == "none")
        return (self.crc_frames, len(self.frames.frames) > 0, round(self.strength, 2), simple)


@dataclass
class BitstreamReport:
    hypothesis: str
    n_bits: int
    randomness: Dict[str, float]
    periodicity: List[Tuple[int, float]]
    autocorr: np.ndarray
    sync: List[sync_search.SyncCandidate]
    rank_profile: Dict[int, float]
    blind_conv: Optional[ConvBlindEstimate]
    frame_length: Optional[int]
    preview_hex: str


@dataclass
class PostDemodResult:
    bitstream: BitstreamReport
    family: FamilyDecision
    scan: StructureScan
    deint_candidates: List[DeintCandidate]
    deint_best: DeintCandidate
    period_estimate: Optional[blk.PeriodEstimate]
    fec_candidates: List[FECCandidate]
    best: Optional[ChainResult]
    hypotheses_tested: int
    seconds: float
    log: List[str]
    histogram_check: Optional["HistogramCheck"] = None

    @property
    def verified(self) -> bool:
        return bool(self.best and self.best.crc_frames > 0)

    @property
    def payload_text(self) -> Optional[str]:
        if not self.best or not self.best.frames.frames:
            return None
        return self.best.frames.payload.decode("ascii", errors="replace")


# ---------------------------------------------------------------------------
# Post-demodulation histogram check
# ---------------------------------------------------------------------------

PHASE_HIST_POSTDEMOD_W = 2.0      # vs the raw-burst prior's 1.0
AMP_HIST_POSTDEMOD_W = 1.5        # vs the raw-burst prior's 0.5
HIST_MIN_SYMBOLS = 16             # fewer symbols: no check at all
AMP_MIN_SYMBOLS = 32              # fewer: noise alone splits the envelope into 3-4 "levels" (16PSK, 22 symbols)
COVERAGE_PER_POINT = 8            # n >= 8 M: every M-PSK phase should appear (lower-order test)
# A lattice this concentrated is PSK, not QAM: QAM phases fit a 4-fold lattice only statistically
# (uniform 16/64/256QAM rho 0.36/0.20/0.16; real QAM <= 0.48, 0.69 on a 35-symbol 256QAM burst).
STRONG_LATTICE_RHO, STRONG_LATTICE_MIN_SYMBOLS = 0.8, 64
LATTICE_SIGNIFICANCE = 6.9        # Rayleigh test n * rho^2: uniform phases exceed it with p = e^-6.9 = 1e-3
# Phase lattice order of each modulation. Square QAM: the corner diagonals put its phases on a
# 4-fold lattice statistically. 16PSK is absent on purpose: 64 phase bins (sigma 2 bins) cannot
# resolve 16 phases, so the cluster count cannot confirm its order.
PHASE_LATTICE = {"BPSK": 2, "QPSK": 4, "4QAM": 4, "8PSK": 8, "16QAM": 4, "64QAM": 4, "256QAM": 4}


@dataclass
class HistogramCheck:
    modulation: str
    phase_cluster_count: int
    amplitude_level_count: int
    lattice_rho: Optional[float]  # |mean exp(j M phi)| for the modulation's lattice order M
    terms: Dict[str, float]       # "phase", "amplitude": log-odds that the modulation is consistent
    log_odds: float
    p_consistent: float
    evidence: List[Evidence]


def histogram_check(symbols: Optional[np.ndarray], modulation: Optional[str]) -> Optional[HistogramCheck]:
    """
    Do the receiver's symbols look like the modulation it demodulated with?

    Real payloads are short and use the points unevenly (ASCII text: an 8PSK burst of 94
    symbols hit one phase once), so an exact cluster count fails on correct demodulation.
    The phase term therefore asks occupancy-independent questions:
      lattice   do the phases sit on the M-fold lattice? Rayleigh test on rho = |mean exp(j M phi)|
                (M-PSK points map to one value whatever symbols the payload uses)
      order     histogram clusters in (M/2, M]: more than a lower order, no more than M
    phase      clusters > M                                   -> -2.0  (more phases than the constellation)
               PSK, clusters <= M/2, M/2-fold lattice significant, n >= 8 M
                                                              -> -2.0  (only a sub-lattice used: lower order)
               lattice significant and clusters in (M/2, M]   -> +2.0
               otherwise (low SNR / few symbols / 16PSK)      ->  0
    amplitude  PSK: 1 level -> +1.5, >= 3 -> -1.5;  QAM: >= 3 -> +1.5, 1 -> -1.5;  2 levels or
               fewer than 32 symbols: 0
    Family guard: QAM phases also fit a 4-fold lattice, so when the amplitude contradicts the
    family (PSK vs QAM) the phase term cannot confirm the order (capped at 0) -- unless the
    lattice is PSK-grade (rho >= 0.8 on >= 64 symbols; an impaired PSK envelope can split
    into 3-4 amplitude "levels" while its phases stay locked).
    None when there are no / too few symbols, the modulation is not PSK/QAM, or the symbols
    are not finite.
    """
    if symbols is None or modulation is None or len(symbols) < HIST_MIN_SYMBOLS:
        return None
    qam = modulation.endswith("QAM") and modulation != "4QAM"
    if not (qam or modulation.endswith("PSK") or modulation == "4QAM"):
        return None
    y = np.asarray(symbols)
    try:
        hev = {h["pillar"]: h for h in histogram_evidence(y, 1.0)}
    except ValueError:
        return None
    n = len(y)
    nc = hev["FAM_PHASE_HIST"]["details"]["phase_cluster_count"]
    na = hev["FAM_AMP_HIST"]["details"]["amplitude_level_count"]
    m = PHASE_LATTICE.get(modulation)
    phi = np.angle(y)
    rho = None if m is None else float(np.abs(np.mean(np.exp(1j * m * phi))))
    on_lattice = rho is not None and n * rho ** 2 >= LATTICE_SIGNIFICANCE
    sub_lattice = (m is not None and not qam and m >= 4 and nc <= m / 2 and n >= COVERAGE_PER_POINT * m
                   and n * np.abs(np.mean(np.exp(1j * (m // 2) * phi))) ** 2 >= LATTICE_SIGNIFICANCE)
    if m is None:
        t_phase = 0.0
    elif nc > m or sub_lattice:
        t_phase = -PHASE_HIST_POSTDEMOD_W
    elif on_lattice and m / 2 < nc:
        t_phase = PHASE_HIST_POSTDEMOD_W
    else:
        t_phase = 0.0
    if na == 2 or na == 0 or n < AMP_MIN_SYMBOLS:
        t_amp = 0.0
    else:
        t_amp = AMP_HIST_POSTDEMOD_W if (na >= 3) == qam else -AMP_HIST_POSTDEMOD_W
    strong = rho is not None and rho >= STRONG_LATTICE_RHO and n >= STRONG_LATTICE_MIN_SYMBOLS
    if t_amp < 0 and not strong:                    # family contradicted: phase cannot confirm the order
        t_phase = min(t_phase, 0.0)
    logit = t_phase + t_amp
    p = float(1.0 / (1.0 + np.exp(-logit)))

    def verdict(t):
        return "agrees" if t > 0 else ("disagrees" if t < 0 else "neutral")

    lat = ("no lattice test (16 phases not resolvable at 64 bins)" if m is None else
           f"{m}-fold lattice rho {rho:.2f} ({'significant' if on_lattice else 'not significant'}, "
           f"n rho^2 {n * rho ** 2:.1f}), clusters expected in ({m // 2}, {m}]"
           + ("; only the half-order sub-lattice is used" if sub_lattice else ""))
    ev = [Evidence(f"Post-demod phase histogram of {n} symbols: {nc} cluster(s); {lat} -> {verdict(t_phase)} "
                   f"(log-odds {t_phase:+.1f}; supersedes FAM_PHASE_HIST)",
                   "FAM_PHASE_HIST_POSTDEMOD", 1.0 if t_phase > 0 else (0.0 if t_phase < 0 else 0.5),
                   {"phase_cluster_count": nc, "lattice_order": m, "lattice_rho": rho, "on_lattice": on_lattice,
                    "log_odds": t_phase}),
          Evidence(f"Post-demod amplitude histogram: {na} level(s), {modulation} expects "
                   + (">= 3 rings" if qam else "1 (constant envelope)")
                   + f" -> {verdict(t_amp)} (log-odds {t_amp:+.1f}; supersedes FAM_AMP_HIST)",
                   "FAM_AMP_HIST_POSTDEMOD", 1.0 if t_amp > 0 else (0.0 if t_amp < 0 else 0.5),
                   {"amplitude_level_count": na, "log_odds": t_amp})]
    return HistogramCheck(modulation, nc, na, rho, {"phase": t_phase, "amplitude": t_amp}, logit, p, ev)


# ---------------------------------------------------------------------------
# Hypotheses from the demodulators
# ---------------------------------------------------------------------------

def hypotheses_from_receiver(family: str, r) -> List[BitHypothesis]:
    """Bit streams to try: the receiver's own resolution first, then every other carrier-phase
    rotation (PSK/QAM; the rotation is invisible without a sync word in clear) or the polarity
    flip (FSK)."""
    hyps = [BitHypothesis(f"receiver output ({getattr(r.decode, 'method', 'primary') if hasattr(r, 'decode') else 'primary'})",
                          np.asarray(r.hard_bits, dtype=np.uint8), np.asarray(r.llrs, dtype=np.float64))]
    if family in ("PSK", "QAM"):
        const0 = r.decode.constellation
        if const0 is None:
            from dsp.constellations import get_constellation
            const0 = get_constellation(r.modulation if r.modulation != "4QAM" else "QPSK")
        span = symbol_span(np.asarray(r.symbols), const0)
        hyps[0].span = span
        from demodulation.psk import rotate
        const = r.decode.constellation
        if const is None:
            from dsp.constellations import get_constellation
            const = get_constellation(r.modulation if r.modulation != "4QAM" else "QPSK")
        nv = 10 ** (-max(float(r.esn0_est_db), 0.0) / 10)
        sym = const.phase_symmetry
        for k in range(1, sym):
            yr = rotate(np.asarray(r.symbols), k, sym)
            hyps.append(BitHypothesis(f"carrier rotation +{360 * k / sym:g} deg", const.demap_hard(yr).astype(np.uint8),
                                      const.demap_soft(yr, nv).astype(np.float64), span))
    else:
        hyps.append(BitHypothesis("polarity inverted", 1 - hyps[0].bits, -hyps[0].llrs))
    return hyps


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

MAX_HYPOTHESES = 12000            # de-interleaver screenings: deterministic search depth


class _Budget:
    """Search budget: a fixed number of hypothesis screenings (so results do not depend on machine
    load or numpy threading) plus a wall-clock safety limit."""
    def __init__(self, seconds: float, max_hyps: int = MAX_HYPOTHESES):
        self.t0 = time.time()
        self.limit = seconds
        self.max_hyps = max_hyps
        self.used = 0

    def exhausted(self) -> bool:
        return self.used > self.max_hyps or time.time() - self.t0 > self.limit


def _frames_for(bits: np.ndarray, llrs: Optional[np.ndarray], prefix: Optional[np.ndarray] = None) -> FrameAnalysis:
    """Sync search + frame/CRC analysis. prefix: sync bits found in clear before a coded body
    (CADU style) are put back in front of the decoded body."""
    if prefix is not None and len(prefix):
        bits = np.concatenate([prefix, bits])
        llrs = None
    cands = sync_search.search(bits, llrs)
    det = sync_search.best_detection(cands)
    word = sync_search.SYNC_LIBRARY.get(det.name) if det else None
    return analyze_frames(bits, det, word)


def _decode_with(bits, llrs, scan: StructureScan, start_known: bool, fec_rows: List[FECCandidate],
                 hyp_name: str, deint: DeintCandidate, allow_rs: bool = True,
                 allow_punctured: bool = False, prefix: Optional[np.ndarray] = None) -> List[ChainResult]:
    """Stage B: decode with the codes the structure tests point at, then frame + CRC. A strong
    parity-check match makes the (slower) RS and punctured-code searches unnecessary unless the
    decode that follows fails to validate."""
    n_rows = len(fec_rows)
    out = _decode_inner(bits, llrs, scan, start_known, fec_rows, hyp_name, deint, allow_rs, allow_punctured, prefix)
    for row in fec_rows[n_rows:]:
        row.stream = hyp_name if deint.kind == "none" else f"{hyp_name} | {deint.kind} {deint.params}"
    return out


def _decode_inner(bits, llrs, scan, start_known, fec_rows, hyp_name, deint, allow_rs, allow_punctured, prefix):
    out = []
    # convolutional (unpunctured, from the syndrome test)
    if scan.conv_strength > 0:
        code, off = scan.conv_code, scan.conv_offset
        cl = -np.asarray(llrs) if scan.conv_inverted else np.asarray(llrs)
        vr = convolutional.viterbi(cl[off:], code, terminated=False, start_known=start_known and off == 0)
        fa = _frames_for(vr.bits, None, prefix)
        fec_rows.append(FECCandidate(code.name, "Convolutional", f"phase {off}"
                                     + (", stream inverted" if scan.conv_inverted else ""),
                                     f"syndrome {scan.conv_syndrome:.3f}, re-encode mismatch {vr.reencode_mismatch:.3f}",
                                     f"Viterbi {len(vr.bits)} bits", fa.valid_frames))
        out.append(ChainResult(hyp_name, deint, code.name, "Convolutional", code.describe(), vr.bits, None, fa,
                               scan.conv_strength, [Evidence(
                                   f"{code.name}: parity-check syndrome weight {scan.conv_syndrome:.3f} "
                                   f"({scan.conv_z:.0f} sigma below random), soft Viterbi re-encoding mismatch "
                                   f"{vr.reencode_mismatch:.3%} (~ channel BER), normalised path metric "
                                   f"{vr.path_metric:.2f}", "VITERBI", 1 - 2 * vr.reencode_mismatch)]))
        if not fa.valid_frames:
            out.extend(_concatenated(vr.bits, cl[off:], code, fec_rows, hyp_name, deint, prefix))
    # Hamming
    if scan.block_strength > 0:
        code, off = scan.block_code, scan.block_offset
        data, corr, bad = block_codes.decode(bits, code, off)
        fa = _frames_for(data, None, prefix)
        fec_rows.append(FECCandidate(code.name, "Block (Hamming)", f"offset {off}",
                                     f"syndrome failures {scan.block_fail:.3f}", f"{corr} corrected, {bad} uncorrectable",
                                     fa.valid_frames))
        out.append(ChainResult(hyp_name, deint, code.name, "Block code", code.describe(), data, None, fa,
                               scan.block_strength, [Evidence(
                                   f"{code.name} at offset {off}: syndrome failures {scan.block_fail:.3f} "
                                   f"({scan.block_z:.0f} sigma below random); {corr} single errors corrected, "
                                   f"{bad} uncorrectable blocks", "SYNDROME", scan.block_strength)]))
    if any(c.crc_frames for c in out):
        return out
    # LDPC (known parity-check matrices): codeword alignment by syndrome, then min-sum decoding
    out.extend(_ldpc(bits, llrs, fec_rows, hyp_name, deint, prefix))
    if any(c.crc_frames for c in out):
        return out
    # Reed-Solomon (systematic, known profiles, 8 alignments x MSB/LSB)
    if allow_rs:
        res = reed_solomon.search(bits)
        if not (res and res[0][1].score > 0):
            res = reed_solomon.search(1 - np.asarray(bits))     # 180 deg / polarity: RS is not closed
        if res and res[0][1].score > 0:
            prof, d = res[0]
            fa = _frames_for(d.data_bits, None, prefix)
            fec_rows.append(FECCandidate(prof.name, "Reed-Solomon", f"bit offset {d.bit_offset}, "
                                         f"{'LSB' if d.lsb_first else 'MSB'}-first",
                                         f"{d.blocks - d.failed_blocks}/{d.blocks} blocks decoded",
                                         f"{d.corrected_symbols} symbols corrected", fa.valid_frames))
            out.append(ChainResult(hyp_name, deint, prof.name, "Reed-Solomon", prof.describe(), d.data_bits, None, fa,
                                   d.score, [Evidence(
                                       f"{prof.name}: {d.blocks - d.failed_blocks}/{d.blocks} codewords decoded at bit "
                                       f"offset {d.bit_offset} ({'LSB' if d.lsb_first else 'MSB'}-first), "
                                       f"{d.corrected_symbols} symbols corrected, last block {d.last_block_len} bytes "
                                       f"(shortened)", "RS_DECODER", d.score)]))
    # punctured convolutional codes: no simple syndrome -> Viterbi re-encoding mismatch
    if allow_punctured and not any(c.crc_frames for c in out):
        for code in [c for c in convolutional.LIBRARY if c.puncture is not None]:
            vr, ph, off = convolutional.decode_unknown_phase(llrs, code, terminated=False)
            base = convolutional.random_mismatch_baseline(code)
            strength = float(np.clip(1 - vr.reencode_mismatch / max(base, 1e-6), 0, 1))
            if strength < 0.5:                       # not clearly better than random data
                continue
            fa = _frames_for(vr.bits, None, prefix)
            fec_rows.append(FECCandidate(code.name, "Convolutional (punctured)", f"bit offset {off}, puncture phase {ph}",
                                         f"re-encode mismatch {vr.reencode_mismatch:.3f} (random {base:.3f})",
                                         f"Viterbi {len(vr.bits)} bits",
                                         fa.valid_frames))
            out.append(ChainResult(hyp_name, deint, code.name, "Convolutional", code.describe(), vr.bits, None, fa,
                                   strength, [Evidence(
                                       f"{code.name}: Viterbi re-encoding mismatch {vr.reencode_mismatch:.3%} at bit "
                                       f"offset {off}, puncturing phase {ph} (random data gives {base:.1%})",
                                       "VITERBI", strength)]))
    return out


def _concatenated(vit_bits, llrs, inner, fec_rows, hyp_name, deint, prefix) -> List[ChainResult]:
    """Outer RS layer after an inner convolutional decode: RS profile x outer byte-interleaver
    hypotheses on the Viterbi output, then iterative RS <-> Viterbi decoding of the best one."""
    hyps = concatenated.search_outer(vit_bits, inner.name)
    if not hyps:
        return []
    spec, ok, tot = hyps[0]
    d = concatenated.decode(llrs, spec)
    fa = _frames_for(d.data_bits, None, prefix)
    fec_rows.append(FECCandidate(f"{spec.rs_name} + {spec.outer} + {inner.name}", "Concatenated",
                                 f"outer interleaver {spec.outer}",
                                 f"blind outer search {ok}/{tot} codewords; iterative {d.history}",
                                 f"{d.decoded}/{d.codewords} RS codewords, {d.corrected_symbols} symbols corrected",
                                 fa.valid_frames))
    return [ChainResult(hyp_name, deint, f"Concatenated: {spec.rs_name} + outer {spec.outer} + {inner.name}",
                        "Concatenated", spec.describe(), d.data_bits, None, fa, d.decoded / max(1, d.codewords),
                        [Evidence(f"Inner {inner.name} (Viterbi), outer {spec.rs_name} behind byte interleaver "
                                  f"'{spec.outer}': blind outer hypotheses ranked by decoded codewords "
                                  f"({ok}/{tot} on the first pass); iterative RS->Viterbi feedback decoded "
                                  f"{d.decoded}/{d.codewords} codewords per iteration {d.history} "
                                  f"({d.erasure_decodes} with erasures)", "CONCATENATED",
                                  d.decoded / max(1, d.codewords))])]


def _ldpc(bits, llrs, fec_rows, hyp_name, deint, prefix, codes=None) -> List[ChainResult]:
    out = []
    hard = np.asarray(bits, dtype=np.uint8)
    for code in codes or ldpc.LIBRARY:
        if len(hard) < code.n:
            continue
        o, f, z = ldpc.align(hard, code)
        if z < LDPC_Z_MIN:
            continue
        d = ldpc.decode(llrs, code, o)
        fa = _frames_for(d.bits, None, prefix)
        fec_rows.append(FECCandidate(code.name, "LDPC", f"codeword offset {o}",
                                     f"unsatisfied checks {f:.3f} ({z:.0f} sigma below random 0.5)",
                                     f"min-sum: {d.converged}/{d.codewords} codewords converged, "
                                     f"{d.corrected_bits} bits corrected", fa.valid_frames))
        out.append(ChainResult(hyp_name, deint, code.name, "LDPC", code.describe(), d.bits, None, fa,
                               float(np.clip((z - 6) / 10, 0, 1)), [Evidence(
                                   f"{code.name}: codeword alignment at bit {o} by syndrome ({f:.1%} unsatisfied "
                                   f"checks, {z:.0f} sigma); normalised min-sum converged on {d.converged}/"
                                   f"{d.codewords} codewords (iterations {d.iterations[:6]})", "LDPC",
                                   d.converged / max(1, d.codewords))]))
    return out


def symbol_span(y: np.ndarray, const) -> Tuple[int, int]:
    """[start, end) in BITS of the data inside a PSK/QAM burst: guard symbols carry (almost) no
    energy, while every constellation point has at least the smallest point's amplitude (the
    inner 64QAM points are small too, so |LLR| is not a usable cue there)."""
    a = np.abs(y)
    thr = 0.5 * float(np.min(np.abs(const.points)))
    good = np.flatnonzero(a > thr)
    if len(good) == 0:
        return 0, len(a) * const.bits_per_symbol
    # first / last symbol that starts / ends a run of 3 above-threshold symbols
    on = (a > thr).astype(int)
    run = np.convolve(on, np.ones(3, dtype=int), mode="valid") == 3
    idx = np.flatnonzero(run)
    s0, s1 = (int(idx[0]), int(idx[-1]) + 3) if len(idx) else (int(good[0]), int(good[-1]) + 1)
    return s0 * const.bits_per_symbol, s1 * const.bits_per_symbol


def data_span(llrs: np.ndarray, win: int = 8, frac: float = 0.35) -> Tuple[int, int]:
    """[start, end) of the data inside a demodulated burst: the receivers also emit a few guard /
    noise symbols at the edges, whose LLR magnitudes are far below those of real data. Block and
    convolutional de-interleavers need the exact first data bit, so the edges are trimmed where the
    moving mean |LLR| over `win` bits is below `frac` x the median."""
    a = np.abs(np.asarray(llrs, dtype=np.float64))
    if len(a) < 4 * win:
        return 0, len(a)
    m = np.convolve(a, np.ones(win) / win, mode="valid")        # m[i] = mean a[i:i+win]
    thr = frac * np.median(a)
    good = np.flatnonzero(m >= thr)
    if len(good) == 0:
        return 0, len(a)
    start = int(good[0])
    while start < len(a) and a[start] < thr:                     # first individually strong bit
        start += 1
    end = int(good[-1]) + win
    while end > start and a[end - 1] < thr:
        end -= 1
    return start, end


def analyze(hyps: List[BitHypothesis], progress: Optional[Callable[[int, str], None]] = None,
            budget_s: float = 600.0, receiver_frame_ok: bool = False,
            symbols: Optional[np.ndarray] = None, modulation: Optional[str] = None) -> PostDemodResult:
    t0 = time.time()
    hist_check = histogram_check(symbols, modulation)
    budget = _Budget(budget_s)
    log: List[str] = []
    fec_rows: List[FECCandidate] = []
    deint_rows: List[DeintCandidate] = []
    chains: List[ChainResult] = []
    tested = 0
    none = DeintCandidate("none", "-", 0, 0.0, "no interleaver")

    def say(p, msg):
        log.append(msg)
        if progress:
            progress(p, msg)

    # ---- L8 bitstream report on the primary hypothesis --------------------------------------
    h0 = hyps[0]
    say(2, f"L8 bitstream: {len(h0.bits)} bits, sync-word library search (Chiani LRT/GLRT)...")
    rnd = BitstreamEntropy.estimate_randomness(h0.bits)
    ac = BitstreamCorrelation.autocorrelation(h0.bits, max_lag=min(512, len(h0.bits) - 1))
    per = BitstreamCorrelation.find_periodicity(h0.bits, max_lag=min(512, len(h0.bits) - 1))
    syncs0 = sync_search.search(h0.bits, h0.llrs)
    prof0 = blk.rank_profile(h0.bits, 64)
    det0 = sync_search.best_detection(syncs0)
    frame_len = det0.period if det0 and det0.period else (per[0][0] if per else None)
    bs = BitstreamReport(h0.name, len(h0.bits), rnd, per[:10], ac, syncs0, prof0, blind_conv_params(prof0), frame_len,
                         np.packbits(h0.bits[:256 // 8 * 8]).tobytes().hex(" ").upper())

    def finish(best: Optional[ChainResult], scan: StructureScan, pe=None):
        fam = classify_family(scan, rs_ok_fraction=1.0 if best and best.fec_family == "Reed-Solomon" and best.crc_frames
                              else 0.0, uncoded_crc_ok=bool(best and best.fec_name == "none" and best.crc_frames),
                              uncoded_sync=bool(det0))
        if best is not None and best.crc_frames:           # decoding + CRC outrank the statistics
            fam_name = "Uncoded" if best.fec_name == "none" else {
                "Convolutional": "Convolutional code", "LDPC": "LDPC", "LDPC (blind)": "LDPC",
                "Concatenated": "Concatenated (RS + convolutional)"}.get(best.fec_family, "Block code (Hamming / RS)")
            for k in ("LDPC", "Concatenated (RS + convolutional)"):
                fam.probabilities.setdefault(k, 0.0)
            others = max(1, len(fam.probabilities) - 1)
            fam.probabilities = {k: (0.97 if k == fam_name else 0.03 / others) for k in fam.probabilities}
            fam.family = fam_name
            fam.evidence.insert(0, Evidence(f"{best.fec_name if best.fec_name != 'none' else 'Uncoded'} confirmed by "
                                            f"{best.crc_frames} CRC-valid frame(s)", "CRC", 1.0))
        deint_best = best.deint if best else none
        for d in deint_rows:
            if best and d is best.deint:
                d.validated = best.crc_frames > 0
        return PostDemodResult(bs, fam, scan, deint_rows, deint_best, pe, fec_rows, best, tested,
                               time.time() - t0, log, hist_check)

    scan0 = structure_scan(h0.bits)
    n_h = len(hyps)
    # ---- A + B: raw stream, every hypothesis -------------------------------------------------
    for hi, h in enumerate(hyps):
        say(5 + int(30 * hi / n_h), f"Hypothesis '{h.name}': raw sync / CRC, code structure...")
        tested += 1
        fa = _frames_for(h.bits, h.llrs)
        chains.append(ChainResult(h.name, none, "none", "Uncoded", "no FEC", h.bits, h.llrs, fa, 1.0 if fa.valid_frames
                                  else 0.0, list(fa.evidence)))
        a0, a1 = h.span if h.span else data_span(h.llrs)
        streams = [(h.bits[a0:a1], np.asarray(h.llrs)[a0:a1], True, f"data span bits {a0}..{a1}", None)]
        if fa.sync is not None and fa.sync.detected:          # sync in clear, body coded (CADU style)
            s0 = fa.sync.occurrences[0] if fa.sync.occurrences else fa.sync.offset
            b = h.bits[s0 + fa.sync.length:]
            l = h.llrs[s0 + fa.sync.length:]
            if fa.sync.inverted:
                b, l = 1 - b, -l
            streams.insert(0, (b, l, True, f"body after sync at bit {s0}",
                               sync_search.SYNC_LIBRARY.get(fa.sync.name)))
        if fa.valid_frames:
            # Frame valid in clear. A systematic code (RS) keeps the data readable, so check whether RS
            # parity explains the rest of the burst before reporting the stream as uncoded.
            for bits, llrs, _, label, prefix in streams:
                res = _decode_with(bits, llrs, NO_STRUCTURE, True, fec_rows,
                                   f"{h.name}; {label}", none, allow_rs=True, prefix=prefix)
                rs_ok = [c for c in res if c.fec_family == "Reed-Solomon" and c.crc_frames and c.strength >= 1.0]
                if rs_ok:
                    chains.extend(rs_ok)
                    say(95, f"{rs_ok[0].fec_name} (systematic) validated by decoding + CRC under '{h.name}'")
                    return finish(rs_ok[0], structure_scan(bits))
            fec_rows.append(FECCandidate("none", "Uncoded", h.name, "sync + CRC in clear", "frame parsed",
                                         fa.valid_frames))
            say(95, f"Uncoded frame validated by CRC under '{h.name}'")
            return finish(chains[-1], structure_scan(h.bits))
        for bits, llrs, start_known, label, prefix in streams:
            scan = structure_scan(bits)
            res = _decode_with(bits, llrs, scan, start_known, fec_rows, f"{h.name}; {label}", none,
                               allow_rs=True, allow_punctured=label.startswith("body") or hi == 0, prefix=prefix)
            chains.extend(res)
            ok = [c for c in res if c.crc_frames]
            if ok:
                best = max(ok, key=lambda c: c.key())
                say(95, f"{best.fec_name} validated by CRC ({h.name}; {label})")
                return finish(best, scan)
        if budget.exhausted():
            break

    # ---- C: de-interleaving -------------------------------------------------------------------
    pe = None

    def screen(b):
        """Code-structure strength; a sync word in clear counts fully only if its CRC validates
        (systematic codes can put the sync back in clear under a wrong de-interleaver too)."""
        sc = structure_scan(b)
        syn = sync_search.best_detection(sync_search.search(b, None))
        st = sc.strength
        detail = f"conv syndrome {sc.conv_syndrome:.3f}, Hamming fail {sc.block_fail:.2f}"
        if syn and syn.p_emulation < 1e-6:
            fa = analyze_frames(b, syn, sync_search.SYNC_LIBRARY.get(syn.name))
            st = max(st, 1.0 if fa.valid_frames else 0.5)
            detail += f", sync '{syn.name}' in clear" + (" + CRC valid" if fa.valid_frames else "")
        return st, sc, detail

    def z_of(sc: StructureScan) -> float:
        return max(sc.conv_z, sc.block_z)

    screened: List[np.ndarray] = []
    for hi, h in enumerate(hyps):
        if budget.exhausted():
            say(90, "Search budget exhausted")
            break
        # A hypothesis that is the bitwise complement of one already searched adds nothing here:
        # the library convolutional codes (odd-weight generators) and Hamming codes are closed under
        # complement, and the frame analyser matches inverted sync words.
        if any(len(p) == len(h.bits) and np.mean(p != h.bits) > 0.99 for p in screened):
            log.append(f"'{h.name}' is the complement of a searched hypothesis: skipped in L9")
            continue
        screened.append(h.bits)
        s0, s1 = h.span if h.span else data_span(h.llrs)
        o0 = min(8, s0)                              # screen at the span start; refine from 8 bits before it
        s0 -= o0
        hb, hl = h.bits[s0:s1], np.asarray(h.llrs)[s0:s1]
        N = len(hb)
        base = 35 + int(55 * hi / n_h)
        if hi == 0:
            log.append(f"Data span inside the burst: bits {s0}..{s1} of {len(h.bits)} (guard symbols trimmed "
                       f"by {'symbol amplitude' if h.span else '|LLR|'})")
        # C1: Burel-Gautier period estimate (only possible when N > P^2)
        if hi == 0:
            say(base, "L9: rank-deficiency period / alignment search (Burel-Gautier, K-S)...")
            pe = blk.estimate_period(hb, L_max=min(128, int(np.sqrt(N)) - 4))
            if pe.period:
                deint_rows.append(DeintCandidate("block-period", f"P = {pe.period}", s0 + pe.offset, pe.deficiency,
                                                 f"rank deficiency {pe.deficiency:.2f} at offset {pe.offset}; K-S "
                                                 f"{'n/a' if pe.ks is None else f'{pe.ks:.2f}'}", stream=h.name))
        # Hypotheses: estimated period, block R x C with the data span a whole number of blocks,
        # convolutional (B, M); each at offset 0 of the trimmed span, refined over offsets when it
        # shows structure.
        dims = sorted({(R, C) for (R, C, o) in blk.dimension_hypotheses(N, slack=16, max_dim=128)},
                      key=lambda d: (d[0] * d[1], d[0]))
        order: List[Tuple[str, tuple]] = []
        if pe is not None and pe.period:
            P = pe.period
            order += [("block", (R, P // R)) for R in range(2, P // 2 + 1) if P % R == 0]
        order += [("block", d) for d in dims]
        order += [("conv", bm) for bm in cvi.hypotheses(N)]
        order += [("diag", (R, C, dr)) for (R, C) in dims for dr in dgi.DIRECTIONS]
        pr_periods = ([pe.period] if pe is not None and pe.period else [])
        order += [("pr", h) for P in pr_periods for h in pri.hypotheses(P, all_seeds=True)]
        order += [("pr", h) for P in range(max(16, N - 16), N + 1) for h in pri.hypotheses(P)]
        say(base + 2, f"L9: {len(order)} interleaver hypotheses (block R x C / convolutional B, M), '{h.name}'...")

        def build(kind, prm, o):
            if kind == "block":
                R, C = prm
                return (blk.deinterleave(hb, R, C, o), lambda l: blk.deinterleave(l, R, C, o),
                        f"{R} x {C} (write rows, read columns)")
            if kind == "diag":
                R, C, dr = prm
                return (dgi.deinterleave(hb, R, C, dr, o), lambda l: dgi.deinterleave(l, R, C, dr, o),
                        f"{R} x {C} diagonal ({'down-right' if dr > 0 else 'down-left'})")
            if kind == "pr":
                perm = prm.perm()
                return (pri.deinterleave(hb, perm, o), lambda l: pri.deinterleave(l, perm, o), prm.describe())
            B, M = prm
            return (cvi.deinterleave(hb[o:], B, M), lambda l: cvi.deinterleave(l[o:], B, M),
                    f"B = {B} branches, M = {M}")

        cand: List[Tuple[float, DeintCandidate, np.ndarray, np.ndarray]] = []
        for kind, prm in order:
            if budget.exhausted():
                break
            b, fn, label = build(kind, prm, o0)
            if len(b) < 64:
                continue
            tested += 1
            budget.used += 1
            st, sc, detail = screen(b)
            if st <= 0 and z_of(sc) < 6:
                continue
            # offset refinement: the span start may still be off by a few bits
            best_o = (z_of(sc) + 100 * (st >= 1.0), o0, st, detail, b, fn)
            max_o = min(32, (prm[0] * prm[1] - 1) if kind in ("block", "diag") else 32)
            for o in range(0, max_o + 1):
                if o == o0:
                    continue
                b2, fn2, _ = build(kind, prm, o)
                if len(b2) < 64:
                    break
                tested += 1
                budget.used += 1
                st2, sc2, d2 = screen(b2)
                zz = z_of(sc2) + 100 * (st2 >= 1.0)
                if zz > best_o[0]:
                    best_o = (zz, o, st2, d2, b2, fn2)
            _, o, st, detail, b, fn = best_o
            if st > 0:
                dc = DeintCandidate(kind, label, s0 + o, st, detail, stream=h.name)
                cand.append((st, dc, b, fn(hl)))
                if st >= STRONG:
                    # decode now; a structured but wrong hypothesis (e.g. a divisor of the true
                    # period) does not stop the search unless its frame validates
                    deint_rows.append(dc)
                    res = _decode_with(b, fn(hl), structure_scan(b), True, fec_rows, h.name, dc, allow_rs=True)
                    chains.extend(res)
                    ok = [c for c in res if c.crc_frames]
                    if ok:
                        best = max(ok, key=lambda c: c.key())
                        say(95, f"De-interleaver {dc.kind} {dc.params} + {best.fec_name} validated by CRC ({h.name})")
                        return finish(best, structure_scan(b), pe)
                    cand.pop()
        cand.sort(key=lambda c: -c[0])
        for st, dc, b, l in cand[:4]:
            deint_rows.append(dc)
            res = _decode_with(b, l, structure_scan(b), True, fec_rows, h.name, dc, allow_rs=True,
                               allow_punctured=False)
            chains.extend(res)
            ok = [c for c in res if c.crc_frames]
            if ok:
                best = max(ok, key=lambda c: c.key())
                say(95, f"De-interleaver {dc.kind} {dc.params} + {best.fec_name} validated by CRC ({h.name})")
                return finish(best, structure_scan(b), pe)

    # ---- D: period found, nothing validated: PR interleaver evidence + blind parity checks ------------
    if pe is not None and pe.period:
        txt = pri.presence_evidence(pe.period, structured_found=False)
        if txt:
            deint_rows.append(DeintCandidate("pseudo-random?", f"period {pe.period}", pe.offset, pe.deficiency, txt,
                                             stream=hyps[0].name))
            log.append(txt)
        h0b, h0l = hyps[0].bits, np.asarray(hyps[0].llrs)
        # Candidate codeword lengths: the most rank-deficient row lengths (a code with local checks,
        # e.g. a Gallager band on consecutive bits, is deficient at a short L too, but the codeword
        # length has the largest normalised deficiency (n - k) / n).
        lengths = [L for L, _ in sorted(pe.candidates, key=lambda c: -c[1])[:3]]
        for P in lengths:
            if budget.exhausted():
                break
            heat = blk.period_offset_map(h0b, [P])[P]
            o = int(np.argmax(heat))
            rows = (len(h0b) - o) // P
            if rows <= P + 8:
                continue
            say(92, f"Blind parity-check recovery on {rows} aligned {P}-bit blocks (null space + sparse checks)...")
            rec = recover_parity_checks(h0b[o:o + rows * P].reshape(rows, P), w_max=16, iterations=40)
            if not len(rec["H"]):
                continue
            code = ldpc.code_from_checks(f"blind parity checks (n={P}, {len(rec['H'])} checks of weight "
                                         f"{min(rec['row_weights'])}-{max(rec['row_weights'])})", rec["H"])
            d = ldpc.decode(h0l[o:], code, 0)
            fa = _frames_for(d.bits, None)
            fec_rows.append(FECCandidate(code.name, "Blind sparse parity checks", f"block offset {o}",
                                         f"dual dimension {rec['dual_dimension']}, recovered "
                                         f"{len(rec['H'])} ({'complete' if rec['complete'] else 'partial'})",
                                         f"min-sum {d.converged}/{d.codewords} converged", fa.valid_frames,
                                         stream=hyps[0].name))
            ch = ChainResult(hyps[0].name, none, code.name, "LDPC (blind)", code.describe(), d.bits, None, fa,
                             d.converged / max(1, d.codewords), [Evidence(
                                 f"Parity checks recovered blindly from {rows} aligned {P}-bit blocks (offset {o}): "
                                 f"dual dimension {rec['dual_dimension']}, {len(rec['H'])} sparse checks "
                                 f"({'complete' if rec['complete'] else 'partial'}); information bits taken from the "
                                 f"leftmost information set", "DUAL", 0.5)])
            chains.append(ch)
            if fa.valid_frames:
                say(95, "Blindly recovered parity checks + min-sum decoding validated by CRC")
                return finish(ch, scan0, pe)

    best = max(chains, key=lambda c: c.key()) if chains else None
    say(95, "No hypothesis validated by CRC: best evidence reported (unverified)")
    return finish(best, scan0, pe)
