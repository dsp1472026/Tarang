"""
Blind FSK / CPFSK / MSK / GFSK / GMSK receiver (Phase 2), L1-L6.

    L1  block AGC; DC from noise-only samples; blind IQ-imbalance correction
    L2  occupied band (97 %), channel filter, energy burst detection
    L3  symbol rate + boundary timing (estimation.fsk_estimator.estimate_rate),
        tone grid (M, h, CFO) from IF integrate-and-dump strobes, phase continuity,
        pulse shape / BT / h / fine timing by re-modulation match
    L4  family evidence (constant envelope) and variant name (MSK/GMSK/GFSK/M-FSK)
    L5  timing and tone alignment above; coherent phase reference for CPM
    L6  three demodulators (discriminator, non-coherent correlator bank, coherent
        max-log BCJR over the CPM trellis), each resolved against the sync word
        over all mapping hypotheses; hard bits + max-log LLRs

Backtracking: if no demodulator finds a CRC-valid frame, the next symbol-rate
candidate (and then the next M) is tried. The only inputs are the samples and,
if known, the sample rate.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.evidence import Evidence, Parameter
from demodulation.fsk_demod import (FSKDecode, bcjr_ll, coherent_reference, correlator_ll, decode_fsk,
                                    discriminator_ll, isi_viterbi, rational_h, strobe_isi_taps)
from dsp.conditioning import channel_filter, detect_burst, estimate_dc, normalize_power
from dsp.cpm import fsk_variant_name
from estimation.fsk_estimator import (boundaries_in, cumulative_phase, dd_refine, estimate_rate, fit_levels,
                                      if_strobes_median, inst_freq, iq_balance, model_match,
                                      model_match_symbols, occupied_band,
                                      phase_continuity, psd_tone_peaks, track_rate)
from modulation.fsk_classifier import FamilyDecision, classify_family

BT_CANDIDATES = (1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2)      # simplest (widest) first
METHOD_PREFERENCE = ("coherent-bcjr", "correlator", "discriminator")
MODEL_SYMBOLS = 300                                        # symbols used by the model-match scans
M_TOL = 0.001                                              # re-modulation match tolerance for choosing M
PULSE_TOL = 0.0003                                         # rect preferred over Gaussian only within this
GAUSS_MAX_ORDER = 4                                        # Gaussian-pulse hypotheses tested for M <= 4


@dataclass
class FSKReceiverConfig:
    expected_payload: Optional[str] = None      # only for captures without a sync word
    max_rate_candidates: int = 10
    methods: tuple = METHOD_PREFERENCE


@dataclass
class ToneModel:
    order: int
    h: float
    center: float            # residual CFO, cycles/symbol
    pulse: str
    bt: Optional[float]
    sps: float
    starts: np.ndarray       # symbol start samples (burst coordinates)
    match: float
    continuous: bool
    level_fits: Dict[int, float]
    pulse_scores: Dict[str, float]
    spike_rates: tuple = (0.0, 0.0)


@dataclass
class FSKReceiverResult:
    family: FamilyDecision
    variant: str
    model: ToneModel
    parameters: Dict[str, Parameter]
    decodes: Dict[str, FSKDecode]
    primary: FSKDecode
    fsk_error_pct: float
    esn0_est_db: float
    evidence: List[Evidence] = field(default_factory=list)
    positions: Optional[np.ndarray] = None    # decided tone positions (0..M-1) of the primary demodulator

    @property
    def payload_text(self) -> str:
        return self.primary.text

    @property
    def hard_bits(self):
        return self.primary.hard_bits

    @property
    def llrs(self):
        return self.primary.llrs


def _param(name, value, unit, source, method, conf=None, ev=None) -> Parameter:
    return Parameter(name, value, unit, source, method, conf, ev or [])


def _levels(fit_pos, order):
    return 2 * fit_pos - order + 1


def _model_window(starts, sps, n_max=MODEL_SYMBOLS):
    """Indices of a central run of <= n_max symbols (for the model scans)."""
    n = len(starts)
    k0 = max(0, (n - n_max) // 2)
    return np.arange(k0, min(n, k0 + n_max))


def fit_tone_model(xr: np.ndarray, fb: np.ndarray, sps: float, boundary: float) -> ToneModel:
    """
    L3/L4 for one rate hypothesis: tone grid, continuity, and the joint choice of
    (M, pulse shape / BT, h, fine timing) by re-modulation match.

    xr: UNFILTERED burst (coarse CFO removed): the channel filter smooths tone
        transitions and would make rectangular pulses look Gaussian.
    fb: discriminator output of the channel-filtered burst (decisions only).
    M is chosen jointly with the pulse: Gaussian ISI makes 2-level GMSK strobes
    look 4-level, which only a full waveform model can disambiguate.
    """
    starts = boundaries_in(boundary, sps, 0.0, float(len(xr)))
    # keep symbols that are >= 75 % inside the capture (whole-file captures clip the ends)
    starts = starts[(starts >= -0.25 * sps) & (starts + sps <= len(xr) + 0.25 * sps)]
    v = if_strobes_median(fb, starts, sps)
    core = v[2:-2] if len(v) > 12 else v
    fits = {M: fit_levels(core, M) for M in (2, 4, 8) if len(core) >= 3 * M}
    if not fits:
        return None                                    # too few symbols for this rate hypothesis
    M0 = max(fits, key=lambda m: fits[m].loglik)
    fit0 = fits[M0]
    max_dev = (M0 - 1) * fit0.spacing / 2 / sps
    cont = phase_continuity(inst_freq(xr), starts, sps, max_dev + abs(fit0.center) / sps)
    win = _model_window(starts, sps)

    taps = {bt: strobe_isi_taps("gaussian", bt) for bt in BT_CANDIDATES}

    cache = {}

    def decisions(M, pulse, bt, offset, h):
        """Level fit + (for Gaussian pulses) MLSE decisions, cached per (M, pulse, offset)."""
        key = (M, pulse, bt, round(float(offset), 3))
        if key not in cache:
            st = starts + offset
            vv = if_strobes_median(fb, st[win], sps)           # the scans only use the model window
            f2 = fit_levels(vv, M, iters=6)
            pos = f2.positions if pulse == "rect" else isi_viterbi(vv, M, taps[bt], h, f2.center)
            full_pos = np.zeros(len(st), dtype=np.int64)
            full_pos[win] = pos
            cache[key] = (st, f2, full_pos)
        return cache[key]

    def score(M, pulse, bt, h, offset):
        st, f2, pos = decisions(M, pulse, bt, offset, h)
        lv = _levels(pos, M)[win]
        s0 = st[win[0]]
        n0, n1 = max(int(np.ceil(s0)), 0), min(int(np.floor(st[win[-1]] + sps)), len(xr))
        t = (np.arange(n0, n1) - s0) / sps
        if cont.continuous:
            m = model_match(xr[n0:n1], t, lv, h, f2.center, pulse, bt)
        else:   # phase jumps: score each symbol on its own (all hypotheses share this rate)
            m = model_match_symbols(xr[n0:n1], t, lv, h, f2.center)
        return m, f2

    pulses = [("rect", None)] + ([("gaussian", bt) for bt in BT_CANDIDATES] if cont.continuous else [])
    offsets = np.linspace(-0.5, 0.5, 9)[:-1] * sps
    results = {}
    for M in sorted(fits):                                         # smallest M first (simplest)
        spacing = fits[M].spacing
        for pulse, bt in pulses:
            if pulse != "rect" and M > GAUSS_MAX_ORDER:
                continue                                           # 8-ary GFSK is out of scope
            h0 = spacing * (1.0 if pulse == "rect" else 1.3)
            # rect normally starts at the estimated boundary (offset 0); the coarse timing scan the
            # Gaussian pulses get is also given to rect so that neither wins on timing alone
            o = max(((score(M, pulse, bt, h0, o)[0], o) for o in offsets))[1]
            hs = spacing * (np.linspace(0.9, 1.1, 9) if pulse == "rect" else np.linspace(0.9, 1.9, 11))
            hb = max(((score(M, pulse, bt, h, o)[0], h) for h in hs))[1]
            fine = o + np.linspace(-1, 1, 9) * sps / 32
            m, o = max(((score(M, pulse, bt, hb, oo)[0], oo) for oo in fine))
            results[(M, pulse, bt)] = (m, hb, o)
    # Selection. Score differences between the right and a wrong pulse are small
    # (~0.001-0.003) but consistent over hundreds of symbols, so the evidence is
    # followed closely:
    #   M     = smallest order within M_TOL of the best (an M=4 grid using only two
    #           levels is the same waveform as M=2, hence near-exact ties);
    #   pulse = best score for that M; rect only wins a near-exact tie (PULSE_TOL).
    # M by a minimum-description-length rule: under AWGN the log-likelihood per
    # symbol is ~ -sps (1 - match) / NSR and describing a symbol costs ln M nats,
    # so a larger alphabet must buy ln(M/2) * NSR / sps of match (NSR from the
    # best model's residual). Otherwise fine-grained grids (small h, many levels)
    # track ISI and noise and always "fit" better.
    top = max(r[0] for r in results.values())
    nsr = max(1.0 - top, 1e-3)
    pen = {k: r[0] - np.log(k[0] / 2) * nsr / sps for k, r in results.items()}
    best_pen = max(pen.values())
    M = min(k[0] for k in results if pen[k] >= best_pen - M_TOL)
    mine = {k: r for k, r in results.items() if k[0] == M}
    best_k = max(mine, key=lambda k: mine[k][0])
    rect_k = (M, "rect", None)
    # Noise-aware tolerance: the spread of a match score over n windows scales with
    # NSR / sqrt(n); at low SNR a Gaussian model must win by more than that.
    n_win = max(1, int(len(win) * sps) // 24)
    tol = max(PULSE_TOL, 0.15 * nsr / np.sqrt(n_win))
    key = rect_k if mine[rect_k][0] >= mine[best_k][0] - tol else best_k
    _, pulse, bt = key
    m, h, o = results[key]
    # fine h / timing refinement for the chosen hypothesis
    for _ in range(2):
        h = max(((score(M, pulse, bt, hh, o)[0], hh) for hh in h * np.linspace(0.95, 1.05, 11)))[1]
        m, o = max(((score(M, pulse, bt, h, oo)[0], oo) for oo in o + np.linspace(-1, 1, 9) * sps / 64))
    _, f2 = score(M, pulse, bt, h, o)
    st = starts + o
    st = st[(st >= -0.25 * sps) & (st + sps <= len(xr) + 0.25 * sps)]   # offset may push edge symbols out
    return ToneModel(M, float(h), float(f2.center), pulse, bt, sps, st, float(m), cont.continuous,
                     {k: round(q.loglik, 3) for k, q in fits.items()},
                     {f"M={k[0]} {k[1]}{'' if k[2] is None else ' BT=' + str(k[2])}": round(r[0], 4)
                      for k, r in results.items()}, (cont.boundary_excess, cont.std_error))


def demodulate(xr: np.ndarray, fb: np.ndarray, tm: ToneModel, methods, expected_payload):
    """
    Run the demodulators on a tone model; returns {method: FSKDecode} and aux data.
    xr: unfiltered burst (coarse CFO removed) -- matched processing in white noise.
    fb: IF of the channel-filtered burst (discriminator only).
    """
    M, sps, starts = tm.order, tm.sps, tm.starts
    v = if_strobes_median(fb, starts, sps)
    lv_cyc = tm.center + 0.5 * tm.h * (2 * np.arange(M) - M + 1)      # tone centres, cycles/symbol
    sigma = float(np.sqrt(np.mean((v - lv_cyc[np.argmin(np.abs(v[:, None] - lv_cyc[None, :]), axis=1)]) ** 2)))
    out, aux = {}, {"v": v, "levels": lv_cyc, "sigma": sigma}

    LL_disc = discriminator_ll(v, lv_cyc, sigma)
    aux["LL"] = {"discriminator": LL_disc}
    n0_idx = max(int(np.ceil(starts[0])), 0)
    n1_idx = min(int(np.floor(starts[-1] + sps)), len(xr))
    t = (np.arange(n0_idx, n1_idx) - starts[0]) / sps
    decisions = np.argmax(LL_disc, axis=1)
    if tm.pulse != "rect":
        decisions = isi_viterbi(v, M, strobe_isi_taps(tm.pulse, tm.bt), tm.h, tm.center)
        aux["mlse"] = True
    ref = coherent_reference(xr[n0_idx:n1_idx], t, 2 * decisions - M + 1, tm.h, tm.center, tm.pulse, tm.bt, sps)
    aux["n0"] = ref.n0
    if "discriminator" in methods:
        out["discriminator"] = decode_fsk(LL_disc, "discriminator", expected_payload)
    if "correlator" in methods:
        tones = lv_cyc / sps
        LL_c = correlator_ll(xr, starts, sps, tones, ref.n0)
        out["correlator"] = decode_fsk(LL_c, "correlator", expected_payload)
        aux["LL"]["correlator"] = LL_c
        if tm.pulse == "rect":
            decisions = np.argmax(LL_c, axis=1)
    hq = rational_h(tm.h)
    if "coherent-bcjr" in methods and tm.continuous and hq is not None and hq.denominator <= 16:
        ref = coherent_reference(xr[n0_idx:n1_idx], t, 2 * decisions - M + 1, float(hq), tm.center, tm.pulse,
                                 tm.bt, sps)
        LL_b = bcjr_ll(ref.r_coh, t, len(starts), M, hq, tm.pulse, tm.bt, ref.n0)
        # One decision-directed iteration: the BCJR decisions are better than the
        # non-coherent ones, so the carrier reference is rebuilt from them.
        ref = coherent_reference(xr[n0_idx:n1_idx], t, 2 * np.argmax(LL_b, axis=1) - M + 1, float(hq), tm.center,
                                 tm.pulse, tm.bt, sps)
        LL_b = bcjr_ll(ref.r_coh, t, len(starts), M, hq, tm.pulse, tm.bt, ref.n0)
        out["coherent-bcjr"] = decode_fsk(LL_b, "coherent-bcjr", expected_payload)
        aux["LL"]["coherent-bcjr"] = LL_b
        aux["h_rational"] = str(hq)
    return out, aux


FULL_FIT_MAX = 4


def screen_candidate(fb: np.ndarray, sps: float, boundary: float) -> bool:
    """
    Cheap tier-1 test of a rate hypothesis: IF-strobe decisions (rect: nearest level
    for M = 2, 4, 8; binary Gaussian: 4-state ISI-MLSE with BT 0.3 / 0.5 taps, at a
    few timing offsets) followed by a sync-word search.
    """
    for off in (0.0, 0.25, 0.5, 0.75):
        starts = boundaries_in(boundary + off * sps, sps, 0.0, float(len(fb) + 1))
        v = if_strobes_median(fb, starts, sps)
        for M in (2, 4, 8):
            if len(v) < 3 * M or (off and M > 2):
                continue
            fit = fit_levels(v, M, iters=8)
            LL = discriminator_ll(v, fit.levels(), fit.sigma)
            if decode_fsk(LL, "screen").resolution == "sync-word":
                return True
        fit = fit_levels(v, 2, iters=8)
        for bt in (0.3, 0.5):
            pos = isi_viterbi(v, 2, strobe_isi_taps("gaussian", bt), 1.3 * fit.spacing, fit.center)
            if decode_fsk(-5.0 * (np.arange(2)[None, :] != pos[:, None]), "screen").resolution == "sync-word":
                return True
    return False


def _pick_primary(decodes: Dict[str, FSKDecode]) -> FSKDecode:
    def key(item):
        name, d = item
        order = METHOD_PREFERENCE.index(name) if name in METHOD_PREFERENCE else 9
        return (not d.crc_ok, d.resolution != "sync-word", d.sync_bit_errors if d.sync_bit_errors is not None else 99,
                order)
    return min(decodes.items(), key=key)[1]


def _repeat_fraction(attempt, k: int) -> float:
    """Largest fraction (over the k alignments) of aligned k-symbol groups whose decided tones are
    all equal, for the primary demodulator of a backtracking attempt."""
    aux, prim = attempt[4], attempt[5]
    LL = aux["LL"].get(getattr(prim, "method", ""), aux["LL"].get("discriminator"))
    if LL is None or len(LL) < 4 * k:
        return 0.0
    pos = np.argmax(np.asarray(LL), axis=1)
    best = 0.0
    for o in range(k):
        g = pos[o:o + (len(pos) - o) // k * k].reshape(-1, k)
        best = max(best, float(np.mean(np.all(g == g[:, :1], axis=1))))
    return best


def receive_fsk(samples: np.ndarray, sample_rate: Optional[float] = None,
                config: Optional[FSKReceiverConfig] = None) -> FSKReceiverResult:
    cfg = config or FSKReceiverConfig()
    fs = sample_rate
    hz = (lambda v: v * fs) if fs else (lambda v: v)
    f_unit = "Hz" if fs else "cycles/sample"
    ev: List[Evidence] = []
    params: Dict[str, Parameter] = {
        "sample_rate": _param("Sample rate", fs, "Hz", "KNOWN" if fs else "UNKNOWN",
                              "metadata" if fs else "not recoverable from samples alone")}

    # ---- L1/L2 ---------------------------------------------------------------
    x, _ = normalize_power(np.asarray(samples, dtype=np.complex128))
    band0 = occupied_band(x)
    burst = detect_burst(x, band0.center, band0.bw99)
    dc, dc_method = estimate_dc(x, burst)
    x = x - dc
    # Blind IQ-imbalance correction is a hypothesis: E[x^2] != 0 can also come from the
    # signal itself (h = 1/2 CPM has x^2 spectral lines; biased payloads make their mean
    # non-zero). Accept it only if it flattens the envelope of this constant-envelope
    # signal (IQ imbalance shows up as envelope ripple).
    xb0 = x[burst.start:burst.stop]
    xb1, iqb = iq_balance(xb0, threshold=0.0)
    cv0, cv1 = (float(np.std(np.abs(z)) / (np.mean(np.abs(z)) + 1e-20)) for z in (xb0, xb1))
    iqb.applied = bool(cv1 < 0.95 * cv0)
    xb = xb1 if iqb.applied else xb0
    x[burst.start:burst.stop] = xb
    noise_ps = None if burst.whole_file else float(np.mean(np.abs(np.r_[x[:burst.start], x[burst.stop:]]) ** 2))
    band = occupied_band(xb)
    ev.append(Evidence(f"DC {abs(dc):.3g} removed ({dc_method}); IQ imbalance "
                       + (f"corrected (Q gain {iqb.gain_db:+.2f} dB, skew {iqb.phase_deg:+.1f} deg; envelope CV "
                          f"{cv0:.3f} -> {cv1:.3f})" if iqb.applied else
                          f"not applied (whitening would not flatten the envelope: CV {cv0:.3f} -> {cv1:.3f})"),
                       "L1", 1.0))
    ev.append(Evidence("Signal occupies the whole capture" if burst.whole_file else
                       f"Burst at samples {burst.start}-{burst.stop}", "L2_BURST", 1.0))

    family = classify_family(xb, band, noise_ps)
    ev.extend(family.evidence)

    n = np.arange(len(x))
    y = channel_filter(x * np.exp(-2j * np.pi * band.center * n), 0.0, band.bw99, 0.6)
    f = inst_freq(y)
    yb, fb = y[burst.start:burst.stop], f[burst.start:burst.stop - 1]
    xr = (x * np.exp(-2j * np.pi * band.center * n))[burst.start:burst.stop]    # unfiltered, coarse CFO removed
    Phi = cumulative_phase(fb)

    # ---- L3: symbol rate candidates -------------------------------------------
    # Phase continuity needs no timing, only a rough rate: test it first so that the
    # rate refinement can use the right re-modulation model.
    rough = estimate_rate(yb, fb, band.bw99)
    # Continuity by the phase-jump spike histogram, scanned over +-1.2 % of the rough
    # rate so that a small rate error cannot smear the spikes across bins. For
    # discontinuous signals the best scan point is also a precise rate estimate.
    f_raw = inst_freq(xr)
    scan = [(phase_continuity(f_raw, np.array([0.0]), s_, 0.0), s_)
            for s_ in rough.sps * (1 + np.linspace(-0.012, 0.012, 97))]
    cont0, s_jump = max(scan, key=lambda cs: cs[0].boundary_excess)
    if cont0.continuous:
        rate = rough
    else:
        rate = estimate_rate(yb, fb, band.bw99, continuous=False)
        _, s_ref, b_ref = dd_refine(yb, Phi, s_jump, span=0.001, steps=5, continuous=False)
        rate.sps, rate.boundary = s_ref, b_ref
        ev.append(Evidence(f"Phase-discontinuous (boundary |IF| excess {cont0.boundary_excess:.3f} cycles/sample): "
                           f"rate refined on the phase-jump spikes", "L3_RATE", 1.0))
    cands = [(rate.sps, rate.boundary, "refined best line")]
    for c in rate.candidates[1:cfg.max_rate_candidates]:
        cands.append((c["sps"], None, c.get("source", "line")))

    # Backtracking. The best-ranked rate gets a full fit first (the common case ends
    # here with a CRC-valid frame). Otherwise every other candidate is screened cheaply
    # (rect model, IF-strobe decisions, sync-word search) and full fits run on those
    # with a sync hit, then on the next FULL_FIT_MAX - 1. A CRC-valid frame (or, for
    # unframed captures, a content-validated decode) ends the search; otherwise the
    # best attempt by (CRC, sync hit, match) is kept.
    attempts, chosen = [], None
    # Whole-burst timing tracking on the best-ranked rate (long bursts only).
    s_tr, b_tr, nseg = track_rate(yb, Phi, cands[0][0], cands[0][1], continuous=cont0.continuous)
    if nseg != 1:
        ev.append(Evidence(f"Whole-burst timing tracking over {abs(nseg)} segments: " +
                           (f"rate corrected by {1e6 * (s_tr / cands[0][0] - 1):+.0f} ppm" if nseg > 0 else
                            "drift not consistent, correction rejected"), "L5_TIMING", float(nseg > 0)))
        cands[0] = (s_tr, b_tr, cands[0][2])

    def full(sps, boundary, src):
        tm = fit_tone_model(xr, fb, sps, boundary)
        if tm is None:
            return False
        decodes, aux = demodulate(xr, fb, tm, cfg.methods, cfg.expected_payload)
        attempts.append((sps, src, tm, decodes, aux, _pick_primary(decodes)))
        p = attempts[-1][5]
        return p.crc_ok or p.resolution in ("known-payload", "text-heuristic")

    if full(*cands[0]):
        chosen = attempts[-1]
    else:
        screened = []
        for sps, boundary, src in cands[1:]:
            if boundary is None:
                _, sps, boundary = dd_refine(yb, Phi, sps, span=0.015, steps=61, continuous=cont0.continuous)
            screened.append((sps, boundary, src, screen_candidate(fb, sps, boundary)))
        order = [c for c in screened if c[3]] + [c for c in screened if not c[3]][:FULL_FIT_MAX - 1]
        for sps, boundary, src, _ in order:
            if full(sps, boundary, src):
                chosen = attempts[-1]
                break
    if chosen is None:
        if not attempts:
            raise ValueError("no usable FSK hypothesis (too few symbols at every candidate rate)")
        chosen = max(attempts, key=lambda a: (a[5].crc_ok, a[5].resolution == "sync-word", a[2].match))
        # No content check decided (e.g. FEC-coded, random-looking data). A rate k times too high
        # also explains the signal (2FSK h=1 at Rs == MSK at 2 Rs with every symbol repeated), so if
        # the chosen decisions repeat in aligned groups of k, the slower attempt is the signal.
        if not chosen[5].crc_ok:
            for a in sorted(attempts, key=lambda a: -a[0]):
                k = a[0] / chosen[0]
                if a is chosen or round(k) < 2 or abs(k - round(k)) > 0.03:
                    continue
                rep = _repeat_fraction(chosen, int(round(k)))
                if rep >= 0.9:
                    ev.append(Evidence(f"Decisions at {chosen[0]:.3f} sps repeat in aligned groups of "
                                       f"{int(round(k))} ({rep:.1%}): the signal is the {a[0]:.3f} sps hypothesis "
                                       f"(simpler model, same waveform)", "L3_RATE_OCCAM", rep))
                    chosen = a
                    break
    sps, src, tm, decodes, aux, prim = chosen
    if len(attempts) > 1:
        ev.append(Evidence(f"Backtracking over symbol-rate hypotheses: tried "
                           + ", ".join(f"{a[0]:.3f} sps ({'CRC ok' if a[5].crc_ok else a[5].resolution})"
                                       for a in attempts), "L3_BACKTRACK", float(prim.crc_ok)))

    # ---- parameters & evidence -------------------------------------------------
    rate_src = "ESTIMATED"
    params["sps"] = _param("Samples/symbol", tm.sps, "samples", rate_src,
                           f"IF transition / power-law lines, re-modulation refined ({src})",
                           float(np.clip(tm.match, 0, 1)))
    params["symbol_rate"] = _param("Symbol rate", hz(1.0 / tm.sps), "Bd" if fs else "symbols/sample", rate_src,
                                   params["sps"].method, params["sps"].confidence)
    params["order"] = _param("Tone count M", tm.order, "", "ESTIMATED",
                             "joint (M, pulse) re-modulation match; IF-strobe M-level fits as evidence", None,
                             [Evidence(f"IF-strobe GMM log-likelihood per symbol {tm.level_fits}", "L3", 1.0)])
    params["h"] = _param("Modulation index h", tm.h, "", "ESTIMATED", "tone spacing / Rs, refined by re-modulation")
    params["tone_spacing"] = _param("Tone spacing", hz(tm.h / tm.sps), f_unit, "ESTIMATED", "h * Rs")
    params["deviation"] = _param("Peak deviation", hz((tm.order - 1) * tm.h / 2 / tm.sps), f_unit, "ESTIMATED",
                                 "(M-1) h Rs / 2")
    params["cfo"] = _param("Carrier offset", hz(band.center + tm.center / tm.sps), f_unit, "ESTIMATED",
                           "occupied-band centre + tone-grid centre")
    params["pulse"] = _param("Pulse shape", tm.pulse if tm.pulse == "rect" else f"gaussian BT={tm.bt}", "",
                             "ESTIMATED", "re-modulation match over rect / Gaussian BT hypotheses", None,
                             [Evidence(f"match per hypothesis {tm.pulse_scores}", "L3", 1.0)])
    params["continuity"] = _param("Phase continuity", "continuous (CPFSK)" if tm.continuous else "discontinuous",
                                  "", "ESTIMATED", "mean |IF| per symbol-phase bin (unfiltered)",
                                  None, [Evidence(f"peak-bin |IF| excess {tm.spike_rates[0]:.3f} cycles/sample "
                                                  f"(std error {tm.spike_rates[1]:.3f})", "L3", 1.0)])
    peaks = psd_tone_peaks(band, 0.5 * tm.h / tm.sps)
    ev.append(Evidence(f"PSD shows {len(peaks)} distinct peak(s) (tones resolve in the PSD only for larger h; "
                       f"evidence only)", "PSD_PEAKS", 1.0, {"peaks": peaks}))
    ev.append(Evidence(f"Tone grid: M={tm.order}, h={tm.h:.3f}, pulse {params['pulse'].value}, "
                       f"re-modulation match {tm.match:.3f}", "L3_TONES", float(np.clip(tm.match, 0, 1))))
    for d in decodes.values():
        ev.extend(d.evidence)

    lv = aux["levels"]
    v = aux["v"]
    err = v - lv[np.argmin(np.abs(v[:, None] - lv[None, :]), axis=1)]
    peak_dev = max((tm.order - 1) * tm.h / 2, 1e-6)
    fsk_err = float(100 * np.sqrt(np.mean(err ** 2)) / peak_dev)
    esn0 = float(10 * np.log10(tm.sps / aux["n0"]))
    params["snr"] = _param("Es/N0", esn0, "dB", "ESTIMATED", "re-modulation residual (noise per sample x sps)")

    variant = fsk_variant_name(tm.order, tm.h, tm.pulse, tm.bt, tm.continuous)
    if family.family != "FSK":
        ev.append(Evidence(f"Family resolved as FSK by demodulation evidence (sync/CRC: {prim.crc_ok}, "
                           f"re-modulation match {tm.match:.3f})" if prim.crc_ok else
                           "Family NOT confirmed as FSK: no CRC-valid frame", "FAMILY_DEMOD", float(prim.crc_ok)))
    pos = np.argmax(aux["LL"][prim.method], axis=1)
    if prim.hyp.inverted:
        pos = tm.order - 1 - pos
    return FSKReceiverResult(family, variant, tm, params, decodes, prim, fsk_err, esn0, ev, pos)
