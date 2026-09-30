"""
Blind PSK receiver: L1 conditioning -> L2 burst detection -> L3 parameter
estimation -> L4 classification -> L5 synchronisation -> L6 demodulation.

The only external inputs are the samples and (optionally) the sample rate.
Every parameter is returned as a core.evidence.Parameter with its provenance,
so nothing is silently assumed.

    L1  block AGC; DC estimated from noise-only samples (payload mean may be != 0)
    L2  channel-filtered energy burst detection (noise floor, active region, SNR)
    L3  coarse CFO (-3 dB band midpoint); symbol rate: -3 dB bandwidth and
        |x|^2 cyclic lines as hypotheses, scored by strobe compactness;
        initial roll-off from a raised-cosine PSD fit
    L5a RRC matched filter, Oerder-Meyr timing + rate refinement, Gardner loop
    L4  PSK order: M-th power lines + ALRT + cumulants + occupancy + envelope
    L5a' decision-directed scan of roll-off x timing (x rate for short bursts)
    L5b carrier: M-th power feedforward (freq + phase), Costas / DD PLL,
        decision-directed residual DC
    L6  sync word / differential / known payload / text heuristic, each over
        all Rs/M carrier aliases; Gray demapping, max-log LLRs, frame/CRC
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.evidence import Evidence, Parameter
from demodulation.psk import DecodeResult, decode_psk, remove_alias
from dsp.conditioning import Burst, channel_filter, detect_burst, estimate_dc, normalize_power
from dsp.constellations import Constellation, get_constellation
from dsp.filters import cubic_interp, matched_filter, rrc_pulse, rrc_taps
from estimation.linear_mod_params import estimate_rolloff, estimate_spectrum, select_symbol_rate
from modulation.psk_classifier import PSKClassification, classify_psk
from synchronization.carrier_psk import apply_carrier, carrier_pll, mth_power_line
from synchronization.symbol_timing import TimingResult, feedforward_symbols, gardner_loop, oerder_meyr


@dataclass
class PSKReceiverConfig:
    modulation: Optional[str] = None       # force BPSK/QPSK/8PSK (skip L4) — otherwise blind
    sps: Optional[float] = None            # known samples/symbol (else estimated)
    rrc_beta: Optional[float] = None       # known roll-off (else estimated)
    cfo_norm: Optional[float] = None       # coarse carrier offset, cycles/sample (else -3 dB band midpoint)
    refine_sps: bool = False               # treat a supplied sps as a hypothesis: Oerder-Meyr refines it
    expected_payload: Optional[str] = None # only used for legacy captures with no sync word
    timing_loop: bool = True               # Gardner tracking after Oerder-Meyr initialisation
    timing_loop_bw: float = 0.004
    carrier_loop_bw: float = 0.01


@dataclass
class PSKReceiverResult:
    modulation: str
    classification: Optional[PSKClassification]
    parameters: Dict[str, Parameter]
    symbols: np.ndarray                 # carrier-corrected, ambiguity-resolved, unit-power symbols
    hard_bits: np.ndarray
    llrs: np.ndarray
    decode: DecodeResult
    evm: float                          # RMS EVM (fraction)
    esn0_est_db: float                  # from MER
    burst: Burst
    timing: TimingResult
    evidence: List[Evidence] = field(default_factory=list)

    @property
    def payload_text(self) -> str:
        return self.decode.text

    def summary(self) -> str:
        p = self.parameters
        lines = [f"Modulation      : {self.modulation}"
                 + (f"  (confidence {self.classification.confidence:.1%})" if self.classification else " (forced)")]
        for key in ("sample_rate", "symbol_rate", "sps", "rrc_beta", "cfo", "snr", "bit_mapping"):
            if key in p:
                q = p[key]
                val = f"{q.value:.6g}" if isinstance(q.value, (int, float)) and q.value is not None else str(q.value)
                lines.append(f"{q.name:16s}: {val} {q.unit}  [{q.source}; {q.method}]")
        lines.append(f"EVM             : {100 * self.evm:.2f} %  (MER-derived Es/N0 {self.esn0_est_db:.1f} dB)")
        lines.append(f"Ambiguity       : {self.decode.method}")
        if self.decode.frame:
            lines.append(f"Frame           : len={self.decode.frame.length} B, "
                         f"CRC {'VALID' if self.decode.frame.crc_ok else 'FAIL'}")
        lines.append(f"Payload         : {self.payload_text!r}")
        return "\n".join(lines)


ROLLOFF_CANDIDATES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)
EST_WINDOW = 32768          # samples used by the L3 estimators (bounded cost on long captures)
DD_MAX_SYMBOLS = 2048       # symbols used by the decision-directed scans


def _dd_evm(mf: np.ndarray, t: np.ndarray, const: Constellation) -> float:
    """EVM of strobes t after M-th power feedforward carrier removal (no loop)."""
    s = cubic_interp(mf, t)
    a = np.abs(s)
    s = s[a > 0.5 * np.median(a)]
    s = s / np.mean(np.abs(s))
    line = mth_power_line(s, const.order, const.points[0])
    return const.evm(apply_carrier(s, line.freq, line.phase))


@dataclass
class DDRefinement:
    beta: float
    offset: float          # samples added to every strobe
    rate_scale: float      # strobes stretched by (1 + rate_scale) about the burst centre
    evm: float
    evm_by_beta: Dict[float, float]
    mf: np.ndarray
    strobes: np.ndarray


def refine_dd(x: np.ndarray, strobes: np.ndarray, const: Constellation, sps: float, betas,
              span: float = 0.1, points: int = 21) -> DDRefinement:
    """
    Decision-directed refinement of roll-off, timing phase and (short bursts) rate.

    1. For each candidate roll-off, matched-filter and scan a common strobe
       offset over +-span symbols; score = EVM after feedforward carrier removal.
       This removes the self-noise bias of Oerder-Meyr on payloads with few
       transitions and picks the matched filter with the least ISI.
    2. Bursts shorter than 200 symbols also get a rate scan: the cyclic-line
       rate estimate on ~30 symbols is only good to ~1 %, which drifts the last
       strobes by a quarter symbol. (Longer bursts are tracked by the Gardner loop.)
    """
    n = len(strobes)
    c = strobes[n // 2]
    sub = strobes[max(0, n // 2 - DD_MAX_SYMBOLS // 2):n // 2 + DD_MAX_SYMBOLS // 2]
    offsets = np.linspace(-span, span, points) * sps
    evm_by_beta, best = {}, None
    for b in betas:
        mf = matched_filter(x, rrc_taps(b, sps, span=10))
        errs = [_dd_evm(mf, sub + d, const) for d in offsets]
        k = int(np.argmin(errs))
        evm_by_beta[b] = float(errs[k])
        if best is None or errs[k] < best[3]:
            best = (b, mf, float(offsets[k]), float(errs[k]))
    b, mf, d, e = best

    scale = 0.0
    if n < 200:
        step = 0.02 / n                       # <= 0.02 symbol of end-to-end drift per step
        for eps in np.arange(-0.012, 0.012 + step / 2, step):
            base = c + (strobes - c) * (1 + eps)
            for dd in offsets:
                ee = _dd_evm(mf, base + dd, const)
                if ee < e:
                    e, d, scale = ee, float(dd), float(eps)
    new = c + (strobes - c) * (1 + scale) + d
    return DDRefinement(b, d, scale, e, evm_by_beta, mf, new)


def remove_dc_dd(y: np.ndarray, const: Constellation, theta: np.ndarray, idx: np.ndarray, iters: int = 2):
    """
    Decision-directed DC estimate in the symbol domain.

    Model: y_k = a_k + D*exp(-j*theta_k) + n_k, where theta_k is the total
    carrier phase removed from symbol k (a raw-domain constant is rotated by it).
    D = mean((y_k - a_hat_k) * exp(j*theta_k)). Kept only if EVM improves by >2 %,
    so payloads with a genuine non-zero mean are never "corrected".
    """
    sl = slice(idx[0], idx[-1] + 1) if len(idx) else slice(None)
    evm0 = const.evm(y[sl])
    z, total = y.copy(), 0j
    for _ in range(iters):
        r = z[sl] - const.points[const.nearest_labels(z[sl])]
        d = complex(np.mean(r * np.exp(1j * theta[sl])))
        z = z - d * np.exp(-1j * theta)
        total += d
    return (z, total) if const.evm(z[sl]) < 0.98 * evm0 else (y, None)


EDGE_HALF_SPAN = 5        # symbols: RRC half-span; strobes this close to a cut capture edge see truncated pulses


def _edge_response(t_all: np.ndarray, rows: np.ndarray, cols: np.ndarray, lo: int, hi: int,
                   sps: float, beta: float) -> np.ndarray:
    """
    C[k, j]: matched-filter output at strobe t_k caused by a unit symbol at t_j, when the
    transmitted signal only exists for samples lo <= n < hi (the capture cuts the pulses).
    """
    n = np.arange(lo, hi, dtype=float)
    taps = rrc_taps(beta, sps, span=10)
    C = np.empty((len(rows), len(cols)), dtype=np.complex128)
    for c, j in enumerate(cols):
        xj = rrc_pulse((n - t_all[j]) / sps, beta).astype(np.complex128)
        mf = matched_filter(xj, taps)
        C[:, c] = cubic_interp(mf, t_all[rows] - lo)
    return C


def equalize_edges(y: np.ndarray, strobes: np.ndarray, n_samples: int, sps: float, beta: float,
                   const: Constellation, cut_start: bool, cut_end: bool, guarded: bool = False):
    """
    Symbols within EDGE_HALF_SPAN of a capture edge that cuts the signal (e.g. a file written
    with 'same' convolution, which drops the outer half of the first/last pulses) are
    attenuated and pulled towards their neighbours; 8PSK/16PSK decisions fail there first.
    Model y_edge = C a with the exact truncated-pulse response C (normalised to a full
    pulse), take the neighbours' decisions as known, and solve the edge symbols by least
    squares. Returns (y, number of symbols re-solved).
    """
    y = y.copy()
    n_fixed = 0
    full = _edge_response(np.array([0.0]), np.array([0]), np.array([0]),
                          -int(12 * sps), int(12 * sps), sps, beta)[0, 0]    # untruncated peak
    ctx = 2 * EDGE_HALF_SPAN + 1
    for side, cut in (("start", cut_start), ("end", cut_end)):
        if not cut or len(y) < 3 * ctx:
            continue
        if side == "start":
            edge = np.flatnonzero(strobes < EDGE_HALF_SPAN * sps)
            known = np.arange(len(edge), min(len(y), len(edge) + ctx))
            lo, hi = 0, int(min(n_samples, strobes[known[-1]] + 6 * sps))
        else:
            edge = np.flatnonzero(strobes > n_samples - 1 - EDGE_HALF_SPAN * sps)
            known = np.arange(max(0, edge[0] - ctx), edge[0]) if len(edge) else np.array([], int)
            lo, hi = int(max(0, strobes[known[0]] - 6 * sps)) if len(known) else 0, n_samples
        if len(edge) == 0 or len(known) == 0:
            continue
        rows = edge
        C_e = _edge_response(strobes, rows, edge, lo, hi, sps, beta) / full
        C_k = _edge_response(strobes, rows, known, lo, hi, sps, beta) / full
        a_known = const.points[const.nearest_labels(y[known])]
        rhs = y[rows] - C_k @ a_known
        a_edge, *_ = np.linalg.lstsq(C_e, rhs, rcond=None)
        # Keep the LS values only if they sit closer to the constellation than the originals:
        # a capture edge that was filtered (e.g. resampled WAV) is not a clean pulse cut.
        dist = lambda v: float(np.sum(np.abs(v - const.points[const.nearest_labels(v)]) ** 2))
        if not guarded or dist(a_edge) < dist(y[edge]):
            y[edge] = a_edge
            n_fixed += len(edge)
    return y, n_fixed


def _param(name, value, unit, source, method, conf=None, ev=None) -> Parameter:
    return Parameter(name, value, unit, source, method, conf, ev or [])


def _copy_param(p: Parameter) -> Parameter:
    return Parameter(p.name, p.value, p.unit, p.source, p.method, p.confidence, list(p.evidence))


def _psk_strobe_quality(xb: np.ndarray, beta: float = 0.35):
    """
    Symbol-rate hypothesis score for PSK: amplitude CV of strobed symbols.

    At the right rate and timing, PSK strobes are constant-envelope (CV ~ noise).
    At a wrong rate, some strobes land between symbols and the CV grows. Only the
    outer 5 % of strobes are dropped (edge transients) — low-amplitude strobes
    must NOT be masked because they are exactly what exposes a wrong rate.
    """
    def quality(sps: float) -> float:
        mf = matched_filter(xb, rrc_taps(beta, sps, span=10))
        om = oerder_meyr(mf, sps, refine=False)
        s = feedforward_symbols(mf, om, 0, len(mf)).symbols
        e = max(1, len(s) // 20)
        a = np.abs(s[e:-e]) if len(s) > 4 * e else np.abs(s)
        return float(np.std(a) / (np.mean(a) + 1e-12))
    return quality


@dataclass
class PSKFrontEnd:
    """L1-L3 + L5a output: everything before classification (shared with the unified classifier)."""
    x: np.ndarray                   # AGC'd, DC- and coarse-CFO-corrected samples
    spec: object                    # SpectrumEstimate
    burst: Burst
    params: Dict[str, Parameter]
    evidence: List[Evidence]
    sps: float
    beta: float
    timing: TimingResult
    idx: np.ndarray                 # indices of steady-state symbols
    s_core: np.ndarray              # timing-recovered symbols (steady part), before carrier recovery


def psk_front_end(samples: np.ndarray, sample_rate: Optional[float] = None,
                  config: Optional[PSKReceiverConfig] = None) -> PSKFrontEnd:
    cfg = config or PSKReceiverConfig()
    fs = sample_rate
    params: Dict[str, Parameter] = {}
    evidence: List[Evidence] = []
    hz = (lambda v: v * fs) if fs else (lambda v: v)
    f_unit = "Hz" if fs else "cycles/sample"

    params["sample_rate"] = _param("Sample rate", fs, "Hz", "KNOWN" if fs else "UNKNOWN",
                                   "metadata" if fs else "not recoverable from samples alone")

    # ---- L1/L2: AGC, band isolation, burst detection, DC removal -------------
    x = np.asarray(samples, dtype=np.complex128)
    x, _ = normalize_power(x)
    spec0 = estimate_spectrum(x)
    burst = detect_burst(x, spec0.cfo_norm, spec0.bw3_norm)
    dc, dc_method = estimate_dc(x, burst)
    x, _ = normalize_power(x - dc)
    evidence.append(Evidence(f"Block AGC to unit power; DC {abs(dc):.3g} (rel. RMS) removed using {dc_method}",
                             "L1", 1.0))
    evidence.append(Evidence(
        "Signal occupies the whole capture (no noise-only region)" if burst.whole_file else
        f"Burst at samples {burst.start}-{burst.stop}; in-band SNR {burst.snr_db:.1f} dB from power split",
        "L2_BURST", 1.0))

    # ---- L3: parameter estimation --------------------------------------------
    # Estimators run on at most EST_WINDOW samples at the burst centre; the
    # tracking loops (Gardner, PLL) then process the whole burst.
    mid = (burst.start + burst.stop) // 2
    w0 = max(burst.start, mid - EST_WINDOW // 2)
    w1 = min(burst.stop, w0 + EST_WINDOW)
    spec = estimate_spectrum(x[w0:w1])
    if cfg.cfo_norm is not None:
        spec = dataclasses.replace(spec, cfo_norm=float(cfg.cfo_norm))
    n = np.arange(len(x))
    x = x * np.exp(-2j * np.pi * spec.cfo_norm * n)
    xb = x[w0:w1]

    if cfg.sps:
        sps = float(cfg.sps)
        params["sps"] = _param("Samples/symbol", sps, "samples", "KNOWN", "user supplied")
    else:
        # Cyclic analysis on a channel-filtered copy: the symbol-rate line comes from the
        # roll-off band (|f| <= (1+beta)Rs/2 <= Rs ~ bw3), so a +-bw3 filter keeps it intact
        # while removing out-of-band noise (~10*log10(sps/2) dB at low SNR). If the payload
        # is strongly coloured, bw3 underestimates Rs and the filter would remove the line,
        # so lines of the unfiltered burst are candidates too.
        x_cyc = channel_filter(xb, 0.0, spec.bw3_norm, scale=1.0)
        sr, hyps = select_symbol_rate(x_cyc, spec.bw3_norm, _psk_strobe_quality(xb), extra_sources=(xb,))
        sps = sr.sps
        tested = ", ".join(f"{h.sps:.3f}({h.source}: CV {h.quality:.3f})" for h in hyps)
        params["sps"] = _param("Samples/symbol", sps, "samples", "ESTIMATED",
                               "-3 dB bandwidth + cyclic lines, hypotheses scored by strobe compactness",
                               sr.confidence,
                               [Evidence(f"-3 dB BW suggests {1 / spec.bw3_norm:.3f} sps; tested {tested}; "
                                         f"chose {sr.sps:.4f} sps (highest rate within tolerance of best)",
                                         "L3", sr.confidence)])

    if cfg.rrc_beta is not None:
        beta = cfg.rrc_beta
        params["rrc_beta"] = _param("RRC roll-off", beta, "", "KNOWN", "user supplied")
    else:
        # Initial matched filter from the PSD fit. The fit assumes white symbols and
        # drifts towards beta = 1 on coloured payloads, so the final roll-off is
        # chosen decision-directed after classification (refine_dd).
        ro = estimate_rolloff(xb, sps)
        beta = ro.beta
        params["rrc_beta"] = _param("RRC roll-off", beta, "", "ESTIMATED", "raised-cosine PSD least-squares fit",
                                    float(np.clip(1 - 10 * ro.fit_residual, 0, 1)))

    # ---- L5a: matched filter + timing ----------------------------------------
    mf = matched_filter(x, rrc_taps(beta, sps, span=10))
    om = oerder_meyr(mf[w0:w1], sps, refine=cfg.sps is None or cfg.refine_sps)
    om.tau = (om.tau + w0) % om.sps
    sps = om.sps
    if cfg.timing_loop:
        timing = gardner_loop(mf, om, burst.start, burst.stop, loop_bw=cfg.timing_loop_bw)
    else:
        timing = feedforward_symbols(mf, om, burst.start, burst.stop)
    s = timing.symbols
    evidence.append(Evidence(f"Oerder-Meyr timing: line strength {om.line_strength:.2f}, tau={om.tau:.2f} samples; "
                             f"{timing.method} produced {len(s)} symbols"
                             + (f", TED error RMS {np.sqrt(np.mean(timing.ted_error ** 2)):.3f}"
                                if timing.method == "gardner" else ""), "L5_TIMING", 1.0,
                             {"om_line_strength": om.line_strength, "tau": om.tau}))

    params["symbol_rate"] = _param("Symbol rate", hz(1.0 / sps), "Bd" if fs else "symbols/sample",
                                   params["sps"].source, "cyclic line + Oerder-Meyr refinement",
                                   params["sps"].confidence)
    params["sps"].value = sps

    # Symbols in the steady part of the burst (drop MF transients at the edges) for L4.
    amp = np.abs(s)
    core = amp > 0.5 * np.median(amp)
    idx = np.flatnonzero(core)
    s_core = s[idx[0]:idx[-1] + 1] if len(idx) else s
    return PSKFrontEnd(x, spec, burst, params, evidence, sps, beta, timing, idx, s_core)


def receive_psk(samples: np.ndarray, sample_rate: Optional[float] = None,
                config: Optional[PSKReceiverConfig] = None, front: Optional[PSKFrontEnd] = None) -> PSKReceiverResult:
    """front: a PSKFrontEnd already computed for these samples/config (skips L1-L5a)."""
    cfg = config or PSKReceiverConfig()
    fs = sample_rate
    hz = (lambda v: v * fs) if fs else (lambda v: v)
    f_unit = "Hz" if fs else "cycles/sample"
    fe = front or psk_front_end(samples, sample_rate, cfg)
    x, spec, burst, sps, beta, timing, idx, s_core = (fe.x, fe.spec, fe.burst, fe.sps, fe.beta, fe.timing,
                                                      fe.idx, fe.s_core)
    timing = copy.copy(timing)                       # strobes/symbols are reassigned below
    params = {k: _copy_param(v) for k, v in fe.params.items()}
    evidence = list(fe.evidence)

    # ---- L4: classification --------------------------------------------------
    classification = None
    if cfg.modulation:
        const: Constellation = get_constellation(cfg.modulation)
    else:
        classification = classify_psk(s_core)
        const = get_constellation(classification.modulation)
        evidence.extend(classification.evidence)

    # ---- L5a': decision-directed refinement (needs the constellation) --------
    betas = (beta,) if cfg.rrc_beta is not None else ROLLOFF_CANDIDATES
    dd = refine_dd(x, timing.strobes, const, sps, betas)
    mf, timing.strobes = dd.mf, dd.strobes
    timing.symbols = cubic_interp(mf, timing.strobes)
    s = timing.symbols
    if cfg.rrc_beta is None:
        params["rrc_beta"].evidence.append(Evidence(
            f"PSD fit {beta}; decision-directed EVM per beta: "
            + ", ".join(f"{b}:{100 * e:.1f}%" for b, e in dd.evm_by_beta.items()), "L5", 1.0))
        params["rrc_beta"].value = dd.beta
        params["rrc_beta"].method = "decision-directed min-EVM over candidates (PSD fit as initial value)"
    evidence.append(Evidence(f"Decision-directed refinement: timing {dd.offset:+.3f} samples, rate x(1{dd.rate_scale:+.4f}),"
                             f" beta {dd.beta}", "L5_TIMING", 1.0))
    if dd.rate_scale:
        sps *= 1 + dd.rate_scale
        params["sps"].value = sps
        params["symbol_rate"].value = hz(1.0 / sps)
        params["symbol_rate"].method += " + decision-directed rate scan"

    # ---- L5b: carrier recovery ----------------------------------------------
    s = s / np.sqrt(np.mean(np.abs(s[idx[0]:idx[-1] + 1] if len(idx) else s) ** 2))
    line = mth_power_line(s, const.order, const.points[0])
    y = apply_carrier(s, line.freq, line.phase)
    pll = carrier_pll(y, const, loop_bw=cfg.carrier_loop_bw)
    y = pll.symbols
    y_core = y[idx[0]:idx[-1] + 1] if len(idx) else y
    y = y / np.mean(np.abs(y_core))                  # PSK: constant envelope -> unit radius

    # Residual DC, decision-directed: a raw-domain DC d appears here as
    # g*d*exp(-j*theta_k), theta_k being the total phase removed from symbol k.
    nn = np.arange(len(y)) - (len(y) - 1) / 2.0
    theta = 2 * np.pi * spec.cfo_norm * timing.strobes + 2 * np.pi * line.freq * nn + line.phase + pll.phase_track
    y, dc_dd = remove_dc_dd(y, const, theta, idx)
    if dc_dd is not None:
        evidence.append(Evidence(f"Decision-directed residual DC {abs(dc_dd):.3f} (rel. symbol amplitude) removed",
                                 "L5_DC", 1.0))
    total_cfo = spec.cfo_norm + (line.freq + pll.freq_final / (2 * np.pi)) / sps
    params["cfo"] = _param("Carrier offset", hz(total_cfo), f_unit, "ESTIMATED",
                           f"PSD -3 dB midpoint + {const.order}th-power line + {pll.detector} PLL",
                           float(line.rho))
    evidence.append(Evidence(f"{const.order}th-power line rho={line.rho:.2f}; coarse CFO {hz(spec.cfo_norm):.4g} "
                             f"{f_unit}, fine {hz(line.freq / sps):.4g} {f_unit}; {pll.detector} PLL residual "
                             f"{hz(pll.freq_final / (2 * np.pi) / sps):.3g} {f_unit}", "L5_CARRIER", line.rho))

    # Symbols next to a capture edge that cuts the signal: truncated-pulse least squares.
    cut_start = burst.start == 0 and len(timing.strobes) and timing.strobes[0] < EDGE_HALF_SPAN * sps
    cut_end = burst.stop >= len(x) and len(timing.strobes) and timing.strobes[-1] > len(x) - 1 - EDGE_HALF_SPAN * sps
    if cut_start or cut_end:
        y, n_edge = equalize_edges(y, timing.strobes, len(x), sps, dd.beta, const, bool(cut_start), bool(cut_end))
        if n_edge:
            evidence.append(Evidence(f"Capture cuts the signal at the {'start' if cut_start else ''}"
                                     f"{' and ' if cut_start and cut_end else ''}{'end' if cut_end else ''}: "
                                     f"{n_edge} edge symbols re-solved by truncated-pulse least squares",
                                     "L5_EDGE", 1.0))

    # ---- L6: demodulation ----------------------------------------------------
    y_core = y[idx[0]:idx[-1] + 1] if len(idx) else y
    evm = const.evm(y_core)
    noise_var = max(evm ** 2, 1e-4)
    decode = decode_psk(y, const, noise_var, expected_payload=cfg.expected_payload)
    evidence.extend(decode.evidence)
    if decode.alias_steps:
        y = remove_alias(y, decode.alias_steps, const.order)
        total_cfo += decode.alias_steps / const.order / sps
        params["cfo"].value = hz(total_cfo)
        params["cfo"].method += " + sync-word alias resolution"
    if decode.method in ("sync-word", "known-payload", "text-heuristic"):
        y = y * np.exp(-2j * np.pi * decode.rotation_steps / const.order)
    esn0 = -20 * np.log10(max(evm, 1e-6))
    params["snr"] = _param("Es/N0 (MER)", esn0, "dB", "ESTIMATED", "decision-directed MER after sync")

    # QPSK and 4QAM are the same constellation; only the bit labels differ. Square-QAM labels
    # (first bit = I sign, second = Q sign) identify the generator's convention as 4QAM.
    name = "4QAM" if const.order == 4 and decode.mapping == "gray-cw" else const.name
    if decode.mapping != "gray-ccw":
        params["bit_mapping"] = _param("Bit labels", decode.mapping, "", "ESTIMATED",
                                       "label-mapping hypotheses scored by payload / text likelihood")
    return PSKReceiverResult(name, classification, params, y, decode.hard_bits, decode.llrs, decode,
                             evm, esn0, burst, timing, evidence)
