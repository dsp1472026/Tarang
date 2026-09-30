"""
Blind square-QAM receiver (16 / 64 / 256QAM): Phase 3.

Shares L1-L3 + L5a with the PSK receiver (psk_front_end: AGC, burst, coarse CFO,
symbol rate, RRC matched filter, Oerder-Meyr + Gardner timing), then:

    L4   order: modulation/linear_classifier (ALRT + lattice fit; EVM per order,
         C40, amplitude kurtosis as evidence) unless forced
    L5a' roll-off x timing (x rate for short bursts) refinement, scored by the
         QAM lattice residual (square-grid quantisation error, scale-free)
    L5b  coarse carrier: 4th-power line (square QAM keeps a line at 4 x CFO),
         then a joint phase / residual-frequency / grid-scale fit (decision-
         directed complex LS from a grid of seeds) -- works on a few dozen symbols
    L5c  whole-burst refinement: phase / frequency / gain LS on confident decisions in growing
         centred windows; then tracking: 2nd-order decision-directed carrier loop + gain loop,
         run from the burst centre outwards in both directions, updated
         only on confident decisions (inside 0.35 d_min of a point), so decision
         errors on dense grids do not drive the loop ("class-based" DD)
    L5d  symbols next to a capture edge that cuts the signal: truncated-pulse LS
    L6   4-fold ambiguity: sync word / expected payload / text heuristic over
         4 rotations x 4 carrier aliases (Rs/4); Gray demapping per axis, max-log LLRs
PAPR is not used anywhere.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.evidence import Evidence, Parameter
from demodulation.psk import (DecodeResult, _search_bits, best_model, text_decisive,
                              best_text_region, decode_psk, remove_alias)
from demodulation.psk_receiver import (EDGE_HALF_SPAN, PSKFrontEnd, PSKReceiverConfig, _copy_param, _param,
                                       equalize_edges, psk_front_end)
from dsp.conditioning import Burst
from dsp.constellations import Constellation, inverse_gray, qam
from synthetic.framing import SYNC_BITS, bits_to_bytes, bytes_to_bits
from dsp.filters import cubic_interp, matched_filter, rrc_taps
from modulation.linear_classifier import (LinearClassification, _qam_carrier, _square_qam_nearest, classify_linear,
                                         lattice_residual)
from synchronization.symbol_timing import TimingResult

QAM_ORDERS = {"16QAM": 16, "64QAM": 64, "256QAM": 256}
ROLLOFF_CANDIDATES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)
REFINE_WINDOW = 256        # symbols used by the roll-off / timing scans
CONFIDENT = 0.35           # DD loops update only when |y - decision| < CONFIDENT * d_min


@dataclass
class QAMReceiverConfig:
    modulation: Optional[str] = None       # force 16QAM / 64QAM / 256QAM (skip L4)
    sps: Optional[float] = None
    rrc_beta: Optional[float] = None
    cfo_norm: Optional[float] = None
    expected_payload: Optional[str] = None
    carrier_loop_bw: float = 0.004         # reduced bandwidth: dense grids are phase-noise sensitive
    gain_loop_mu: float = 0.01
    edge_mode: str = "ls"                  # "ls" | "guarded" | "off": symbols next to a cut capture edge


@dataclass
class QAMReceiverResult:
    modulation: str
    classification: Optional[LinearClassification]
    parameters: Dict[str, Parameter]
    symbols: np.ndarray                    # carrier-corrected, ambiguity-resolved, unit-power symbols
    hard_bits: np.ndarray
    llrs: np.ndarray
    decode: DecodeResult
    evm: float
    esn0_est_db: float
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
        for key in ("sample_rate", "symbol_rate", "sps", "rrc_beta", "cfo", "snr"):
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


def _grid_residual(s: np.ndarray, order: int) -> float:
    """Scale-free QAM lattice residual (same test as the classifier; edge symbols excluded)."""
    return lattice_residual(s, [f"{order}QAM"])[0]


def refine_timing(x: np.ndarray, strobes: np.ndarray, order: int, sps: float, betas, span: float = 0.2,
                  points: int = 7):
    """Roll-off x common timing offset, scored by the lattice residual (rate comes from the front end)."""
    n = len(strobes)
    c = n // 2
    sub = slice(max(0, c - REFINE_WINDOW // 2), c + REFINE_WINDOW // 2)
    offsets = np.linspace(-span, span, points) * sps
    by_beta, best = {}, None
    for b in betas:
        mf = matched_filter(x, rrc_taps(b, sps, span=10))
        errs = [_grid_residual(cubic_interp(mf, strobes[sub] + d), order) for d in offsets]
        k = int(np.argmin(errs))
        by_beta[b] = float(errs[k])
        if best is None or errs[k] < best[3]:
            best = (b, mf, float(offsets[k]), float(errs[k]))
    b, mf, d, e = best
    scale = 0.0
    if n < 200:
        # Short bursts: a rate error of 0.1 % already moves the outer strobes of a 40-symbol
        # burst by 0.02 symbol, enough to flip dense-QAM decisions there. Scan the rate
        # (chosen roll-off only) about the burst centre.
        center = strobes[c]
        d0, d_best = d, d
        for eps in np.linspace(-0.006, 0.006, 25):
            base = center + (strobes - center) * (1 + eps)
            for dd in d0 + np.linspace(-0.05, 0.05, 3) * sps:          # fixed reference: no wandering
                ee = _grid_residual(cubic_interp(mf, base[sub] + dd), order)
                if ee < e - 1e-4:
                    e, scale, d_best = ee, float(eps), float(dd)
        d = d_best
        strobes = center + (strobes - center) * (1 + scale)
    return b, mf, strobes + d, d, scale, e, by_beta


AXIS_MAPS = ("gray-asc", "gray-desc", "natural-asc", "natural-desc")


def mapped_qam(order: int, i_map: str, q_map: str, q_first: bool) -> Constellation:
    """
    Square QAM with an explicit label convention (legacy captures differ): per axis, the
    half-label A (bits) selects level index gray-asc: inverse_gray(A), gray-desc:
    side-1-inverse_gray(A), natural-asc: A, natural-desc: side-1-A; q_first puts the
    Q half-label first. ("gray-asc", "gray-asc", False) == dsp.constellations.qam(order).
    """
    side = int(round(np.sqrt(order)))
    half = int(np.log2(side))
    levels = 2.0 * np.arange(side) - (side - 1)
    k = np.arange(order)
    hi, lo = k >> half, k & (side - 1)
    a_i, a_q = (lo, hi) if q_first else (hi, lo)

    def pos(a, m):
        p = inverse_gray(a) if m.startswith("gray") else a
        return side - 1 - p if m.endswith("desc") else p

    pts = levels[pos(a_i, i_map)] + 1j * levels[pos(a_q, q_map)]
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    return Constellation(name=f"{order}QAM", family="QAM", points=pts.astype(np.complex128),
                         bits_per_symbol=int(np.log2(order)), phase_symmetry=4)


def decode_mapping_hypotheses(z: np.ndarray, order: int, noise_var: float, expected_payload: Optional[str]):
    """
    No sync word: every (label mapping, carrier alias, rotation) is a hypothesis; the
    expected payload (bit errors) or the text likelihood decides. Returns
    (DecodeResult, Constellation) or (None, None) when nothing is decisive.
    Text readings are searched first at byte alignments consistent with the symbol grid
    (multiples of gcd(8, bits/symbol): the transmitted byte stream starts on a symbol
    boundary), then at every alignment if that is not decisive. Without that prior, an
    I/Q-swapped reading shifted by half a 256QAM symbol spells the same digit string
    shifted by one nibble ("8765432100 ..." for "9876543210 ...").
    """
    pat = bytes_to_bits(expected_payload.encode("ascii")) if expected_payload else None
    bps = int(round(np.log2(order)))
    hyps = []
    for q_first in (False, True):
        for i_map in AXIS_MAPS:
            for q_map in AXIS_MAPS:
                const = mapped_qam(order, i_map, q_map, q_first)
                default = (i_map, q_map, q_first) == ("gray-asc", "gray-asc", False)
                label = f"{'Q' if q_first else 'I'}-first, I {i_map}, Q {q_map}"
                for a in range(4):
                    ya = remove_alias(z, a, 4)
                    for r in range(4):
                        yr = ya * np.exp(-2j * np.pi * r / 4)
                        hyps.append((default, a, r, label, const, yr, const.demap_hard(yr)))

    def rank(step):
        scored = []
        for default, a, r, label, const, yr, bits in hyps:
            if pat is not None:
                off, err = _search_bits(bits, pat, const.bits_per_symbol)
                if off is None:
                    continue
                score = -err
            else:
                score, off = best_text_region(bits, step)
            # ties: prefer the standard mapping, then no alias / rotation
            scored.append((score, default, -min(a, 4 - a), -r, label, a, r, off, const, yr, bits))
        if not scored:
            return None
        scored.sort(key=lambda t: t[:4], reverse=True)
        best = scored[0]
        off, bits = best[7], best[10]
        runner = next((t[0] for t in scored if bits_to_bytes(t[10][t[7]:]) != bits_to_bytes(bits[off:])),
                      -np.inf)
        return best, runner, len(scored)

    steps = (1,) if pat is not None else tuple(dict.fromkeys((int(np.gcd(8, bps)), 1)))
    for step in steps:
        ranked = rank(step)
        if ranked is None:
            return None, None
        (score, _, _, _, label, a, r, off, const, yr, bits), runner, n_hyp = ranked
        signed = a if a <= 2 else a - 4
        if pat is not None:
            err = -score
            ev = Evidence(f"No sync word; label mapping [{label}], alias {signed}*Rs/4, rotation {90 * r} deg chosen "
                          f"by matching the expected payload ({err}/{len(pat)} bit errors at bit {off}; "
                          f"{n_hyp} hypotheses)", "KNOWN_PAYLOAD", 1.0 - err / len(pat))
            return DecodeResult("known-payload", r, False, bits, const.demap_soft(yr, noise_var), off, None,
                                bits_to_bytes(bits[off:off + len(pat)]), None, None, None, [ev], signed), const
        if text_decisive(score, runner, (len(bits) - off) // 8):
            ev = Evidence(f"No sync word; label mapping [{label}], alias {signed}*Rs/4, rotation {90 * r} deg chosen "
                          f"by text likelihood {score:.2f} vs runner-up {runner:.2f} nats/char, "
                          f"{best_model(bits_to_bytes(bits[off:]))[1]} message model (heuristic; {n_hyp} "
                          f"hypotheses)" + (f"; text starts at bit {off} (header skipped)" if off else ""),
                          "TEXT_HEURISTIC", 0.5)
            return DecodeResult("text-heuristic", r, False, bits, const.demap_soft(yr, noise_var), off, None,
                                bits_to_bytes(bits[off:]), None, None, None, [ev], signed), const
    return None, None


def dd_track(z: np.ndarray, const: Constellation, loop_bw: float, mu: float):
    """
    2nd-order decision-directed carrier loop + first-order gain loop at 1 sps.
    Updates only on confident decisions (|y - d| < CONFIDENT * d_min).
    """
    side = int(round(np.sqrt(const.order)))
    rms_grid = np.sqrt(2 * (const.order - 1) / 3.0)
    dmin = 2.0 / rms_grid
    damping = 0.707
    theta = loop_bw / (damping + 1.0 / (4.0 * damping))
    den = 1.0 + 2.0 * damping * theta + theta ** 2
    kp, ki = 4.0 * damping * theta / den, 4.0 * theta ** 2 / den
    out = np.empty_like(z)
    phases = np.empty(len(z))
    phi, freq, g = 0.0, 0.0, 1.0
    used = 0
    for i, v in enumerate(z):
        y = v * np.exp(-1j * phi) / g
        d = _square_qam_nearest(np.array([y * rms_grid]), side)[0] / rms_grid
        out[i] = y
        phases[i] = phi
        if abs(y - d) < CONFIDENT * dmin:
            e = float(np.imag(y * np.conj(d)) / (abs(d) ** 2 + 1e-12))
            freq += ki * e
            phi += kp * e + freq
            g *= 1.0 + mu * (abs(y) - abs(d)) / (abs(d) + 1e-12)
            used += 1
        else:
            phi += freq
    return out, phases, freq, used


def grow_fit(z: np.ndarray, const: Constellation, windows=(256, 512, 1024, None)):
    """
    Whole-burst carrier refinement after the centre-window fit: in growing centred windows,
    decide the confident symbols, then fit phase + frequency (weighted LS on the phase
    errors, amplitude-weighted) and gain (LS), and apply them to the whole burst. A residual
    frequency of 1e-4 cycles/symbol is 5 deg at the ends of a 300-symbol frame: invisible in
    a 128-symbol fit, fatal for 256QAM.
    """
    side = int(round(np.sqrt(const.order)))
    rms_grid = np.sqrt(2 * (const.order - 1) / 3.0)
    dmin = 2.0 / rms_grid
    n = len(z)
    nn = np.arange(n) - (n - 1) / 2.0
    for w in windows:
        lo, hi = (0, n) if w is None or w >= n else ((n - w) // 2, (n + w) // 2)
        seg = z[lo:hi]
        d = _square_qam_nearest(seg * rms_grid, side) / rms_grid
        ok = np.abs(seg - d) < CONFIDENT * dmin
        if ok.sum() < 8:
            continue
        th = np.angle(seg[ok] * np.conj(d[ok]))
        wgt = np.abs(d[ok]) ** 2
        t = nn[lo:hi][ok]
        A = np.stack([np.ones_like(t), t], 1) * np.sqrt(wgt)[:, None]
        (ph, fr), *_ = np.linalg.lstsq(A, th * np.sqrt(wgt), rcond=None)
        z = z * np.exp(-1j * (ph + fr * nn))
        seg = z[lo:hi]
        g = float(np.real(np.vdot(d[ok], seg[ok])) / (np.vdot(d[ok], d[ok]).real + 1e-12))
        if g > 0:
            z = z / g
        if w is None or w >= n:
            break
    return z


def dd_track_center_out(z: np.ndarray, const: Constellation, loop_bw: float, mu: float):
    """Decision-directed tracking from the burst centre outwards (both directions), where the fit is best."""
    c = len(z) // 2
    fwd, _, f1, u1 = dd_track(z[c:], const, loop_bw, mu)
    bwd, _, f2, u2 = dd_track(z[:c + 1][::-1], const, loop_bw, mu)
    out = np.concatenate([bwd[::-1][:-1], fwd])
    return out, (f1 - f2) / 2.0, u1 + u2


def sync_aided_refine(z: np.ndarray, const: Constellation, decode: DecodeResult, noise_var: float):
    """
    Sync word found but the CRC fails: on short 256QAM bursts the blind carrier fit can leave a
    residual carrier (0.5 deg/symbol measured) or a slightly wrong scale, and the outer rings then
    decide wrongly. The located sync word's known symbols give a data-aided phase + frequency
    (weighted LS line through their phase errors) and gain; the burst is corrected, optionally
    refitted decision-directed (grow_fit), and decoded again. Adopted only if the CRC then
    validates. Returns (z, DecodeResult) or (None, None).
    """
    if decode.sync_symbol_index is None:
        return None, None
    zz = remove_alias(z, decode.alias_steps, 4) * np.exp(-2j * np.pi * decode.rotation_steps / 4)
    p = const.map_bits(SYNC_BITS)
    k = int(decode.sync_symbol_index)
    seg = zz[k:k + len(p)]
    if len(seg) < len(p):
        return None, None
    # phase + frequency: weighted LS line through the sync symbols' phase errors (a residual
    # carrier of 0.5 deg/symbol left by the DD loop on a 21-symbol 256QAM burst was the cause
    # of a CRC failure), then gain by LS
    t = np.arange(len(p)) - (len(p) - 1) / 2.0
    th = np.unwrap(np.angle(seg * np.conj(p)))
    wgt = np.abs(p) ** 2
    A = np.stack([np.ones_like(t), t], 1) * np.sqrt(wgt)[:, None]
    (ph, fr), *_ = np.linalg.lstsq(A, th * np.sqrt(wgt), rcond=None)
    n_all = np.arange(len(zz)) - (k + (len(p) - 1) / 2.0)
    variants = []
    for name, slope in (("phase + frequency", fr), ("phase", 0.0)):
        zc = zz * np.exp(-1j * (ph + slope * n_all))
        g = float(np.real(np.vdot(p, zc[k:k + len(p)])) / np.vdot(p, p).real)
        if g <= 1e-9:
            continue
        zc = zc / g
        variants += [(f"sync-aided {name} + gain", zc, g, slope),
                     (f"sync-aided {name} + gain + DD refit", grow_fit(zc, const), g, slope)]
    for variant, zc, g, slope in variants:
        d = decode_psk(zc, const, noise_var, allow_differential=False)
        if d.frame is not None and d.frame.crc_ok:
            d.evidence.insert(0, Evidence(f"Sync word found but CRC failed; {variant} from the {len(p)} known "
                                          f"sync symbols (gain {g:.3f}, phase {np.degrees(ph):+.1f} deg, residual "
                                          f"{np.degrees(slope):+.2f} deg/symbol), re-decoded: CRC VALID",
                                          "SYNC_AIDED", 1.0))
            return zc, d
    return None, None


def receive_qam(samples: np.ndarray, sample_rate: Optional[float] = None,
                config: Optional[QAMReceiverConfig] = None, front: Optional[PSKFrontEnd] = None,
                classification: Optional[LinearClassification] = None) -> QAMReceiverResult:
    cfg = config or QAMReceiverConfig()
    fs = sample_rate
    hz = (lambda v: v * fs) if fs else (lambda v: v)
    f_unit = "Hz" if fs else "cycles/sample"
    if front is None:
        if cfg.sps is not None or cfg.cfo_norm is not None:
            front = psk_front_end(samples, fs, PSKReceiverConfig(sps=cfg.sps, rrc_beta=cfg.rrc_beta,
                                                                 cfo_norm=cfg.cfo_norm))
            if cfg.cfo_norm is None:                       # coarse-CFO hypotheses, lattice decides
                from modulation.unified_classifier import _alt_cfo_front_end, _lattice
                front, _ = _alt_cfo_front_end(samples, fs, front, _lattice(front.s_core)[0], evidence0 := [],
                                              sps=cfg.sps)
        else:
            # Blind: the unified classifier's front end tests coarse-CFO x symbol-rate hypotheses
            # (incl. integer sub-rates, which a constant-envelope rate score picks for QAM).
            from modulation.unified_classifier import classify_linear_capture
            ev0: List[Evidence] = []
            front, _, lin = classify_linear_capture(samples, fs, ev0)
            classification = classification or lin
    fe = front
    params = {k: _copy_param(v) for k, v in fe.params.items()}
    evidence = list(fe.evidence)
    x, spec, burst, sps, timing, idx = fe.x, fe.spec, fe.burst, fe.sps, fe.timing, fe.idx

    # ---- L4: order -----------------------------------------------------------
    if cfg.modulation:
        order = QAM_ORDERS[cfg.modulation.upper().replace("-", "")]
    else:
        classification = classification or classify_linear(fe.s_core)
        evidence.extend(classification.evidence)
        if classification.family != "QAM":
            evidence.append(Evidence(f"Classifier says {classification.modulation} (not QAM); demodulating with the "
                                     "best QAM hypothesis anyway", "L4", 0.0))
            post = {k: v for k, v in classification.posterior.items() if k in QAM_ORDERS}
            order = QAM_ORDERS[max(post, key=post.get)] if post else 16
        else:
            order = classification.order
    const = qam(order)

    # ---- L5a': roll-off x timing (x rate) refinement -------------------------
    betas = (cfg.rrc_beta,) if cfg.rrc_beta is not None else ROLLOFF_CANDIDATES
    beta, mf, strobes, d_off, rate_scale, r_lat, by_beta = refine_timing(x, timing.strobes, order, sps, betas)
    s = cubic_interp(mf, strobes)
    if rate_scale:
        sps *= 1 + rate_scale
    params["sps"].value = sps
    params["symbol_rate"].value = hz(1.0 / sps)
    params["rrc_beta"] = _param("RRC roll-off", beta, "", "KNOWN" if cfg.rrc_beta is not None else "ESTIMATED",
                                "lattice-residual scan over roll-off x timing" if cfg.rrc_beta is None else "user supplied")
    evidence.append(Evidence(f"Lattice-directed refinement: beta {beta}, timing {d_off:+.3f} samples, rate "
                             f"x(1{rate_scale:+.4f}); lattice residual {r_lat:.3f} (per beta: "
                             + ", ".join(f"{b}:{e:.3f}" for b, e in by_beta.items()) + ")", "L5_TIMING", 1.0))

    # ---- L5b: coarse carrier + scale -----------------------------------------
    s = s / np.sqrt(np.mean(np.abs(s[idx[0]:idx[-1] + 1] if len(idx) else s) ** 2))
    y, a = _qam_carrier(s, order)
    rms_grid = np.sqrt(2 * (order - 1) / 3.0)
    z = y * a / rms_grid                                  # unit-power grid

    # ---- L5c: decision-directed tracking -------------------------------------
    z = grow_fit(z, const)
    z, f_res, used = dd_track_center_out(z, const, cfg.carrier_loop_bw, cfg.gain_loop_mu)
    evidence.append(Evidence(f"4th-power line + phase/frequency/scale fit, then decision-directed carrier + gain "
                             f"loops (updated on {used}/{len(z)} confident symbols); residual "
                             f"{hz(f_res / (2 * np.pi) / sps):.3g} {f_unit}", "L5_CARRIER", used / max(len(z), 1)))

    # ---- L5d: capture edges ----------------------------------------------------
    cut_start = burst.start == 0 and strobes[0] < EDGE_HALF_SPAN * sps
    cut_end = burst.stop >= len(x) and strobes[-1] > len(x) - 1 - EDGE_HALF_SPAN * sps
    if (cut_start or cut_end) and cfg.edge_mode != "off":
        z, n_edge = equalize_edges(z, strobes, len(x), sps, beta, const, bool(cut_start), bool(cut_end),
                                   guarded=cfg.edge_mode == "guarded")
        if n_edge:
            evidence.append(Evidence(f"{n_edge} symbols next to a cut capture edge re-solved by truncated-pulse "
                                     "least squares", "L5_EDGE", 1.0))

    # ---- L6 ------------------------------------------------------------------
    z_core = z[idx[0]:idx[-1] + 1] if len(idx) else z
    evm = const.evm(z_core)
    noise_var = max(evm ** 2, 1e-5)
    decode = decode_psk(z, const, noise_var, allow_differential=False)     # sync word, standard Gray mapping
    if decode.method != "sync-word":
        d2, c2 = decode_mapping_hypotheses(z, order, noise_var, cfg.expected_payload)
        if d2 is not None:
            decode, const = d2, c2
    elif not (decode.frame and decode.frame.crc_ok):
        z2, d2 = sync_aided_refine(z, const, decode, noise_var)
        if d2 is not None:
            z, decode = z2, d2
            evm = const.evm(z[idx[0]:idx[-1] + 1] if len(idx) else z)
    evidence.extend(decode.evidence)
    if decode.alias_steps:
        z = remove_alias(z, decode.alias_steps, 4)
    if decode.method in ("sync-word", "known-payload", "text-heuristic"):
        z = z * np.exp(-2j * np.pi * decode.rotation_steps / 4)
    params["cfo"] = _param("Carrier offset", hz(spec.cfo_norm + (decode.alias_steps / 4) / sps), f_unit, "ESTIMATED",
                           "PSD -3 dB midpoint + 4th-power line + DD loop", None)
    esn0 = -20 * np.log10(max(evm, 1e-6))
    params["snr"] = _param("Es/N0 (MER)", esn0, "dB", "ESTIMATED", "decision-directed MER after sync")
    return QAMReceiverResult(const.name, classification, params, z, decode.hard_bits, decode.llrs, decode, evm,
                             esn0, burst, timing, evidence)
