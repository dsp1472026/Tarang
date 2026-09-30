"""
Linear-family classification at 1 sample/symbol: PSK vs QAM, then the order.

Input: timing-recovered symbols (PSKFrontEnd.s_core), before carrier recovery.

Hypotheses: BPSK, QPSK, 8PSK, 16PSK, 16QAM, 64QAM, 256QAM.

Checks (decision tree of the spec, applied at the symbol level where the
theoretical values hold; RRC pulse overlap distorts them at the sample level):
  1. Amplitude spread sigma_A^2 of the symbols: PSK ~ noise only, QAM has
     several rings (16QAM: normalised amplitude CV 0.32 even noiseless).
  2. |C40| (unit-power, after carrier removal): BPSK 2, QPSK 1, 8/16PSK 0,
     16QAM 0.68, 64QAM 0.62, 256QAM 0.61.
  3. Phase clusters: number of occupied angle bins of the PSK-corrected symbols.
  4. ALRT over all seven hypotheses. Every hypothesis gets its own carrier
     recovery (M-th power line + PLL) and gain, and is scored on the SAME
     unit-RMS symbols, so log-likelihoods of PSK and QAM are comparable. The
     equiprobable mixture carries the ln(1/M) prior penalty that keeps nested
     hypotheses (QPSK inside 16QAM, 8PSK inside 16PSK) from over-fitting.
The ALRT decides; the other checks are reported as agreeing/disagreeing
evidence and scale the confidence.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from core.evidence import Evidence
from dsp.constellations import CHUNK, get_constellation, psk
from modulation.psk_classifier import classify_psk
from synchronization.carrier_psk import apply_carrier, carrier_pll, mth_power_line

PSK_HYP = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16PSK": 16}
QAM_HYP = {"16QAM": 16, "64QAM": 64, "256QAM": 256}
C40_THEORY = {"BPSK": 2.0, "QPSK": 1.0, "8PSK": 0.0, "16PSK": 0.0, "16QAM": 0.68, "64QAM": 0.619, "256QAM": 0.605}
N_EFF_MAX = 100
QAM_MIN_AMP_VAR = 0.05     # sigma_A^2 of unit-mean-amplitude symbols: 16/64/256QAM >= 0.08 measured, PSK <= 0.035
RADIAL_RATIO = 3.0         # radial spread must exceed 3 x the tangential (noise) spread to be QAM
EDGE_SYMBOLS = 5           # ~ half the RRC span: strobes that may see a truncated pulse


def _dmin(name):
    p = get_constellation(name).points
    d = np.abs(p[:, None] - p[None, :])
    return float(d[d > 1e-9].min())


D_MIN = {k: _dmin(k) for k in (*PSK_HYP, *QAM_HYP)}
PSK16 = psk(16)
EVM_LIMIT = {"16QAM": 0.125, "64QAM": 0.08, "256QAM": 0.035}   # ETSI / 3GPP required EVM


def _amp_kurtosis(name):
    a = np.abs(get_constellation(name).points)
    return float(np.mean((a - a.mean()) ** 4) / np.var(a) ** 2)


AMP_KURTOSIS = {k: _amp_kurtosis(k) for k in QAM_HYP}


@dataclass
class LinearClassification:
    modulation: str
    family: str                          # "PSK" | "QAM"
    order: int
    confidence: float
    posterior: Dict[str, float]
    features: Dict[str, float]
    evidence: List[Evidence]
    corrected: Dict[str, np.ndarray] = field(repr=False, default_factory=dict)


def _mixture_loglik(z: np.ndarray, pts: np.ndarray, noise_var: float) -> float:
    total = 0.0
    for i in range(0, len(z), CHUNK):                 # (N, M) in blocks: bounded memory
        a = -np.abs(z[i:i + CHUNK, None] - pts[None, :]) ** 2 / noise_var
        amax = a.max(axis=1, keepdims=True)
        total += float(np.sum(amax[:, 0] + np.log(np.mean(np.exp(a - amax), axis=1))))
    return total / len(z) - float(np.log(np.pi * noise_var))


def _nearest(z: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = np.empty(len(z), dtype=pts.dtype)
    for i in range(0, len(z), CHUNK):
        out[i:i + CHUNK] = pts[np.argmin(np.abs(z[i:i + CHUNK, None] - pts[None, :]), axis=1)]
    return out


def _fit(z: np.ndarray, pts: np.ndarray, gain0: float, iters: int = 4):
    """Decision-directed least-squares gain; returns (gain, noise_var)."""
    g = gain0
    for _ in range(iters):
        d = _nearest(z / g, pts)
        g_new = float(np.real(np.vdot(d, z)) / (np.vdot(d, d).real + 1e-12))
        if g_new <= 0 or abs(g_new - g) < 1e-4 * g:
            break
        g = g_new
    d = _nearest(z / g, pts)
    return g, float(np.mean(np.abs(z - g * d) ** 2))


QAM_FIT_SYMBOLS = 128      # centre window used for the QAM carrier / gain grid search
PLL_MIN_SYMBOLS = 256      # longer bursts also get a decision-directed PLL (drift tracking)
QAM_MAX_RESIDUAL_CFO = 0.03   # cycles/symbol


def _square_qam_nearest(z: np.ndarray, side: int) -> np.ndarray:
    """Nearest point of the unscaled square grid (odd integer levels), per axis: O(N)."""
    def q(v):
        return np.clip(2 * np.round((v - 1) / 2) + 1, -(side - 1), side - 1)
    return q(z.real) + 1j * q(z.imag)


def _qam_carrier(s: np.ndarray, order: int):
    """
    Square-QAM carrier and gain for short bursts.

    The 4th-power line gives frequency and phase (mod 90 deg), but on a few dozen
    non-uniform QAM symbols its phase is biased by several degrees, which on 64/256QAM
    is a large fraction of the decision distance, and a PLL has no time to converge.
    So phase (+-45 deg) and grid scale are seeded on a grid, each seed is refined by a
    decision-directed complex least-squares fit (phase + scale together), and the
    residual frequency is scanned; the score is the per-axis quantisation residual
    (O(N) per evaluation) on a centre window of <= QAM_FIT_SYMBOLS symbols.
    Returns (corrected symbols, grid scale in units of the unscaled odd-integer grid).
    """
    side = int(round(np.sqrt(order)))
    y = np.asarray(s, dtype=np.complex128)
    n = len(y)
    mid = n // 2
    w = np.arange(max(0, mid - QAM_FIT_SYMBOLS // 2), min(n, mid + QAM_FIT_SYMBOLS // 2))
    yw, nn = y[w], w - (n - 1) / 2.0
    rms_grid = np.sqrt(2 * (order - 1) / 3.0)             # RMS of the odd-integer square grid
    # Residual-frequency candidates: the coarse CFO is already removed (band centre), so a
    # fine grid over +-QAM_MAX_RESIDUAL_CFO, plus a fine grid around the 4th-power line (restricted to small
    # offsets: short non-uniform payloads put data-induced 4th-power lines near Nyquist).
    line = mth_power_line(y, 4, 1 + 1j, max_freq=QAM_MAX_RESIDUAL_CFO)
    # Grid step: worst-case drift at the window edge, (step/2)(len/2) cycles, must stay inside
    # the decision region of the outermost point (~1/(2 pi sqrt(2) side) cycles).
    edge_tol = 1.0 / (2 * np.pi * np.sqrt(2) * side)
    # (x3: each seed's decision-directed slope update pulls in the remaining offset)
    step = 12 * edge_tol / max(len(w), 8)
    k = int(np.ceil(QAM_MAX_RESIDUAL_CFO / step))
    dfs = np.r_[np.arange(-k, k + 1) * step, line.freq + np.arange(-4, 5) * step]
    wts = nn - nn.mean()
    # Seed spacing scales with the decision region: the complex LS converges from about
    # +-(64/side) deg in phase, so 16QAM needs 16 deg seeds and 256QAM 4 deg.
    ph_step = 64.0 / side
    n_gain = 4 if side <= 4 else 7
    phases = np.deg2rad(np.arange(-45 + ph_step / 2, 45, ph_step))
    seeds = (rms_grid * np.linspace(0.75, 1.35, n_gain))[None, :] * np.exp(-1j * phases)[:, None]
    seeds = seeds.ravel()

    # All (phase, scale) seeds of one frequency candidate are refined together:
    # decision-directed complex LS (phase + scale) and a phase-slope (frequency) update.
    best = (np.inf, 0.0, 1.0)
    for df0 in dfs:
        c = seeds.copy()
        df = np.full(len(seeds), df0)
        for _ in range(4):
            zw = yw[None, :] * np.exp(-2j * np.pi * df[:, None] * nn[None, :])
            g = _square_qam_nearest(c[:, None] * zw, side)
            c = np.sum(np.conj(zw) * g, axis=1) / (np.sum(np.abs(zw) ** 2, axis=1) + 1e-30)
            th = np.angle(g * np.conj(c[:, None] * zw))
            wgt = np.abs(g) ** 2
            df = df + np.sum(wgt * wts * th, axis=1) / (np.sum(wgt * wts ** 2, axis=1) + 1e-30) / (2 * np.pi)
        z = c[:, None] * yw[None, :] * np.exp(-2j * np.pi * df[:, None] * nn[None, :])
        r = np.mean(np.abs(z - _square_qam_nearest(z, side)) ** 2, axis=1)
        k = int(np.argmin(r))
        if r[k] < best[0]:
            best = (float(r[k]), float(df[k]), complex(c[k]))
    _, df, c = best
    nn_all = np.arange(n) - (n - 1) / 2.0
    return y * np.exp(-2j * np.pi * df * nn_all) * np.exp(1j * np.angle(c)), abs(c)


def _score(s_unit: np.ndarray, name: str):
    """Carrier-recover s under hypothesis `name`; return (unit-RMS corrected symbols, loglik, gain, noise_var)."""
    const = get_constellation(name)
    if const.family == "QAM":
        y, a = _qam_carrier(s_unit, const.order)
        if len(y) > PLL_MIN_SYMBOLS:
            y = carrier_pll(y, const, loop_bw=0.005).symbols
        z = y / np.sqrt(np.mean(np.abs(y) ** 2))
        # y * a ~ odd-integer grid G, unit-power points P = G / rms_grid, z = y / rms(y) ~ g P
        gain0 = float(np.sqrt(2 * (const.order - 1) / 3.0) / (a * np.sqrt(np.mean(np.abs(y) ** 2))))
        seeds = [gain0]
    else:
        line = mth_power_line(s_unit, const.order, const.points[0])
        y = carrier_pll(apply_carrier(s_unit, line.freq, line.phase), const, loop_bw=0.005).symbols
        z = y / np.sqrt(np.mean(np.abs(y) ** 2))
        seeds = [float(np.mean(np.abs(z)))]
    best = None
    for g0 in seeds:
        g, nv = _fit(z, const.points, g0)
        nv = max(nv, 1e-4)
        ll = _mixture_loglik(z, g * const.points, nv)
        if best is None or ll > best[1]:
            best = (z, ll, g, nv)
    return best


def lattice_residual(s: np.ndarray, hyps=None):
    """
    Best normalised lattice fit over the hypotheses: rms error / (gain * d_min).
    ~0 when the strobes sit on a PSK/QAM lattice (right rate and timing), ~0.3 for
    scattered strobes. Returns (residual, hypothesis).
    """
    s = _trim(np.asarray(s, dtype=np.complex128))
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    best = (np.inf, None)
    for k in hyps or (*PSK_HYP, *QAM_HYP):
        z, ll, g, nv = _score(s, k)
        best = min(best, (float(np.sqrt(nv) / (g * D_MIN[k])), k))
    return best


def _trim(s: np.ndarray) -> np.ndarray:
    """Drop EDGE_SYMBOLS at each end: captures cut mid-pulse (e.g. 'same' convolution) distort them."""
    e = min(EDGE_SYMBOLS, len(s) // 10)
    return s[e:len(s) - e] if e else s


def classify_linear(s_core: np.ndarray) -> LinearClassification:
    s = _trim(np.asarray(s_core, dtype=np.complex128))
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    n = len(s)
    evidence: List[Evidence] = []
    feats: Dict[str, float] = {}

    hyps = {**PSK_HYP, **QAM_HYP}
    scores = {k: _score(s, k) for k in hyps}

    # 1. sigma_A^2 decides PSK vs QAM (decision tree), measured against the noise: on the
    #    finest PSK grid (16PSK) circular noise spreads the symbols equally in the radial
    #    and tangential directions, while QAM rings add radial spread only.
    z16 = scores["16PSK"][0]
    z16 = z16 / np.mean(np.abs(z16))
    ref = PSK16.points[PSK16.nearest_labels(z16)]
    radial = np.abs(z16) - 1.0
    tangential = np.abs(z16) * np.sin(np.angle(z16 * np.conj(ref)))
    feats["amp_var"] = float(np.var(np.abs(z16)))
    feats["amp_cv"] = float(np.std(np.abs(s)) / np.mean(np.abs(s)))
    feats["radial_var"] = float(np.mean(radial ** 2))
    feats["tangential_var"] = float(np.mean(tangential ** 2))
    is_qam = feats["amp_var"] > QAM_MIN_AMP_VAR and feats["radial_var"] > RADIAL_RATIO * feats["tangential_var"]
    fam = "QAM" if is_qam else "PSK"
    evidence.append(Evidence(
        f"Symbol amplitude variance sigma_A^2 {feats['amp_var']:.4f}; radial / tangential spread about the "
        f"16PSK grid {feats['radial_var']:.4f} / {feats['tangential_var']:.4f}: "
        + ("several amplitude levels -> QAM" if is_qam else "constant envelope up to noise -> PSK"),
        "SYM_ENVELOPE", 1.0, {"amp_var": feats["amp_var"], "radial": feats["radial_var"],
                              "tangential": feats["tangential_var"]}))

    # ALRT within the family chosen by sigma_A^2
    fam_hyps = QAM_HYP if is_qam else PSK_HYP
    ll = np.array([scores[k][1] for k in fam_hyps])
    logits = min(n, N_EFF_MAX) * (ll - ll.max())
    post = np.exp(logits) / np.exp(logits).sum()
    posterior = {k: float(p) for k, p in zip(fam_hyps, post)}
    winner = max(posterior, key=posterior.get)

    # 2. |C40| on the winner's corrected symbols
    z = scores[winner][0]
    m20, m40 = np.mean(z ** 2), np.mean(z ** 4)
    feats["|C40|"] = float(np.abs(m40 - 3 * m20 ** 2))
    feats["C42"] = float(np.mean(np.abs(z) ** 4) - np.abs(m20) ** 2 - 2)
    near = min(C40_THEORY, key=lambda k: abs(C40_THEORY[k] - feats["|C40|"]))
    c40_ok = abs(C40_THEORY[winner] - feats["|C40|"]) < 0.35 or near == winner
    evidence.append(Evidence(
        f"Symbol cumulants |C40|={feats['|C40|']:.2f} C42={feats['C42']:.2f}; theory for {winner} "
        f"|C40|={C40_THEORY[winner]:.2f} (nearest theory: {near})", "CUMULANTS", float(c40_ok),
        {"|C40|": feats["|C40|"]}))

    # 3b. QAM order features (reported; the ALRT decides): EVM per order with the ETSI/3GPP
    #     limits, amplitude kurtosis vs each order's theoretical value. PAPR is not used.
    if fam == "QAM":
        evm = {k: float(np.sqrt(scores[k][3]) / scores[k][2]) for k in QAM_HYP}
        rule = next((k for k in QAM_HYP if evm[k] < EVM_LIMIT[k]), None)
        a = np.abs(scores[winner][0])
        kurt = float(np.mean((a - a.mean()) ** 4) / (np.var(a) ** 2 + 1e-12))
        near_k = min(AMP_KURTOSIS, key=lambda k: abs(AMP_KURTOSIS[k] - kurt))
        feats.update({f"evm_{k}": v for k, v in evm.items()})
        feats["amp_kurtosis"] = kurt
        evidence.append(Evidence(
            "EVM per order: " + ", ".join(f"{k} {100 * v:.1f}% (limit {100 * EVM_LIMIT[k]:.1f}%)" for k, v in evm.items())
            + f" -> EVM rule: {rule or 'none within limits'}", "QAM_EVM", float(rule == winner),
            {"evm": evm}))
        evidence.append(Evidence(
            f"Amplitude kurtosis {kurt:.2f} (theory " + ", ".join(f"{k} {v:.2f}" for k, v in AMP_KURTOSIS.items())
            + f"; nearest {near_k}; short or non-uniform payloads shift it)", "QAM_KURTOSIS", float(near_k == winner),
            {"kurtosis": kurt}))

    # 3. phase clusters (PSK hypotheses) / amplitude rings (QAM)
    if fam == "PSK":
        order = PSK_HYP[winner]
        ang = np.angle(scores[winner][0]) * order / (2 * np.pi)
        visited = np.unique(np.round(ang).astype(int) % order)
        feats["phase_clusters"] = float(len(visited))
        # payloads can leave points unvisited; clusters beyond M/2 rule out the next lower order
        ok = order == 2 or len(visited) > order // 2
        evidence.append(Evidence(f"Phase clusters: {len(visited)} of {order} visited", "PHASE_CLUSTERS",
                                 float(ok), {"visited": len(visited)}))
    else:
        order = QAM_HYP[winner]

    # 4. ALRT
    evidence.append(Evidence(f"ALRT within {fam}: " + ", ".join(f"{k}={v:.3f}" for k, v in posterior.items()
                                                                if v > 1e-3), "ALRT", posterior[winner],
                             {"loglik_per_symbol": dict(zip(fam_hyps, ll.tolist()))}))

    # The existing PSK classifier's M-th power line check (its within-PSK decision is the same ALRT).
    if fam == "PSK":
        pc = classify_psk(s)
        line_ev = next(e for e in pc.evidence if e.source == "M_POWER_LINE")
        evidence.append(line_ev)

    checks = np.mean([e.score for e in evidence if e.source != "ALRT"])
    conf = float(posterior[winner] * (0.5 + 0.5 * checks))
    return LinearClassification(winner, fam, order, conf, posterior, feats, evidence,
                                {k: v[0] for k, v in scores.items()})
