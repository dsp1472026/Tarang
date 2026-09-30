"""
L4 modulation analysis within the PSK family (BPSK / QPSK(=4QAM) / 8PSK / 16PSK).

Operates on 1-sample/symbol output of timing recovery (before carrier
recovery). PAPR is deliberately NOT used: it is dominated by pulse-shaping
overshoot and is what made the legacy classifier call 16/64QAM "256QAM".

Evidence per hypothesis M in {2, 4, 8, 16}:
  1. M-th power spectral line rho_M. M-PSK data vanish under s^M, so a strong
     line at M with no line at M/2 identifies the order. Invariant to carrier
     frequency/phase and to data statistics (ASCII payloads are not uniform).
  2. Average-likelihood ratio test (ALRT): after carrier removal under
     hypothesis M, the symbols are scored against an equiprobable M-point
     Gaussian mixture. Nested orders are handled naturally: a BPSK signal fits
     8PSK too, but pays a ln(8/2) per-symbol prior penalty.
  3. Normalised cumulants |C20|, |C40|, C42 compared to theory.
  4. Constellation occupancy: fraction of the M points actually visited.

Decision = ALRT winner; the other checks are reported as agreeing/disagreeing
evidence and feed the confidence.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from core.evidence import Evidence
from dsp.constellations import CHUNK, psk
from synchronization.carrier_psk import apply_carrier, carrier_pll, mth_power_line

HYPOTHESES = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16PSK": 16}

# Theoretical normalised cumulants for unit-power, equiprobable constellations.
CUMULANT_THEORY = {
    "BPSK": {"|C20|": 1.0, "|C40|": 2.0, "C42": -2.0},
    "QPSK": {"|C20|": 0.0, "|C40|": 1.0, "C42": -1.0},
    "8PSK": {"|C20|": 0.0, "|C40|": 0.0, "C42": -1.0},
    "16PSK": {"|C20|": 0.0, "|C40|": 0.0, "C42": -1.0},
}


def cumulants(s: np.ndarray) -> Dict[str, float]:
    x = s - s.mean()
    x = x / np.sqrt(np.mean(np.abs(x) ** 2))
    m20 = np.mean(x ** 2)
    m40 = np.mean(x ** 4)
    m42 = np.mean(np.abs(x) ** 4)
    c40 = m40 - 3 * m20 ** 2
    c42 = m42 - np.abs(m20) ** 2 - 2.0
    return {"|C20|": float(np.abs(m20)), "|C40|": float(np.abs(c40)), "C42": float(np.real(c42))}


@dataclass
class HypothesisScore:
    name: str
    order: int
    rho: float
    freq: float
    loglik_per_symbol: float
    evm: float
    occupancy: float
    noise_var: float
    corrected: np.ndarray = field(repr=False)


@dataclass
class PSKClassification:
    modulation: str
    order: int
    confidence: float
    posterior: Dict[str, float]
    scores: Dict[str, HypothesisScore]
    features: Dict[str, float]
    evidence: List[Evidence]
    is_psk: bool


def _mixture_loglik(y: np.ndarray, points: np.ndarray, noise_var: float) -> float:
    total = 0.0
    for i in range(0, len(y), CHUNK):                 # (N, M) in blocks: bounded memory
        a = -np.abs(y[i:i + CHUNK, None] - points[None, :]) ** 2 / noise_var
        amax = a.max(axis=1, keepdims=True)
        total += float(np.sum(amax[:, 0] + np.log(np.mean(np.exp(a - amax), axis=1))))
    return total / len(y) - float(np.log(np.pi * noise_var))


def score_hypothesis(s: np.ndarray, name: str) -> HypothesisScore:
    """Carrier-correct `s` assuming M-PSK and score how well M-PSK explains it."""
    order = HYPOTHESES[name]
    const = psk(order)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    line = mth_power_line(s, order, const.points[0])
    y = apply_carrier(s, line.freq, line.phase)
    y = carrier_pll(y, const, loop_bw=0.005).symbols
    # Fix the gain on the mean amplitude (PSK is constant-envelope), then fit noise.
    y = y / np.mean(np.abs(y))
    labels = const.nearest_labels(y)
    err = y - const.points[labels]
    noise_var = float(np.mean(np.abs(err) ** 2)) + 1e-6
    counts = np.bincount(labels, minlength=order)
    expected = len(y) / order
    occupancy = float(np.mean(counts >= max(1.0, 0.2 * expected)))
    return HypothesisScore(name, order, line.rho, line.freq, _mixture_loglik(y, const.points, noise_var),
                           float(np.sqrt(noise_var)), occupancy, noise_var, y)


def classify_psk(s: np.ndarray) -> PSKClassification:
    n = len(s)
    scores = {name: score_hypothesis(s, name) for name in HYPOTHESES}
    amp = np.abs(s)
    feats = {}
    feats["amp_cv"] = float(np.std(amp) / np.mean(amp))
    for name, sc in scores.items():
        feats[f"rho{sc.order}"] = sc.rho

    # --- ALRT posterior ------------------------------------------------------
    # Per-symbol log-likelihoods are scaled by an effective sample count capped at
    # 100 so a long capture cannot produce a meaningless 1 - 1e-300 confidence.
    ll = np.array([scores[k].loglik_per_symbol for k in HYPOTHESES])
    n_eff = min(n, 100)
    logits = n_eff * (ll - ll.max())
    post = np.exp(logits) / np.exp(logits).sum()
    posterior = {k: float(p) for k, p in zip(HYPOTHESES, post)}
    winner = max(posterior, key=posterior.get)
    order = HYPOTHESES[winner]

    # --- Independent checks ---------------------------------------------------
    evidence = []
    line_floor = max(0.25, 3.0 / np.sqrt(n))           # random-data line level for n symbols
    rho = {m: scores[k].rho for k, m in HYPOTHESES.items()}
    # "No line at M/2": below the random-data floor, or clearly weaker than the line at M.
    # (Short captures of non-uniform data, e.g. 30 ASCII symbols, leave residual lines ~0.4.)
    line_ok = rho[order] >= line_floor and (order == 2 or rho[order // 2] < max(line_floor, 0.6 * rho[order]))
    evidence.append(Evidence(
        "M-th power lines " + " ".join(f"rho{m}={r:.2f}" for m, r in rho.items()) + "; "
        f"{'line at M=' + str(order) + ' with none at M/2' if line_ok else 'line pattern inconsistent with ' + winner}",
        "M_POWER_LINE", float(line_ok), {"rho": rho, "floor": line_floor}))

    # Cumulants on the winner's own carrier-corrected symbols: its residual frequency
    # alias (a multiple of Rs/M) leaves every order-M statistic unchanged.
    feats.update(cumulants(scores[winner].corrected))
    theory = CUMULANT_THEORY
    dist = {k: float(np.sqrt(sum((feats[c] - v) ** 2 for c, v in theory[k].items()))) for k in theory}
    cum_winner = min(dist, key=dist.get)
    evidence.append(Evidence(
        f"Cumulants |C20|={feats['|C20|']:.2f} |C40|={feats['|C40|']:.2f} C42={feats['C42']:.2f}; "
        f"nearest theory {cum_winner} (d={dist[cum_winner]:.2f})",
        "CUMULANTS", float(cum_winner == winner), {"distance": dist}))

    sc = scores[winner]
    evidence.append(Evidence(
        f"Constellation clustering: {winner} occupancy {sc.occupancy:.0%}, EVM {100 * sc.evm:.1f}%",
        "CONSTELLATION", float(sc.occupancy >= 0.75), {"occupancy": sc.occupancy, "evm": sc.evm}))

    # Constant envelope: amplitude CV of PSK symbols ~ noise only. QAM (Phase 3)
    # has an intrinsic CV >= 0.2 even noiseless, so a large CV argues against PSK.
    expected_cv = sc.evm / np.sqrt(2)
    is_psk = feats["amp_cv"] < max(0.2, 2.0 * expected_cv)
    evidence.append(Evidence(
        f"Amplitude CV {feats['amp_cv']:.3f} ({'constant envelope' if is_psk else 'NOT constant envelope'})",
        "ENVELOPE", float(is_psk), {"amp_cv": feats["amp_cv"]}))

    evidence.append(Evidence(
        f"ALRT posterior: " + ", ".join(f"{k}={v:.3f}" for k, v in posterior.items()),
        "ALRT", posterior[winner], {"loglik_per_symbol": dict(zip(HYPOTHESES, ll.tolist()))}))

    checks = np.mean([e.score for e in evidence[:4]])
    confidence = float(posterior[winner] * (0.5 + 0.5 * checks))
    return PSKClassification(winner, order, confidence, posterior, scores, feats, evidence, bool(is_psk))
