"""
Unified blind modulation classification: FSK family vs PSK vs QAM, then the order.

Decision tree (DSP only):

    x[n] -> A[n], f[n] (timing-free features, modulation/signal_features.py)
      |
      +-- FSK vs linear: log-odds from independent checks
      |     envelope   noise-corrected envelope CV (constant envelope -> FSK/CPM)
      |     cumulants  sample-level |C20|, |C40| > 0.15 -> linear (FSK tones rotate
      |                x^2, x^4 away from DC; a biased linear payload or BPSK does not)
      |     IF kurt.   amplitude-gated IF kurtosis: bounded multi-tone IF is platykurtic
      |     IF hist.   >= 2 peaks with a valley at 0 Hz -> FSK; single peak at 0 -> linear
      |     cyclic     symmetric line pair in x or x^2 at +-tone -> FSK
      |     phase hist 2/4/8 phase clusters -> linear, otherwise -> FSK (weak, sample level)
      |     amp. hist  one amplitude level -> FSK, >= 3 -> linear/QAM (weak, sample level)
      |
      +-- FSK: tone count from the IF histogram as a hint; the FSK receiver fits
      |        M / h / pulse / BT / continuity by re-modulation and names the variant
      |
      +-- linear: PSK front end (AGC, burst, rate, RRC MF, Gardner) -> 1-sps symbols
            lattice check: do the strobes sit on a PSK/QAM lattice? If not and the
            burst is short, test coarse-CFO hypotheses x a constellation-agnostic
            rate scan and keep the one whose strobes fit a lattice best
            -> sigma_A^2 (PSK vs QAM), |C40|, phase clusters, ALRT over
               BPSK/QPSK/8PSK/16PSK/16QAM/64QAM/256QAM

Two textbook rules were NOT adopted as stated because they fail on this data
(see signal_features.py): "IF kurtosis > 3 -> FSK" is inverted for pulse-shaped
PSK/QAM (IF spikes at envelope nulls make them highly kurtotic), and "IF peak
count >= 2 -> FSK" fires on half of the PSK captures (symbol-transition IF lobes).
Both enter only as weak, gated evidence.
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.evidence import Evidence
from demodulation.psk_receiver import PSKFrontEnd, PSKReceiverConfig, psk_front_end
from dsp.conditioning import detect_burst, normalize_power
from dsp.filters import cubic_interp, matched_filter, rrc_taps
from dsp.conditioning import channel_filter
from estimation.fsk_estimator import estimate_rate, inst_freq, occupied_band
from estimation.linear_rate_check import ring_residual, scan_rate, strobes
from modulation.fsk_classifier import classify_family
from modulation.linear_classifier import LinearClassification, classify_linear, lattice_residual
from modulation.signal_features import SignalFeatures, compute_features
from dsp.histogram_features import histogram_evidence

LATTICE_OK = 0.15          # normalised lattice residual of a clean constellation
MIN_LATTICE_SYMBOLS = 20  # fewer strobes fit almost any dense lattice: the residual is meaningless
X4_MIN_SAMPLES = 1200      # the x^4 line needs ~150+ symbols; on short ASCII bursts data lines dominate
X4_MIN_RHO = 0.05
X4_MIN_DIFF = 0.002        # cycles/sample
LATTICE_WINDOW = 128       # symbols used by the lattice test (bounded cost on long bursts)
MOMENT_LINEAR = 0.15       # sample-level |C20| or |C40| above this is not FSK (FSK max 0.09 measured)
PURE_TONE_RHO = 0.85       # x^2 / x^4 single-line strength of BPSK / QPSK (>= 0.9), FSK <= 0.63 measured
CONST_ENVELOPE_EXCESS = 0.12   # linear signal this flat has rectangular pulses (RRC PSK >= 0.19 measured)
FSK_MIN_H = 0.05           # an "FSK" fit with tone spacing below this x Rs is a single carrier with phase jumps
HIST_MIN_SAMPLES = 64      # shorter bursts: too few samples to count histogram modes


@dataclass
class UnifiedClassification:
    family: str                              # "FSK" | "PSK" | "QAM"
    modulation: str                          # e.g. "QPSK", "16QAM"; "FSK" until the FSK receiver names it
    confidence: float
    p_fsk: float
    features: Dict[str, float]
    evidence: List[Evidence]
    signal: SignalFeatures = field(repr=False, default=None)
    linear: Optional[LinearClassification] = field(repr=False, default=None)
    front: Optional[PSKFrontEnd] = field(repr=False, default=None)
    front_config: Optional[PSKReceiverConfig] = field(repr=False, default=None)
    fsk_tone_hint: Optional[int] = None


def _sigmoid(v: float) -> float:
    return float(1.0 / (1.0 + np.exp(-v)))


def fsk_vs_linear(samples: np.ndarray):
    """Returns (log-odds FSK, SignalFeatures, evidence, features)."""
    sf = compute_features(samples)
    x, _ = normalize_power(np.asarray(samples, dtype=np.complex128))
    band0 = occupied_band(x)
    burst = detect_burst(x, band0.center, band0.bw99)
    xb = x[burst.start:burst.stop]
    noise = None if burst.whole_file else float(np.mean(np.abs(np.r_[x[:burst.start], x[burst.stop:]]) ** 2))
    fam = classify_family(xb, occupied_band(xb), noise)

    ev: List[Evidence] = []
    terms: Dict[str, float] = {}

    # 1. envelope (calibrated: FSK excess <= 0.19, pulse-shaped PSK >= 0.19, QAM >= 0.34)
    terms["envelope"] = float(np.clip((0.19 - fam.excess_cv) / 0.02, -6, 6))
    ev.append(Evidence(f"Envelope: CV {fam.env_cv:.3f}, noise-corrected excess {fam.excess_cv:.3f} "
                       f"-> {'constant' if fam.excess_cv < 0.19 else 'varying'} envelope "
                       f"(log-odds FSK {terms['envelope']:+.1f})", "FAM_ENVELOPE",
                       _sigmoid(abs(terms["envelope"])), {"excess_cv": fam.excess_cv}))

    # 2. cumulants / moments
    mom = max(sf.c20, sf.c40)
    terms["cumulants"] = -4.0 if mom > MOMENT_LINEAR else 0.0
    ev.append(Evidence(f"Sample cumulants |C20|={sf.c20:.2f} |C40|={sf.c40:.2f} C42={sf.c42:.2f}: "
                       + ("x^2/x^4 keep a DC term -> linear modulation" if mom > MOMENT_LINEAR else
                          "no DC term in x^2/x^4 (FSK, or linear with balanced data): neutral")
                       + f" (log-odds {terms['cumulants']:+.1f})", "FAM_CUMULANTS", 1.0 if mom > MOMENT_LINEAR else 0.5,
                       {"|C20|": sf.c20, "|C40|": sf.c40, "C42": sf.c42}))

    # 3. IF kurtosis (amplitude-gated)
    k = sf.if_kurtosis_gated
    terms["if_kurtosis"] = 1.0 if k < 2.5 else (-1.0 if k > 3.5 else 0.0)
    ev.append(Evidence(f"IF kurtosis: gated {k:.2f} (raw {sf.if_kurtosis_raw:.1f}); "
                       + ("bounded multi-tone IF" if k < 2.5 else "heavy-tailed IF" if k > 3.5 else "inconclusive")
                       + f" (log-odds {terms['if_kurtosis']:+.1f})", "FAM_IF_KURTOSIS", 0.6,
                       {"gated": k, "raw": sf.if_kurtosis_raw}))

    # 4. IF histogram peaks
    p, c0 = sf.if_peaks, sf.if_center_density
    terms["if_histogram"] = 1.0 if (p >= 2 and c0 < 0.5) else (-1.0 if (p == 1 and c0 > 0.8) else 0.0)
    ev.append(Evidence(f"IF histogram ({len(sf.if_hist)} bins): {p} peak(s) at "
                       + ", ".join(f"{v / max(sf.bw, 1e-9):+.2f}" for v in sf.if_peak_freqs)
                       + f" x BW, density at 0 Hz {c0:.2f} of max"
                       + f" (log-odds {terms['if_histogram']:+.1f})", "FAM_IF_HISTOGRAM", 0.6,
                       {"peaks": p, "center_density": c0}))

    # 5. cyclic tone pair
    pair = any(sf.tone_pairs.values())
    terms["cyclic"] = 1.5 if pair else 0.0
    ev.append(Evidence("Cyclic lines: " + (f"symmetric tone pair in x^{[k for k, v in sf.tone_pairs.items() if v][0]}"
                                           " -> FSK tones" if pair else "no symmetric tone pair (neutral)")
                       + f" (log-odds {terms['cyclic']:+.1f})", "FAM_CYCLIC", 0.6 if pair else 0.5,
                       {"pairs": sf.tone_pairs}))

    # 6. M-th power pure tone: x^2 (BPSK) or x^4 (QPSK) collapses to ONE carrier line whatever the
    #    pulse shape, also for rectangular-pulse (constant-envelope) PSK that the envelope test
    #    calls FSK. FSK gives at most a symmetric pair there (max single-line rho 0.63 measured).
    pl = max(sf.power_line.values()) if sf.power_line else 0.0
    kpl = max(sf.power_line, key=sf.power_line.get) if sf.power_line else 0
    terms["power_line"] = -8.0 if pl >= PURE_TONE_RHO else 0.0
    ev.append(Evidence(f"M-th power line: strongest single line rho={pl:.2f} in x^{kpl} "
                       + (f"-> x^{kpl} is a pure carrier: {'B' if kpl == 2 else 'Q'}PSK (any pulse shape)"
                          if pl >= PURE_TONE_RHO else "(no pure carrier line: neutral)")
                       + f" (log-odds {terms['power_line']:+.1f})", "FAM_POWER_LINE", 1.0 if pl >= PURE_TONE_RHO else 0.5,
                       {"rho": sf.power_line}))

    # 7./8. phase / amplitude histograms (dsp/histogram_features.py) on the burst with the band
    #    centre removed. Sample-level counts are unreliable on pulse-shaped signals, so both stay
    #    weak. Short bursts or bad samples: no terms, no evidence (behaviour unchanged).
    y = xb * np.exp(-2j * np.pi * sf.band_center * np.arange(len(xb)))
    try:
        hev = histogram_evidence(y, 1.0) if len(y) >= HIST_MIN_SAMPLES else []
    except ValueError:
        hev = []
    for h in hev:
        if h["pillar"] == "FAM_PHASE_HIST":
            terms["phase_hist"] = -1.0 if h["details"]["power_of_two"] else 1.0
        else:
            na = h["details"]["amplitude_level_count"]
            terms["amp_hist"] = 0.5 if na == 1 else (-0.5 if na >= 3 else 0.0)
        t = terms["phase_hist" if h["pillar"] == "FAM_PHASE_HIST" else "amp_hist"]
        ev.append(Evidence(h["description"] + f" (log-odds FSK {t:+.1f})", h["pillar"], h["score"], h["details"]))

    logit = float(sum(terms.values()))
    feats = {**sf.as_dict(), "excess_cv": fam.excess_cv, **{f"logodds_{a}": b for a, b in terms.items()},
             "logodds_fsk": logit}
    return logit, sf, ev, feats


def _cfo_candidates(samples: np.ndarray, front: PSKFrontEnd) -> List[float]:
    x, _ = normalize_power(np.asarray(samples, dtype=np.complex128))
    xb = x[front.burst.start:front.burst.stop]
    cands = [front.spec.cfo_norm, occupied_band(xb).center, 0.0]
    X = np.abs(np.fft.fft(xb, 8192)) ** 2
    f = np.fft.fftfreq(8192)
    cands.append(float(np.sum(f * X) / np.sum(X)))                     # power centroid
    for k in (2, 4):                                                     # x^k spectral lines
        z = xb ** k / (np.abs(xb) ** (k - 1) + 1e-12)
        nfft = int(2 ** np.ceil(np.log2(max(16 * len(z), 4096))))
        S = np.abs(np.fft.fft(z, nfft))
        fk = np.fft.fftfreq(nfft)
        S[np.abs(fk) > k * 0.05] = 0
        cands.append(float(fk[int(np.argmax(S))] / k))
    out: List[float] = []
    for c in cands:
        if all(abs(c - o) > 0.0015 for o in out):
            out.append(c)
    return out


def _lattice(s):
    """Lattice residual (central LATTICE_WINDOW symbols), or +inf with too few symbols to mean anything."""
    if len(s) < MIN_LATTICE_SYMBOLS:
        return float("inf"), f"only {len(s)} symbols"
    if len(s) > LATTICE_WINDOW:
        c = len(s) // 2
        s = s[c - LATTICE_WINDOW // 2:c + LATTICE_WINDOW // 2]
    return lattice_residual(s)


REFINE_BETAS = (0.25, 0.35, 0.5)
REFINE_TOP = 3             # candidates that get the roll-off x timing refinement
CFO_SCAN_TOP, CFO_SCAN_SPAN, CFO_SCAN_STEP = 2, 0.012, 0.002   # cycles/sample
CFO_SCAN_MAX = 4           # distinct rates micro-scanned


def _refine(fe: PSKFrontEnd, r_base: float):
    """
    Roll-off x timing refinement of the strobes, scored by the lattice fit.

    The front end's roll-off comes from a PSD fit, which on a few dozen coloured
    symbols can land at 1.0; for dense QAM the resulting ISI alone spoils the lattice.
    Returns (residual, front end with refined s_core, hypothesis).
    """
    t = fe.timing.strobes[fe.idx[0]:fe.idx[-1] + 1] if len(fe.idx) else fe.timing.strobes
    best = (r_base, fe, None)
    for b in sorted({*REFINE_BETAS, round(float(fe.beta), 3)}):
        mf = matched_filter(fe.x, rrc_taps(b, fe.sps, span=10))
        for d in np.linspace(-0.15, 0.15, 3) * fe.sps:
            s = cubic_interp(mf, t + d)
            r, h = _lattice(s)
            if r < best[0]:
                timing = copy.copy(fe.timing)
                timing.strobes = fe.timing.strobes + d          # keep the receiver on the refined timing
                best = (r, dataclasses.replace(fe, s_core=s, beta=b, timing=timing), h)
    return best


def _pick(cands):
    """
    Best candidate by lattice residual, then the sub-rate rule: strobing at k x the true
    samples/symbol still lands on valid symbols (every k-th), so a candidate whose sps is
    an integer fraction 1/k of the best one's and that still fits a clean lattice is the
    truth and the best one is its sub-rate. Candidates: (residual, front end, cfg, label).
    """
    best = min(cands, key=lambda c: c[0])
    tol = max(LATTICE_OK, 1.25 * best[0] + 0.02)
    faster = [c for c in cands if c[0] <= tol and c[1].sps < best[1].sps / 1.5
              and abs(best[1].sps / c[1].sps - round(best[1].sps / c[1].sps)) < 0.03 * best[1].sps / c[1].sps]
    return min(faster, key=lambda c: c[1].sps) if faster else best


def _phase_jump_rate(samples: np.ndarray) -> float:
    """
    Symbol rate of a constant-envelope (rectangular-pulse) PSK signal, in samples/symbol.

    With flat +-1 pulses, strobing at ANY multiple of the true rate lands on valid
    constellation points and |x|^2 has almost no cyclic line, so the PSK front end
    cannot find the rate. The only rate information is WHEN the phase jumps happen:
    the IF-transition line estimator of the FSK receiver (phase-discontinuous mode)
    measures exactly that.
    """
    x, _ = normalize_power(np.asarray(samples, dtype=np.complex128))
    band = occupied_band(x)
    burst = detect_burst(x, band.center, band.bw99)
    n = np.arange(len(x))
    y = channel_filter(x * np.exp(-2j * np.pi * band.center * n), 0.0, band.bw99, 0.6)
    f = inst_freq(y)
    r = estimate_rate(y[burst.start:burst.stop], f[burst.start:burst.stop - 1], band.bw99, continuous=False)
    return float(r.sps)


def fourth_power_cfo(samples: np.ndarray):
    """
    Coarse CFO from the x^4 spectral line (square QAM and QPSK keep a line at 4 x CFO).
    Returns (cfo cycles/sample, line strength rho, burst length) on the detected burst.
    """
    x, _ = normalize_power(np.asarray(samples, dtype=np.complex128))
    band = occupied_band(x)
    burst = detect_burst(x, band.center, band.bw99)
    xb = x[burst.start:burst.stop]
    z = np.abs(xb) * np.exp(4j * np.angle(xb))
    nfft = int(2 ** np.ceil(np.log2(max(8 * len(z), 4096))))
    S = np.abs(np.fft.fft(z, nfft)) / (np.sum(np.abs(z)) + 1e-30)
    f = np.fft.fftfreq(nfft)
    k = int(np.argmax(S))
    return float(f[k] / 4.0), float(S[k]), len(xb)


def band_edge_cfo(samples: np.ndarray, level_db: float = -10.0) -> float:
    """
    Coarse CFO = midpoint of the OUTERMOST PSD bins above level_db (relative to the peak).
    The -3 dB estimator grows the band outward from the highest bin; a periodic payload puts
    narrow data lines in the spectrum, and when the highest bin is such a line the "band"
    collapses onto it (0.047 cycles/sample observed for a true 0.0016). The band edges at
    -10 dB are set by the pulse roll-off, not by the data.
    """
    from scipy.signal import medfilt, welch
    x, _ = normalize_power(np.asarray(samples, dtype=np.complex128))
    band = occupied_band(x)
    burst = detect_burst(x, band.center, band.bw99)
    xb = x[burst.start:burst.stop]
    nseg = int(min(256, 2 ** np.floor(np.log2(max(len(xb) // 4, 16)))))
    f, p = welch(xb, nperseg=nseg, noverlap=nseg // 2, return_onesided=False, detrend=False)
    f, p = np.fft.fftshift(f), medfilt(np.fft.fftshift(p), 3)
    m = np.flatnonzero(10 * np.log10(p / p.max() + 1e-30) >= level_db)
    return float((f[m[0]] + f[m[-1]]) / 2.0)


def _alt_cfo_front_end(samples, fs, fe, r0, ev, sps=None):
    """
    Coarse-CFO hypotheses besides the -3 dB midpoint: the -10 dB band-edge midpoint and, on
    long bursts, the x^4 line (square QAM / QPSK). Each is tested with the lattice fit.
    """
    hyps = [("-10 dB band edges", band_edge_cfo(samples))]
    cfo4, rho4, n = fourth_power_cfo(samples)
    if n >= X4_MIN_SAMPLES and rho4 >= X4_MIN_RHO:
        hyps.append((f"x^4 line (rho {rho4:.2f})", cfo4))
    best, txt = (fe, r0, "-3 dB midpoint"), [f"-3 dB midpoint {fe.spec.cfo_norm:+.4f}: {r0:.3f}"]
    for name, c in hyps:
        if abs(c - fe.spec.cfo_norm) < X4_MIN_DIFF:
            continue
        f2 = psk_front_end(samples, fs, PSKReceiverConfig(cfo_norm=c, sps=sps))
        r2, _ = _lattice(f2.s_core)
        txt.append(f"{name} {c:+.4f}: {r2:.3f}")
        if r2 < best[1]:
            best = (f2, r2, name)
    if len(txt) > 1:
        ev.append(Evidence("Coarse CFO hypotheses (cycles/sample: lattice residual): " + "; ".join(txt)
                           + f" -> {best[2]}", "LIN_CFO", 1.0))
    return best[0], best[1]


def classify_linear_capture(samples: np.ndarray, fs: Optional[float], ev: List[Evidence],
                            constant_envelope: bool = False):
    cfg = PSKReceiverConfig()
    if constant_envelope:
        cfg.sps = _phase_jump_rate(samples)
        ev.append(Evidence(f"Constant-envelope PSK (rectangular pulses): symbol rate from the phase-jump "
                           f"(IF transition) lines: {cfg.sps:.3f} samples/symbol", "LIN_RATE", 1.0))
    fe = psk_front_end(samples, fs, cfg)
    r0, h0 = _lattice(fe.s_core)
    if not constant_envelope:
        fe, r0 = _alt_cfo_front_end(samples, fs, fe, r0, ev)
        cfg = PSKReceiverConfig(cfo_norm=fe.spec.cfo_norm) if fe.spec.cfo_norm != cfg.cfo_norm else cfg
    ev.append(Evidence(f"Front end: {fe.sps:.3f} sps, {len(fe.s_core)} symbols; lattice residual {r0:.3f} "
                       f"(best fit {h0}; < {LATTICE_OK} = clean constellation)", "LIN_LATTICE",
                       float(r0 < LATTICE_OK), {"residual": r0}))
    if r0 <= LATTICE_OK:
        return fe, cfg, classify_linear(fe.s_core)
    # Cheap first: the rate may be right and only the PSD-fit roll-off / timing off.
    r1, fe1, _ = _refine(fe, r0)
    if r1 <= LATTICE_OK:
        ev.append(Evidence(f"Roll-off/timing refinement of the front-end estimate: lattice residual {r1:.3f} "
                           f"(beta {fe1.beta})", "LIN_LATTICE", 1.0, {"residual": r1}))
        return fe1, cfg, classify_linear(fe1.s_core)

    # No clean lattice (or too few symbols to test one): test coarse-CFO x
    # symbol-rate hypotheses, including integer sub-rates of the front-end rate.
    tried: List[str] = []
    cands = [(r0, fe, cfg, "front end")]

    def test(cfo, sps, label):
        # Oerder-Meyr rate refinement helps long bursts (a 0.1 % error drifts hundreds of strobes)
        # but drifts to data-induced lines on short ones, so only >= 200 symbols.
        c = PSKReceiverConfig(cfo_norm=cfo, sps=sps, refine_sps=len(samples) / sps >= 200)
        f = psk_front_end(samples, fs, c)
        r, h = _lattice(f.s_core)
        tried.append(f"{label} CFO {cfo:+.4f}, {sps:.3f} sps: {r:.3f} ({h})")
        cands.append((r, f, c, f"{label}: CFO {cfo:+.4f} cycles/sample, {sps:.3f} sps"))

    # Integer sub-rates of the front-end rate, under every coarse-CFO hypothesis (the -3 dB
    # midpoint can be far off on coloured payloads, and a matched filter that far off-centre
    # blurs the rings). The front-end rate is itself off by a few %: sps/k is fine-tuned on
    # the ring fit first.
    cfo_hyps = [fe.spec.cfo_norm]
    for c in (band_edge_cfo(samples), fourth_power_cfo(samples)[0]):
        if all(abs(c - h) > X4_MIN_DIFF for h in cfo_hyps):
            cfo_hyps.append(c)
    for c in cfo_hyps:
        fc = fe if c == fe.spec.cfo_norm else psk_front_end(samples, fs, PSKReceiverConfig(cfo_norm=c, sps=fe.sps))
        xb0 = fc.x[fc.burst.start:fc.burst.stop][:1024]
        for k in range(2, 9):
            if fe.sps / k >= 2.2:
                grid = fe.sps / k * np.exp(np.linspace(-0.04, 0.04, 161))
                sk = float(grid[int(np.argmin([ring_residual(strobes(xb0, g))[0] for g in grid]))])
                test(c, sk, f"front-end rate / {k}")
    # Early exit: an integer sub-rate of the front-end estimate that fits a clean lattice
    # (after roll-off/timing refinement) needs no further search.
    for i in np.argsort([c[0] for c in cands])[:2]:
        r_c, f_c, c_c, label = cands[i]
        r_f, f_f, _ = _refine(f_c, r_c)
        if r_f < r_c:
            cands[i] = (r_f, f_f, c_c, label + f" + roll-off/timing refinement (beta {f_f.beta})")
    pick = _pick(cands)
    if pick[1] is not fe and pick[0] <= LATTICE_OK:
        ev.append(Evidence("No clean lattice at the front-end rate: hypotheses tested [" + "; ".join(tried)
                           + f"] -> adopted {pick[3]} (residual {pick[0]:.3f}, {len(pick[1].s_core)} symbols)",
                           "LIN_HYPOTHESES", 1.0))
        return pick[1], pick[2], classify_linear(pick[1].s_core)

    for cfo in _cfo_candidates(samples, fe):
        f1 = psk_front_end(samples, fs, PSKReceiverConfig(cfo_norm=cfo))
        xb = f1.x[f1.burst.start:f1.burst.stop]
        for j, sps in enumerate(scan_rate(xb, 2.2, 64.0).candidates):   # whole range: -3 dB width unreliable
            test(cfo, sps, "ring-best rate" if j == 0 else "rate scan")

    # roll-off x timing refinement of the most promising candidates
    for i in np.argsort([c[0] for c in cands])[:REFINE_TOP]:
        r_c, f_c, c_c, label = cands[i]
        r_f, f_f, _ = _refine(f_c, r_c)
        tried.append(f"refined [{label}]: {r_f:.3f}")
        if r_f < r_c:
            cands[i] = (r_f, f_f, c_c, label + f" + roll-off/timing refinement (beta {f_f.beta})")

    # Still no clean lattice: micro-scan the coarse CFO around the two best candidates
    # (the QAM fit itself only searches +-0.03 cycles/symbol of residual frequency).
    if min(c[0] for c in cands) >= LATTICE_OK:
        # around the best candidates AND each CFO hypothesis's ring-best rate (a right rate with a
        # slightly wrong coarse CFO fits no lattice, so it is not among the lowest residuals)
        seeds = sorted(cands, key=lambda c: c[0])[:CFO_SCAN_TOP] + [c for c in cands if c[3].startswith("ring-best")]
        uniq = []
        for c in seeds:
            if c[2].sps is not None and all(abs(c[2].sps / u[2].sps - 1) > 0.01 for u in uniq):
                uniq.append(c)
        for _, f_c, c_c, _ in uniq[:CFO_SCAN_MAX]:
            if c_c.sps is None:
                continue
            for dc in np.arange(-CFO_SCAN_SPAN, CFO_SCAN_SPAN + 1e-9, CFO_SCAN_STEP):
                if abs(dc) < 1e-9:
                    continue
                c3 = PSKReceiverConfig(cfo_norm=(c_c.cfo_norm or 0.0) + dc, sps=c_c.sps, rrc_beta=f_c.beta)
                f3 = psk_front_end(samples, fs, c3)
                r3, _ = _lattice(f3.s_core)
                cands.append((r3, f3, c3, f"CFO micro-scan {c3.cfo_norm:+.4f} cycles/sample, {c3.sps:.3f} sps"))
        tried.append(f"CFO micro-scan +-{CFO_SCAN_SPAN} around {len(uniq[:CFO_SCAN_MAX])} rates: "
                     f"{min(c[0] for c in cands):.3f}")

    pick = _pick(cands)
    adopt = pick[1] is not fe and pick[0] < LATTICE_OK
    ev.append(Evidence("No clean lattice: hypotheses tested [" + "; ".join(tried) + "] -> "
                       + (f"adopted {pick[3]} (residual {pick[0]:.3f}, {len(pick[1].s_core)} symbols)" if adopt
                          else "no clean lattice found: kept the front-end estimate"),
                       "LIN_HYPOTHESES", 1.0 if adopt else 0.5))
    fe, cfg = (pick[1], pick[2]) if adopt else (fe, cfg)
    return fe, cfg, classify_linear(fe.s_core)


def classify_capture(samples: np.ndarray, sample_rate: Optional[float] = None) -> UnifiedClassification:
    ev: List[Evidence] = []
    logit, sf, fev, feats = fsk_vs_linear(samples)
    ev.extend(fev)
    p_fsk = _sigmoid(logit)
    tone_hint = sf.if_peaks if sf.if_peaks >= 2 else None
    if logit > 0:
        ev.append(Evidence(f"Family: FSK (P = {p_fsk:.3f}); IF histogram suggests {tone_hint or '?'} tones "
                           "(the FSK receiver fits M, h and pulse by re-modulation)", "FAMILY", p_fsk))
        return UnifiedClassification("FSK", "FSK", p_fsk, p_fsk, feats, ev, sf, fsk_tone_hint=tone_hint)

    ev.append(Evidence(f"Family: linear (PSK/QAM) (P = {1 - p_fsk:.3f})", "FAMILY", 1 - p_fsk))
    fe, cfg, lin = classify_linear_capture(samples, sample_rate, ev,
                                           constant_envelope=feats["excess_cv"] < CONST_ENVELOPE_EXCESS)
    ev.extend(lin.evidence)
    feats.update({f"sym_{k}": v for k, v in lin.features.items()})
    return UnifiedClassification(lin.family, lin.modulation, float((1 - p_fsk) * lin.confidence), p_fsk, feats, ev,
                                 sf, lin, fe, cfg, None)


def fsk_fit_is_single_carrier(fsk_result) -> bool:
    """An FSK fit whose tone spacing is ~0 (h < FSK_MIN_H) is one carrier with phase jumps: PSK."""
    h = fsk_result.parameters.get("h")
    return h is not None and h.value is not None and float(h.value) < FSK_MIN_H


def reclassify_as_psk(samples: np.ndarray, sample_rate: Optional[float], cls: UnifiedClassification,
                      reason: str) -> UnifiedClassification:
    """Re-run the linear branch (constant-envelope PSK rate path) after the FSK fit was rejected."""
    ev = list(cls.evidence) + [Evidence(reason, "FAMILY", 1.0)]
    fe, cfg, lin = classify_linear_capture(samples, sample_rate, ev, constant_envelope=True)
    ev.extend(lin.evidence)
    return UnifiedClassification(lin.family, lin.modulation, lin.confidence, cls.p_fsk, cls.features, ev,
                                 cls.signal, lin, fe, cfg, None)
