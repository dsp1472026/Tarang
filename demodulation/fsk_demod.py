"""
L6 FSK demodulators and bit mapping.

Every demodulator returns a per-symbol tone log-likelihood matrix LL (N x M),
LL[n, k] = log-likelihood (up to a per-symbol constant) that symbol n used tone
position k. One mapping layer (symbol_ll_to_bits) then turns LL into hard bits
and max-log LLRs for any label mapping / spectral inversion / differential
hypothesis. LLR convention (as in Phase 1): llr = ln P(b=0)/P(b=1).

  discriminator_ll   IF integrate-and-dump strobes vs. the tone grid (Gaussian model)
  correlator_ll      non-coherent tone-correlator bank over each symbol,
                     LL = 2|y_k| / N0 (max-log of ln I0)
  bcjr_ll            coherent max-log-MAP (BCJR) over the CPM trellis; states =
                     (accumulated phase in steps of pi/q) x (previous L-1 symbols),
                     valid for continuous-phase signals with rational h = p/q.
                     The carrier-phase reference comes from a decision-directed
                     comparison with a model built from non-coherent decisions.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d

from core.evidence import Evidence
from dsp.constellations import gray
from dsp.cpm import cpm_phase, freq_pulse, phase_pulse
from synthetic.framing import SYNC_BITS, ParsedFrame, bits_to_bytes, bytes_to_bits, parse_frame

SYNC_MAX_BIT_ERRORS = 12


# ---------------------------------------------------------------------------
# Demodulators
# ---------------------------------------------------------------------------

def discriminator_ll(v: np.ndarray, levels_hz: np.ndarray, sigma: float) -> np.ndarray:
    """v: IF strobes (cycles/symbol); levels_hz: tone centres (cycles/symbol)."""
    return -((v[:, None] - levels_hz[None, :]) ** 2) / (2 * max(sigma, 1e-4) ** 2)


def correlator_ll(r: np.ndarray, starts: np.ndarray, sps: float, tones: np.ndarray, n0: float) -> np.ndarray:
    """
    Non-coherent correlator bank. tones: tone frequencies in cycles/sample.
    Symbol n uses the samples in [starts[n], starts[n] + sps).
    """
    N, M = len(starts), len(tones)
    out = np.zeros((N, M))
    for n, s0 in enumerate(starts):
        i1, i2 = int(np.ceil(s0)), int(np.ceil(s0 + sps))
        i1, i2 = max(i1, 0), min(i2, len(r))
        if i2 <= i1:
            continue
        m = np.arange(i1, i2)
        y = np.exp(-2j * np.pi * np.outer(tones, m)) @ r[i1:i2]
        out[n] = 2 * np.abs(y) / max(n0, 1e-9)
    return out


def strobe_isi_taps(pulse: str, bt: Optional[float], frac: float = 0.6, tol: float = 0.02) -> np.ndarray:
    """
    ISI taps seen by a centred integrate-and-dump IF strobe of width `frac`:
    g_j = mean over u in [-frac/2, frac/2] of p(u - j), j = -J..J (taps >= tol kept).
    """
    if pulse == "rect":
        return np.array([1.0])
    u = np.linspace(-frac / 2, frac / 2, 41)
    J = 0
    while np.mean(freq_pulse(u - (J + 1), pulse, bt)) >= tol:
        J += 1
    return np.array([np.mean(freq_pulse(u - j, pulse, bt)) for j in range(-J, J + 1)])


def isi_viterbi(v: np.ndarray, order: int, taps: np.ndarray, h: float, center: float) -> np.ndarray:
    """
    MLSE of tone positions from IF strobes through the pulse's ISI:
        v_k = center + (h/2) * sum_j taps[j] * a_{k+j-J},   a = 2k - M + 1
    Viterbi over states (a_{k-J+1} .. a_{k+J}); returns positions (0..M-1).
    Used for Gaussian (partial-response) hypotheses, where per-symbol strobe
    decisions fail on the ISI.
    """
    M = order
    L = len(taps)
    if L == 1:
        return np.clip(np.round((v - center) / h + (M - 1) / 2), 0, M - 1).astype(np.int64)
    J = (L - 1) // 2
    lv = 2 * np.arange(M) - M + 1
    n_mem = M ** (L - 1)
    mem = np.array([[(i // M ** (L - 2 - d)) % M for d in range(L - 1)] for i in range(n_mem)])   # oldest first
    # branch: state (L-1 positions) + input -> window of L positions -> predicted strobe
    win = np.concatenate([np.repeat(mem, M, axis=0), np.tile(np.arange(M), n_mem)[:, None]], axis=1)  # (n_mem*M, L)
    pred = center + 0.5 * h * (lv[win] @ taps)
    nxt = (np.repeat(np.arange(n_mem), M) * M + np.tile(np.arange(M), n_mem)) % n_mem
    N = len(v)
    # window for strobe k covers a_{k-J} .. a_{k+J}; process k = J .. N-1-J
    cost = np.zeros(n_mem)
    back = []
    for k in range(J, N - J):
        c = np.repeat(cost, M) + (v[k] - pred) ** 2
        new = np.full(n_mem, np.inf)
        arg = np.zeros(n_mem, dtype=np.int64)
        order_idx = np.lexsort((c, nxt))
        first = np.ones(len(order_idx), dtype=bool)
        first[1:] = nxt[order_idx][1:] != nxt[order_idx][:-1]
        sel = order_idx[first]
        new[nxt[sel]] = c[sel]
        arg[nxt[sel]] = sel
        back.append(arg)
        cost = new
    # trace back
    s = int(np.argmin(cost))
    windows = []
    for arg in reversed(back):
        b = arg[s]
        windows.append(win[b])
        s = b // M
    windows = windows[::-1]
    pos = np.zeros(N, dtype=np.int64)
    for i, w in enumerate(windows):
        k = i + J
        pos[k] = w[J]
        if i == 0:
            pos[:J] = w[:J]
        if i == len(windows) - 1:
            pos[k + 1:] = w[J + 1:]
    return pos


def rational_h(h: float, max_q: int = 16, tol: float = 0.08, q_cost: float = 0.003) -> Optional[Fraction]:
    """
    Fraction p/q for the CPM trellis, chosen by min |h - p/q| + q_cost * q among
    candidates within `tol` (relative). Physical modulation indices are simple
    fractions; the q penalty stops a noisy estimate (0.47) from snapping to a
    meaningless large-q fraction (7/15) while an exact 0.7 still gives 7/10.
    """
    best = None
    for q in range(1, max_q + 1):
        p = round(h * q)
        err = abs(p / q - h) if p > 0 else np.inf
        if err <= tol * max(h, 1e-3):
            cost = err + q_cost * q
            if best is None or cost < best[0]:
                best = (cost, Fraction(p, q))
    return None if best is None else best[1]


def trellis_span(pulse: str, bt: Optional[float], tol: float = 0.03) -> int:
    """Odd pulse length used by the trellis (residual tails < tol are ignored)."""
    if pulse == "rect":
        return 1
    L = 1
    while freq_pulse(np.array([L / 2.0]), pulse, bt)[0] >= tol:
        L += 2
    return L


@dataclass
class CoherentRef:
    r_coh: np.ndarray          # derotated samples aligned with the model phase reference
    n0: float                  # noise variance per sample
    phase_track: np.ndarray    # removed phase (rad) per sample


def coherent_reference(r: np.ndarray, t_sym: np.ndarray, levels: np.ndarray, h: float, cfo: float,
                       pulse: str, bt: Optional[float], sps: float, avg_symbols: float = 16.0) -> CoherentRef:
    """
    Decision-directed carrier phase: e(t) = r(t) conj(s_model(t)), smoothed over
    `avg_symbols`, gives the phase of r relative to a CPM model built from
    non-coherent decisions (phase reference: 0 at the first decided symbol).
    """
    s = np.exp(1j * (cpm_phase(levels, t_sym, h, pulse, bt) + 2 * np.pi * cfo * t_sym))
    e = r * np.conj(s)
    w = max(3, int(avg_symbols * sps))
    es = uniform_filter1d(e.real, w) + 1j * uniform_filter1d(e.imag, w)
    ph = np.unwrap(np.angle(es))
    r_coh = r * np.exp(-1j * (ph + 2 * np.pi * cfo * t_sym))
    amp = np.mean(np.abs(es))
    n0 = float(np.mean(np.abs(r * np.exp(-1j * ph) - amp * s) ** 2))
    return CoherentRef(r_coh, max(n0, 1e-6), ph)


def bcjr_ll(r_coh: np.ndarray, t_sym: np.ndarray, n_sym: int, order: int, h: Fraction, pulse: str,
            bt: Optional[float], n0: float) -> np.ndarray:
    """
    Max-log-MAP over the CPM trellis.

    Step n covers J_n = [n + 1/2 - L/2, n + 3/2 - L/2) (symbol units): exactly the
    symbols n-L+1 .. n are inside their frequency pulse there; older symbols are
    folded into the phase state theta = j * pi/q (j mod 2q). State = (j, a_{n-L+1..n-1}),
    input = a_n; next j' = j + p * a_{n-L+1}.
    Branch metric = (2/N0) Re sum_m r_coh[m] conj(exp(j phi_branch(t_m))).
    Returns LL (n_sym x M) over tone positions.
    """
    p, q = h.numerator, h.denominator
    L = trellis_span(pulse, bt)
    M = order
    lv = 2 * np.arange(M) - M + 1                                      # tone position -> level
    n_prev = L - 1
    n_mem = M ** n_prev
    S = 2 * q * n_mem
    # memory index -> tuple of previous positions (oldest first)
    mem = np.array([[(i // M ** (n_prev - 1 - d)) % M for d in range(n_prev)] for i in range(n_mem)],
                   dtype=np.int64).reshape(n_mem, n_prev)
    # branch tables over (state, input)
    st_j = np.repeat(np.arange(2 * q), n_mem)                           # (S,)
    st_m = np.tile(np.arange(n_mem), 2 * q)                             # (S,)
    combos = np.concatenate([mem[st_m][:, None, :].repeat(M, axis=1),
                             np.broadcast_to(np.arange(M)[None, :, None], (S, M, 1))], axis=2)  # (S, M, L)
    oldest = combos[:, :, 0]
    nj = (st_j[:, None] + p * lv[oldest]) % (2 * q)
    new_mem = np.zeros((S, M), dtype=np.int64)
    for d in range(1, L):                                               # drop oldest, append input
        new_mem = new_mem * M + combos[:, :, d]
    if n_prev == 0:
        new_mem[:] = 0
    nxt = nj * n_mem + new_mem                                          # (S, M)
    lv_combo = lv[combos].astype(np.float64)                             # (S, M, L)
    theta = st_j * np.pi / q                                            # (S,)

    # sample indices per step
    edges = np.arange(n_sym + 1) + 0.5 - L / 2.0
    idx = np.searchsorted(t_sym, edges)
    gam = np.full((n_sym, S, M), 0.0)
    for n in range(n_sym):
        m1, m2 = idx[n], idx[n + 1]
        if m2 <= m1:
            continue
        t = t_sym[m1:m2]
        k = n - L + 1 + np.arange(L)                                    # active symbols, oldest first
        P = phase_pulse(t[None, :] - (k[:, None] + 0.5), pulse, bt)     # (L, Ns); symbol k centred at k + 1/2
        ph = theta[:, None, None] + np.pi * float(h) * np.einsum("sml,ln->smn", lv_combo, P)
        gam[n] = (2.0 / n0) * np.real(np.exp(-1j * ph) @ r_coh[m1:m2])

    NEG = -1e30
    alpha = np.zeros((n_sym + 1, S))
    for n in range(n_sym):
        cand = alpha[n][:, None] + gam[n]
        a_next = np.full(S, NEG)
        np.maximum.at(a_next, nxt.ravel(), cand.ravel())
        alpha[n + 1] = a_next - a_next.max()
    beta = np.zeros((n_sym + 1, S))
    for n in range(n_sym - 1, -1, -1):
        b = (gam[n] + beta[n + 1][nxt]).max(axis=1)
        beta[n] = b - b.max()
    LL = np.zeros((n_sym, M))
    for n in range(n_sym):
        tot = alpha[n][:, None] + gam[n] + beta[n + 1][nxt]
        LL[n] = tot.max(axis=0)
    return LL - LL.max(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Bit mapping hypotheses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MapHyp:
    mapping: str        # "gray" | "natural"
    inverted: bool      # spectral inversion (tone position k -> M-1-k), e.g. swapped I/Q
    differential: bool  # data carried by tone-position increments

    def label(self) -> str:
        return (f"{self.mapping}{', inverted' if self.inverted else ''}"
                f"{', differential' if self.differential else ''}")


def hypotheses(order: int) -> List[MapHyp]:
    maps = ["gray"] if order == 2 else ["gray", "natural"]
    return [MapHyp(m, inv, d) for d in (False, True) for inv in (False, True) for m in maps]


def _label_of_position(order: int, mapping: str) -> np.ndarray:
    return gray(np.arange(order)) if mapping == "gray" else np.arange(order)


def symbol_ll_to_bits(LL: np.ndarray, hyp: MapHyp):
    """Per-symbol tone LL -> (hard bits, max-log LLRs) under a mapping hypothesis."""
    N, M = LL.shape
    bps = int(np.log2(M))
    if hyp.inverted:
        LL = LL[:, ::-1]
    if hyp.differential:
        # increment d = (k_n - k_{n-1}) mod M:  LL_d = max_k LL_n[(k+d)%M] + LL_{n-1}[k]
        k = np.arange(M)
        LLd = np.empty((N - 1, M))
        for d in range(M):
            LLd[:, d] = np.max(LL[1:, (k + d) % M] + LL[:-1, k], axis=1)
        LL = LLd
    labels = _label_of_position(M, hyp.mapping)
    table = ((labels[:, None] >> np.arange(bps - 1, -1, -1)[None, :]) & 1).astype(np.uint8)   # (M, bps)
    llr = np.empty((LL.shape[0], bps))
    for b in range(bps):
        llr[:, b] = LL[:, table[:, b] == 0].max(axis=1) - LL[:, table[:, b] == 1].max(axis=1)
    llr = llr.ravel().astype(np.float32)
    return (llr < 0).astype(np.uint8), llr


# ---------------------------------------------------------------------------
# Framing / ambiguity resolution
# ---------------------------------------------------------------------------

@dataclass
class FSKDecode:
    method: str                    # demodulator name
    resolution: str                # "sync-word" | "known-payload" | "text-heuristic" | "none"
    hyp: MapHyp
    hard_bits: np.ndarray
    llrs: np.ndarray
    payload_bit_start: Optional[int]
    frame: Optional[ParsedFrame]
    payload: bytes
    sync_bit_errors: Optional[int] = None
    evidence: List[Evidence] = field(default_factory=list)

    @property
    def crc_ok(self) -> bool:
        return bool(self.frame and self.frame.crc_ok)

    @property
    def text(self) -> str:
        return self.payload.decode("ascii", errors="replace")


def _search(stream: np.ndarray, pattern: np.ndarray, step: int, min_overlap: float = 1.0):
    """
    Best (offset, errors) for `pattern` in `stream` at offsets that are multiples of
    `step`. With min_overlap < 1 the pattern may run off either end of the stream
    (errors counted on the overlap, missing bits counted as errors) - used for the
    expected-payload test on whole-file captures whose edge symbols are clipped.
    """
    n, m = len(stream), len(pattern)
    if min_overlap < 1.0:
        best = (None, m)
        lo, hi = -int((1 - min_overlap) * m), n - int(min_overlap * m)
        for off in range(lo - lo % step, hi + 1, step):
            a, b = max(off, 0), min(off + m, n)
            if b - a < min_overlap * m:
                continue
            err = int(np.count_nonzero(stream[a:b] != pattern[a - off:b - off])) + (m - (b - a))
            if err < best[1]:
                best = (off, err)
        return best
    if n < m:
        return None, m
    offs = np.arange(0, n - m + 1, step)
    win = np.lib.stride_tricks.sliding_window_view(stream, m)[offs]
    errs = np.count_nonzero(win != pattern[None, :], axis=1)
    k = int(np.argmin(errs))
    return int(offs[k]), int(errs[k])


def decode_fsk(LL: np.ndarray, method: str, expected_payload: Optional[str] = None) -> FSKDecode:
    """
    Resolve the bit-mapping ambiguity and parse the frame.
    Every mapping hypothesis (Gray/natural x inversion x differential) is tried;
    sync-word hits are ranked CRC-valid first, then fewest sync errors, then the
    simplest hypothesis. Unframed captures fall back to an expected payload or
    to the ASCII-text score used in Phase 1 (demodulation.psk.text_score).
    """
    from demodulation.psk import best_model, best_text_region, text_decisive

    M = LL.shape[1]
    bps = int(np.log2(M))
    hyps = hypotheses(M)
    results = []
    for rank, hyp in enumerate(hyps):
        bits, llr = symbol_ll_to_bits(LL, hyp)
        off, err = _search(bits, SYNC_BITS, bps)
        if off is not None and err <= SYNC_MAX_BIT_ERRORS:
            frame = parse_frame(bits[off + len(SYNC_BITS):])
            results.append((bool(frame and frame.crc_ok), -err, -rank, hyp, bits, llr, off, err, frame))
    if results:
        crc, _, _, hyp, bits, llr, off, err, frame = max(results, key=lambda r: r[:3])
        ev = [Evidence(f"{method}: sync word at bit {off} with {err}/96 errors under mapping '{hyp.label()}' "
                       f"({len(results) - 1} other hypotheses with a sync hit)", "SYNC_WORD", 1 - err / 96)]
        if frame:
            ev.append(Evidence(f"Frame length {frame.length} B, CRC-16 {'VALID' if frame.crc_ok else 'FAIL'}",
                               "CRC", float(frame.crc_ok)))
        start = off + len(SYNC_BITS) + 16
        return FSKDecode(method, "sync-word", hyp, bits, llr, start, frame, frame.payload if frame else b"",
                         err, ev)

    decoded = [(hyp,) + symbol_ll_to_bits(LL, hyp) for hyp in hyps]
    if expected_payload:
        pat = bytes_to_bits(expected_payload.encode("ascii"))
        best = None
        for hyp, bits, llr in decoded:
            off, err = _search(bits, pat, 1, min_overlap=0.9)
            if off is not None and (best is None or err < best[0]):
                best = (err, off, hyp, bits, llr)
        if best:
            err, off, hyp, bits, llr = best
            seg = bits[max(off, 0):off + len(pat)]
            if off < 0:      # leading bits clipped: pad with the (unknown) missing bits as zeros
                seg = np.concatenate([np.zeros(-off, dtype=np.uint8), seg])
            return FSKDecode(method, "known-payload", hyp, bits, llr, off, None,
                             bits_to_bytes(seg), None,
                             [Evidence(f"{method}: no sync word; mapping '{hyp.label()}' chosen by the expected "
                                       f"payload ({err}/{len(pat)} bit errors at bit {off})", "KNOWN_PAYLOAD",
                                       1 - err / len(pat))])

    # Unframed capture: the byte alignment (and a possible binary preamble) is unknown too, so
    # every mapping is scored with the text search used for PSK (bit offsets 0..7, header skip,
    # general / English / numeric message models); the runner-up is the best DIFFERENT payload.
    regions = [best_text_region(bits) for _, bits, _ in decoded]
    order = sorted(range(len(decoded)), key=lambda i: (regions[i][0], -i), reverse=True)
    b = order[0]
    best_ll, o = regions[b]
    hyp, bits, llr = decoded[b]
    payload = bits_to_bytes(bits[o:])
    runner = next((regions[i][0] for i in order[1:]
                   if bits_to_bytes(decoded[i][1][regions[i][1]:]) != payload), -np.inf)
    if text_decisive(best_ll, runner, len(payload)):
        return FSKDecode(method, "text-heuristic", hyp, bits, llr, o, None, payload, None,
                         [Evidence(f"{method}: no sync word; mapping '{hyp.label()}', text at bit {o} chosen by "
                                   f"text likelihood {best_ll:.2f} vs {runner:.2f} nats/char, "
                                   f"{best_model(payload)[1]} message model (heuristic)", "TEXT_HEURISTIC", 0.5)])
    hyp, bits, llr = decoded[0]
    return FSKDecode(method, "none", hyp, bits, llr, None, None, bits_to_bytes(bits), None,
                     [Evidence(f"{method}: no sync word and no decisive content test; bits reported under "
                               f"'{hyp.label()}'", "AMBIGUITY", 0.0)])
