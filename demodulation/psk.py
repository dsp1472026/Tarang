"""
L6 PSK demodulation: phase-ambiguity resolution, hard/soft demapping, framing.

After carrier recovery the constellation is correct up to a rotation of
2*pi*r/S (S = phase symmetry: M for M-PSK, 4 for square QAM). Strategies, tried
in this order and always reported:

  1. "sync-word"     coherent correlation with the known 96-bit sync word
                     (the correlation phase gives r directly; then CRC check)
  2. "differential"  non-coherent differential detection + sync-word search
                     (for differentially encoded signals; rotation-immune)
     Strategies 1 and 2 are evaluated together, for every carrier-frequency
     alias a*Rs/M (the M-th power estimator's inherent ambiguity), and ranked
     by CRC validity, then sync-bit errors.
  3. "known-payload" legacy captures without a sync word: choose the rotation
                     whose bits best match a caller-supplied expected payload
  4. "text-heuristic" choose the (alias, rotation) whose bytes are most English-like
                     (weak evidence; flagged as such)
  5. "unresolved"    rotation 0, reported as unresolved
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.evidence import Evidence
from dsp.constellations import Constellation, differential_demap
from synthetic.framing import SYNC_BITS, ParsedFrame, bits_to_bytes, bytes_to_bits, parse_frame

SYNC_CORR_THRESHOLD = 0.6     # normalised |correlation| to declare the sync word found
SYNC_MAX_BIT_ERRORS = 12      # of 96 sync bits (12.5 %)


@dataclass
class DecodeResult:
    method: str
    rotation_steps: int                   # r in 2*pi*r/M removed (coherent only)
    differential: bool
    hard_bits: np.ndarray                 # whole symbol stream after ambiguity resolution
    llrs: np.ndarray                      # matching max-log LLRs, ln P(0)/P(1)
    payload_bit_start: Optional[int]      # bit index of the first payload bit (None if unknown)
    frame: Optional[ParsedFrame]
    payload: bytes
    sync_symbol_index: Optional[int] = None
    sync_bit_errors: Optional[int] = None
    sync_corr: Optional[float] = None
    evidence: List[Evidence] = field(default_factory=list)
    alias_steps: int = 0                  # carrier alias a (a*Rs/M) removed before decoding
    mapping: str = "gray-ccw"             # bit-label convention used (captures without a sync word)
    constellation: Optional[Constellation] = None   # the labelled constellation actually used

    @property
    def text(self) -> str:
        return self.payload.decode("ascii", errors="replace")


def _text_model() -> np.ndarray:
    """
    Log-probabilities of a generic ASCII-text byte model (not tuned to any payload): a
    mixture of English text (letter frequencies shared equally between upper and lower
    case, space, digits, common punctuation) and 25 % spread uniformly over ALL printable
    ASCII (32..126, TAB, LF, CR), so messages full of digits and symbols (# $ % @ [ ] { } ~ ...)
    still read as text. Non-printable bytes are very unlikely.
    """
    freq = dict(zip("ETAOINSHRDLCUMWFGYPBVKJXQZ",
                    [12.7, 9.1, 8.2, 7.5, 7.0, 6.7, 6.3, 6.1, 6.0, 4.3, 4.0, 2.8, 2.8, 2.4, 2.4, 2.2, 2.0,
                     2.0, 1.9, 1.5, 1.0, 0.8, 0.15, 0.15, 0.1, 0.07]))
    p = np.full(256, 1e-5)                         # non-printable bytes: very unlikely
    p[PRINTABLE] = 0.25 / len(PRINTABLE)           # any printable character
    p[ord(" ")] += 0.16
    for d in "0123456789":
        p[ord(d)] += 0.004
    for ch in ".,!?'-:;\n":
        p[ord(ch)] += 0.004
    for ch, f in freq.items():
        p[ord(ch)] += 0.5 * 0.75 * f / 100.0
        p[ord(ch.lower())] += 0.5 * 0.75 * f / 100.0
    return np.log(p / p.sum())


PRINTABLE = np.r_[9, 10, 13, 32:127]
_TEXT_LOGP = _text_model()
# nats/char. With the charset terms below: random bytes score < -10, English ~ -3.5..-4.5, text
# dense in digits/symbols ~ -5..-6.6 (a symbols-only line). The margin test does the real work.
TEXT_MIN_LOGLIK = -7.0
TEXT_MARGIN = 0.5        # required lead over the runner-up hypothesis
TEXT_MIN_BYTES = 6       # a "text" reading shorter than this is no evidence (1-3 byte fragments of wrong
                         # rate / order hypotheses otherwise passed as text)
SHORT_TEXT_BYTES = 24    # below this length the decision is tightened (see text_decisive)
SHORT_TEXT_MIN_LOGLIK = -5.5


def text_decisive(score: float, runner: float, n_bytes: int) -> bool:
    """
    Accept a text reading? score >= TEXT_MIN_LOGLIK with a TEXT_MARGIN lead over the best
    different reading. On short readings the best of hundreds of hypotheses (QAM: 512 label
    mappings x alias x rotation x 8 alignments) fluctuates more (~1/sqrt(n)): a 13-byte
    reading 'S=4OM29i...' of garbage passed. Below SHORT_TEXT_BYTES the reading must score
    >= SHORT_TEXT_MIN_LOGLIK (real text scores -2.6 (digits) .. -4.5 (English)) and the
    margin grows as sqrt(SHORT_TEXT_BYTES / n).
    """
    if n_bytes < TEXT_MIN_BYTES:
        return False
    if n_bytes >= SHORT_TEXT_BYTES:
        return score >= TEXT_MIN_LOGLIK and score - runner >= TEXT_MARGIN
    return (score >= SHORT_TEXT_MIN_LOGLIK
            and score - runner >= TEXT_MARGIN * float(np.sqrt(SHORT_TEXT_BYTES / n_bytes)))


def text_loglik(data: bytes) -> float:
    b = np.frombuffer(data, dtype=np.uint8)
    return float(np.mean(_TEXT_LOGP[b])) if len(b) else -np.inf


PSK_MAPPINGS = ("gray-ccw", "gray-cw", "natural-ccw", "natural-cw")
PLAIN_TEXT = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'\"-()\n")
PRINTABLE_WEIGHT = 5.0     # nats/char per unit fraction of NON-printable bytes
CHARSET_WEIGHT = 1.0       # nats/char per unit fraction of printable symbols outside PLAIN_TEXT
# Per-byte score = unigram log-probability + the two charset terms.
_PER_BYTE = (_TEXT_LOGP + PRINTABLE_WEIGHT * (np.isin(np.arange(256), PRINTABLE) - 1.0)
             + CHARSET_WEIGHT * (np.isin(np.arange(256), list(PLAIN_TEXT)) - 1.0))


def _numeric_model() -> np.ndarray:
    """Per-byte scores of a NUMERIC message (digit strings: counters, phone numbers, readings,
    dates): 70 % digits, 12 % space, 12 % separators . , - + : / ( ), 6 % any other printable.
    Under the English unigram a digit is a rare character, so a wrong rotation that spells
    lowercase junk out-scored the true digits; under this model true digits score ~-2.7
    nats/char, far above any wrong reading (random bytes are digits 4 % of the time)."""
    p = np.full(256, 1e-6)
    p[PRINTABLE] = 0.06 / len(PRINTABLE)
    p[ord(" ")] += 0.12
    for d in "0123456789":
        p[ord(d)] += 0.07
    for ch in ".,-+:/()":
        p[ord(ch)] += 0.015
    return np.log(p / p.sum()) + PRINTABLE_WEIGHT * (np.isin(np.arange(256), PRINTABLE) - 1.0)


_NUMERIC_PER_BYTE = _numeric_model()

# Generic English prose (written for this model, not any test payload) for the letter-pair model.
_ENGLISH_SAMPLE = """
the receiver has no knowledge of the transmitter so every parameter is estimated from the signal
itself and each decision is reported together with the evidence that supports it. when the message
is plain text the letters follow the patterns of the language: some pairs of letters appear in almost
every sentence while others are very rare, and words are separated by single spaces. this report
describes how the signal was found, how the symbol rate was measured, how the carrier was locked
and how the bits were recovered. the operator should check the results before using them because a
weak signal or a strong interference can change the outcome. please send the weather report for the
northern region before noon tomorrow. meet at the station near the old bridge after the evening train
arrives. all units must return to base when the storm passes and wait for further orders from the
command centre. the quick brown fox jumps over the lazy dog while the five boxing wizards jump quickly.
there is nothing that cannot be measured with patience, and the first answer is not always the best
one. we will continue the test with a different message and a longer frame to see whether the result
changes. thank you for your help with the project; it would not have been possible without the team.
people often write short notes, questions and replies such as yes, no, maybe, hello, good morning,
see you soon, where are you, what time is it, how much does it cost and who is calling.
"""


def _english_pair_bonus() -> np.ndarray:
    """256x256 table of per-pair log-likelihood-ratio bonuses, log P(b|a) - log P(b), for case-folded
    letter/space pairs (add-one smoothed letter bigram of _ENGLISH_SAMPLE), 0 for other pairs.
    Label-mapping searches (QAM especially) produce letter-shaped garbage ("PACO I] BK WMTL")
    within a few tenths of a nat of the truth under the unigram; letter ORDER separates them."""
    alpha = "abcdefghijklmnopqrstuvwxyz "
    idx = {c: i for i, c in enumerate(alpha)}
    s = " ".join(_ENGLISH_SAMPLE.lower().split())
    cnt = np.ones((27, 27))
    for a, b in zip(s, s[1:]):
        if a in idx and b in idx:
            cnt[idx[a], idx[b]] += 1
    cond = cnt / cnt.sum(axis=1, keepdims=True)
    uni = cnt.sum(axis=0) / cnt.sum()
    pmi = np.log(cond) - np.log(uni)[None, :]
    code = np.full(256, -1)
    for c, i in idx.items():
        code[ord(c)] = i
        if c != " ":
            code[ord(c.upper())] = i
    table = np.zeros((256, 256))
    ok = np.flatnonzero(code >= 0)
    table[np.ix_(ok, ok)] = pmi[np.ix_(code[ok], code[ok])]
    lower, upper = np.arange(97, 123), np.arange(65, 91)
    table[np.ix_(lower, upper)] -= 2.0             # "aB": case switching inside a word is rare
    return table


_PAIR_BONUS = _english_pair_bonus()
PAIR_WEIGHT = 0.7          # nats per pair bonus unit (the PMI of real text averages ~ +0.7/char)


def model_scores(b: np.ndarray) -> dict:
    """Mean per-byte score of a byte array under each message model."""
    if len(b) == 0:
        return {"text": -np.inf, "english": -np.inf, "numeric": -np.inf}
    base = _PER_BYTE[b]
    pair = _PAIR_BONUS[b[:-1], b[1:]].sum() if len(b) > 1 else 0.0
    return {"text": float(np.mean(base)),
            "english": float((base.sum() + PAIR_WEIGHT * pair) / len(b)),
            "numeric": float(np.mean(_NUMERIC_PER_BYTE[b]))}


def best_model(data: bytes):
    """(score, model name) of the best-fitting message model: general text (any printable
    characters), English (letter-pair statistics) or numeric (digit strings)."""
    sc = model_scores(np.frombuffer(data, dtype=np.uint8))
    name = max(sc, key=sc.get)
    return sc[name], name


def text_score(data: bytes) -> float:
    """Text score = the best of three message models (see best_model). The general model is the
    mean per-byte unigram + charset score: label-mapping searches produce letter-like garbage
    close to the truth under the unigram alone; it swaps letters for rarer symbols, which the weak
    charset term penalises, while wrong alignments / rotations produce non-printable bytes,
    which the strong term rules out. Symbols are allowed in real text (weak penalty only).
    The English and numeric models only ever RAISE a reading's score, so messages the general
    model already accepted keep their score; they separate English and digit messages from the
    near-text readings of wrong rotations / label mappings."""
    return best_model(data)[0]


MAX_HEADER_BYTES = 8       # a binary preamble / sync header before the text is skipped when scoring


MIN_BYTES_FOR_REGION_SEARCH = 16
NON_TEXT_BYTE = -10.0      # per-byte score below which a byte is certainly not text (printable >= -7.3,
                           # non-printable ~ -17.7)


def best_text_region(bits: np.ndarray, align_step: int = 1):
    """
    (score, bit offset) of the most text-like reading of a bit stream. For every bit
    alignment 0..7, a binary header is dropped: the prefix up to the LAST certainly-non-text
    byte among the first MAX_HEADER_BYTES (a 32-bit preamble before the text otherwise pulled
    the score below the acceptance threshold). Real text is never trimmed, because no
    letter scores below NON_TEXT_BYTE. A trailer is cut the same way (from the first certainly-
    non-text byte among the last MAX_HEADER_BYTES). Score = text_score of the kept bytes; the
    returned offset is the start (the trailer stays in the reported payload).
    align_step > 1 restricts the alignments to multiples of it (callers pass gcd(8, bits per
    symbol): a transmitted byte stream starts on a symbol boundary).
    """
    best = (-np.inf, 0)
    if len(bits) < 8 * MIN_BYTES_FOR_REGION_SEARCH:
        # short streams: thousands of (alignment x header) readings of a few bytes let random
        # hypotheses come close to the truth, so only the plain byte-aligned reading is scored
        data = bits_to_bytes(bits)
        return (text_score(data) if len(data) >= TEXT_MIN_BYTES else -np.inf), 0
    for align in range(0, 8, align_step):
        b = np.frombuffer(bits_to_bytes(bits[align:]), dtype=np.uint8)
        if len(b) == 0:
            continue
        sc = _PER_BYTE[b]
        head = np.flatnonzero(sc[:min(MAX_HEADER_BYTES, len(b) // 4)] <= NON_TEXT_BYTE)
        skip = int(head[-1]) + 1 if len(head) else 0
        # likewise a trailer (guard / ramp-down symbols, padding): cut from the FIRST certainly-non-text
        # byte among the last MAX_HEADER_BYTES
        n_tail = min(MAX_HEADER_BYTES, (len(b) - skip) // 4)
        tail = np.flatnonzero(sc[len(b) - n_tail:] <= NON_TEXT_BYTE) if n_tail > 0 else []
        end = len(b) - n_tail + int(tail[0]) if len(tail) else len(b)
        if end - skip < TEXT_MIN_BYTES:
            continue
        score = text_score(b[skip:end].tobytes())
        if score > best[0]:
            best = (score, align + 8 * skip)
    return best


def mapped_psk(const: Constellation, mapping: str) -> Constellation:
    """
    M-PSK with another bit-label convention. Angular position p (counter-clockwise steps from
    the reference point) carries label gray(p) ("gray-ccw", the default), natural p
    ("natural-ccw"), or the same going clockwise ("-cw"). For QPSK, "gray-cw" is the square
    4QAM labelling (first bit = I sign, second = Q sign, 00 at (-,-)).
    """
    from dsp.constellations import gray
    order = const.order
    p = np.arange(order)
    phi0 = np.angle(const.points[0])                # position 0 of the default constellation
    ang = phi0 + (1 if mapping.endswith("ccw") else -1) * 2 * np.pi * p / order
    labels = gray(p) if mapping.startswith("gray") else p
    pts = np.empty(order, dtype=np.complex128)
    pts[labels] = np.exp(1j * ang)
    return Constellation(name=const.name, family=const.family, points=pts,
                         bits_per_symbol=const.bits_per_symbol, phase_symmetry=const.phase_symmetry)


def rotate(y: np.ndarray, steps: int, order: int) -> np.ndarray:
    return y * np.exp(-2j * np.pi * steps / order)


def _search_bits(stream: np.ndarray, pattern: np.ndarray, step: int):
    """Best (offset, errors) for `pattern` in `stream` over offsets that are multiples of `step`."""
    n, m = len(stream), len(pattern)
    if n < m:
        return None, m
    offsets = np.arange(0, n - m + 1, step)
    win = np.lib.stride_tricks.sliding_window_view(stream, m)[offsets]
    errs = np.count_nonzero(win != pattern[None, :], axis=1)
    k = int(np.argmin(errs))
    return int(offsets[k]), int(errs[k])


def _finish_frame(bits: np.ndarray, sync_bit: int) -> tuple:
    frame = parse_frame(bits[sync_bit + len(SYNC_BITS):])
    start = sync_bit + len(SYNC_BITS) + 16
    return frame, (frame.payload if frame else b""), start


def coherent_sync(y: np.ndarray, const: Constellation):
    """Complex correlation with the modulated sync word -> (index, rotation steps, |corr|)."""
    p = const.map_bits(SYNC_BITS)
    if len(y) < len(p):
        return None, 0, 0.0
    c = np.correlate(y, p, mode="valid")              # c[k] = sum_i y[k+i] conj(p[i])
    energy = np.convolve(np.abs(y) ** 2, np.ones(len(p)), mode="valid")
    norm = np.abs(c) / (np.sqrt(energy * np.sum(np.abs(p) ** 2)) + 1e-12)
    k = int(np.argmax(norm))
    sym = const.phase_symmetry
    steps = int(np.round(np.angle(c[k]) / (2 * np.pi / sym))) % sym
    return k, steps, float(norm[k])


def remove_alias(y: np.ndarray, alias: int, order: int) -> np.ndarray:
    """Undo a carrier-frequency alias of alias*Rs/M (a rotation of 2*pi*alias/M per symbol)."""
    return y * np.exp(-2j * np.pi * alias * np.arange(len(y)) / order)


def _sync_hypotheses(y, const, noise_var, allow_differential):
    """
    Enumerate {coherent, differential} x {carrier alias} hypotheses that contain the sync word.

    The M-th-power frequency estimate is only unique modulo Rs/M: an error of
    a*Rs/M rotates successive symbols by 2*pi*a/M, which the M-th power and the
    decision-directed PLL cannot see. The known sync word can, so every alias
    a in [0, M) is tried.
    """
    order, bps = const.phase_symmetry, const.bits_per_symbol     # M for PSK, 4 for square QAM
    out = []
    for alias in range(order):
        ya = remove_alias(y, alias, order)
        k, steps, corr = coherent_sync(ya, const)
        if k is not None and corr >= SYNC_CORR_THRESHOLD:
            yr = rotate(ya, steps, order)
            bits = const.demap_hard(yr)
            err = int(np.count_nonzero(bits[k * bps:k * bps + len(SYNC_BITS)] != SYNC_BITS))
            if err <= SYNC_MAX_BIT_ERRORS:
                frame, payload, start = _finish_frame(bits, k * bps)
                out.append(dict(method="sync-word", alias=alias, steps=steps, diff=False, bits=bits,
                                llr=lambda yr=yr: const.demap_soft(yr, noise_var), start=start, frame=frame,
                                payload=payload, k=k, err=err, corr=corr))
        if allow_differential:
            dbits, dllr = differential_demap(ya, const, noise_var)
            off, err = _search_bits(dbits, SYNC_BITS, bps)
            if off is not None and err <= SYNC_MAX_BIT_ERRORS:
                frame, payload, start = _finish_frame(dbits, off)
                out.append(dict(method="differential", alias=alias, steps=0, diff=True, bits=dbits,
                                llr=lambda dllr=dllr: dllr, start=start, frame=frame, payload=payload,
                                k=off // bps, err=err, corr=None))
    return out


def decode_psk(y: np.ndarray, const: Constellation, noise_var: float,
               expected_payload: Optional[str] = None, allow_differential: bool = True) -> DecodeResult:
    # Rotations / carrier aliases are multiples of 2*pi/S, S = the constellation's phase
    # symmetry (M for M-PSK, 4 for square QAM).
    order, bps = const.phase_symmetry, const.bits_per_symbol
    ev: List[Evidence] = []

    # 1+2. Sync word: coherent / differential x carrier alias, ranked -----------
    hyps = _sync_hypotheses(y, const, noise_var, allow_differential)
    if hyps:
        # CRC-valid first, then fewest sync-bit errors, then the smallest alias.
        rank = lambda h: (bool(h["frame"] and h["frame"].crc_ok), -h["err"], -min(h["alias"], order - h["alias"]))
        best = max(hyps, key=rank)
        a = best["alias"]
        alias_txt = "" if a == 0 else f"; carrier alias {a if a <= order // 2 else a - order}*Rs/{order} corrected"
        others = len(hyps) - 1
        if best["diff"]:
            ev.append(Evidence(f"Differential detection: sync at bit {best['k'] * bps}, {best['err']}/96 bit errors"
                               f"{alias_txt} ({others} other hypotheses rejected)", "SYNC_WORD_DIFF",
                               1.0 - best["err"] / 96))
        else:
            ev.append(Evidence(f"Coherent sync at symbol {best['k']}: |corr|={best['corr']:.2f}, {best['err']}/96 bit "
                               f"errors, rotation {360 * best['steps'] / order:.0f} deg removed{alias_txt} "
                               f"({others} other hypotheses rejected)", "SYNC_WORD", best["corr"]))
        frame = best["frame"]
        if frame:
            ev.append(Evidence(f"Frame length {frame.length} B, CRC-16 {'VALID' if frame.crc_ok else 'FAIL'}",
                               "CRC", float(frame.crc_ok)))
        return DecodeResult(best["method"], best["steps"], best["diff"], best["bits"], best["llr"](), best["start"],
                            frame, best["payload"], best["k"], best["err"], best["corr"], ev,
                            a if a <= order // 2 else a - order)

    # 3. Known payload (legacy captures without framing) ----------------------
    # No sync word: every (bit-label mapping, carrier alias a, rotation r) is a hypothesis.
    # Legacy generators label PSK differently (natural binary, clockwise, square-QAM style
    # for 4QAM); a rotation cannot turn one convention into another.
    def signed(a):
        return a if a <= order // 2 else a - order

    mappings = PSK_MAPPINGS if const.family == "PSK" and const.order >= 4 else ("gray-ccw",)
    cands = []
    for mp in mappings:
        cm = const if mp == "gray-ccw" else mapped_psk(const, mp)
        for a in range(order):
            ya = remove_alias(y, a, order)
            for r in range(order):
                yr = rotate(ya, r, order)
                cands.append((a, r, yr, cm.demap_hard(yr), mp, cm))

    def label(mp):
        return "" if mp == "gray-ccw" else f"; bit labels {mp}" + (" (square-QAM 4QAM labelling)"
                                                                   if mp == "gray-cw" and order == 4 else "")

    if expected_payload:
        pat = bytes_to_bits(expected_payload.encode("ascii"))
        best = None
        for a, r, yr, bits, mp, cm in cands:
            off, err = _search_bits(bits, pat, bps)
            key = (err, mp != "gray-ccw")
            if off is not None and (best is None or key < best[0]):
                best = (key, off, a, r, yr, bits, mp, cm)
        if best is not None:
            (err, _), off, a, r, yr, bits, mp, cm = best
            ev.append(Evidence(f"No sync word; alias {signed(a)}*Rs/{order} and rotation {360 * r / order:.0f} deg "
                               f"chosen by matching the expected payload ({err}/{len(pat)} bit errors at bit {off}; "
                               f"{len(cands)} hypotheses){label(mp)}", "KNOWN_PAYLOAD", 1.0 - err / len(pat)))
            return DecodeResult("known-payload", r, False, bits, cm.demap_soft(yr, noise_var), off, None,
                                bits_to_bytes(bits[off:off + len(pat)]), None, None, None, ev, signed(a), mp, cm)

    # 4. Text-likelihood heuristic (unigram model + plain-text charset) ---------
    regions = [best_text_region(c[3]) for c in cands]

    def decide(idx):
        """Best candidate among idx and its lead over the best DIFFERENT payload."""
        scored = sorted(((regions[i][0], cands[i][4] == "gray-ccw", -abs(signed(cands[i][0])), i) for i in idx),
                        reverse=True)
        b_ll, _, _, b = scored[0]
        b_bytes = bits_to_bytes(cands[b][3][regions[b][1]:])
        r_ll = next((sc[0] for sc in scored[1:]
                     if bits_to_bytes(cands[sc[3]][3][regions[sc[3]][1]:]) != b_bytes), -np.inf)
        return b_ll, r_ll, b

    # Standard labels first (exactly the single-mapping behaviour); the other conventions only if
    # that is not decisive: more hypotheses mean more chances for garbage to look like text.
    std = [i for i, c in enumerate(cands) if c[4] == "gray-ccw"]
    def n_bytes(i):
        return (len(cands[i][3]) - regions[i][1]) // 8

    best_ll, runner_ll, i = decide(std)
    if not text_decisive(best_ll, runner_ll, n_bytes(i)) and len(std) < len(cands):
        best_ll, runner_ll, i = decide(range(len(cands)))
    off = regions[i][1]
    if text_decisive(best_ll, runner_ll, n_bytes(i)):
        a, r, yr, bits, mp, cm = cands[i]
        hdr = f"; text starts at bit {off} (header skipped)" if off else ""
        ev.append(Evidence(f"No sync word; alias {signed(a)}*Rs/{order}, rotation {360 * r / order:.0f} deg chosen by "
                           f"text likelihood {best_ll:.2f} nats/char vs runner-up {runner_ll:.2f}, "
                           f"{best_model(bits_to_bytes(bits[off:]))[1]} message model (heuristic; "
                           f"{len(cands)} hypotheses){label(mp)}{hdr}", "TEXT_HEURISTIC", 0.5))
        return DecodeResult("text-heuristic", r, False, bits, cm.demap_soft(yr, noise_var), off, None,
                            bits_to_bytes(bits[off:]), None, None, None, ev, signed(a), mp, cm)

    bits = const.demap_hard(y)
    ev.append(Evidence(f"Phase ambiguity unresolved ({order} rotations x {order} carrier aliases; best text "
                       f"likelihood {best_ll:.2f} vs runner-up {runner_ll:.2f} nats/char is not decisive); "
                       f"bits reported at rotation 0", "AMBIGUITY", 0.0))
    return DecodeResult("unresolved", 0, False, bits, const.demap_soft(y, noise_var), None, None,
                        bits_to_bytes(bits), None, None, None, ev)
