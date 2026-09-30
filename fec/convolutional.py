"""
Convolutional codes: library, encoder, puncturing, soft-decision Viterbi, syndrome test.

Conventions
-----------
* Generators are given in octal, MSB = tap on the current input bit:
  c_j[t] = sum_i g_j[i] u[t - i]  (g_j[0] = MSB of the octal word). K = constraint length.
* Output order per trellis step: c_1[t], c_2[t], ... (then puncturing column t mod P).
* LLRs are ln P(bit=0)/P(bit=1) (positive = 0), as produced by the demodulators.
* Zero-tail termination (K-1 zero input bits) unless terminate=False.

Syndrome test (hard bits, unpunctured rate 1/n): for every pair (1, j),
    c_1 * g_j + c_j * g_1 = u * g_1 * g_j + u * g_j * g_1 = 0   (mod 2)
so the syndrome weight of a correct (code, phase) hypothesis is ~ 2 x BER, while
random / wrong hypotheses give ~0.5 (Marazin-Gautier-Burel style parity-check evidence).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ConvCode:
    name: str
    K: int
    gens: Tuple[int, ...]                              # octal, e.g. (0o171, 0o133)
    puncture: Optional[Tuple[Tuple[int, ...], ...]] = None   # one row per generator

    @property
    def n(self) -> int:
        return len(self.gens)

    @property
    def rate(self) -> str:
        if self.puncture is None:
            return f"1/{self.n}"
        period = len(self.puncture[0])
        kept = int(sum(map(sum, self.puncture)))
        from math import gcd
        g = gcd(period, kept)
        return f"{period // g}/{kept // g}"

    @property
    def rate_value(self) -> float:
        a, b = self.rate.split("/")
        return int(a) / int(b)

    def taps(self) -> np.ndarray:
        """(n, K) 0/1 coefficient matrix, column 0 = current input."""
        return np.array([[(g >> (self.K - 1 - i)) & 1 for i in range(self.K)] for g in self.gens], dtype=np.uint8)

    def describe(self) -> str:
        g = ", ".join(f"{x:o}" for x in self.gens)
        p = "" if self.puncture is None else ", punctured " + " / ".join("".join(map(str, r)) for r in self.puncture)
        return f"rate {self.rate}, K = {self.K}, generators ({g})_8{p}"


_NASA = (0o171, 0o133)
LIBRARY: List[ConvCode] = [
    ConvCode("CC K=7 (171,133) r1/2 [NASA/CCSDS]", 7, _NASA),
    ConvCode("CC K=3 (7,5) r1/2", 3, (0o7, 0o5)),
    ConvCode("CC K=4 (15,17) r1/2", 4, (0o15, 0o17)),
    ConvCode("CC K=5 (23,35) r1/2", 5, (0o23, 0o35)),
    ConvCode("CC K=6 (53,75) r1/2", 6, (0o53, 0o75)),
    ConvCode("CC K=9 (561,753) r1/2 [IS-95/3GPP]", 9, (0o561, 0o753)),
    ConvCode("CC K=7 (133,171,165) r1/3 [LTE-style]", 7, (0o133, 0o171, 0o165)),
    ConvCode("CC K=3 (7,7,5) r1/3", 3, (0o7, 0o7, 0o5)),
    ConvCode("CC K=7 (171,133) r2/3 punct [DVB/CCSDS]", 7, _NASA, ((1, 0), (1, 1))),
    ConvCode("CC K=7 (171,133) r3/4 punct [DVB/CCSDS]", 7, _NASA, ((1, 0, 1), (1, 1, 0))),
    ConvCode("CC K=7 (171,133) r5/6 punct [DVB]", 7, _NASA, ((1, 0, 1, 0, 1), (1, 1, 0, 1, 0))),
]
BY_NAME = {c.name: c for c in LIBRARY}


def get_code(name: str) -> ConvCode:
    if name in BY_NAME:
        return BY_NAME[name]
    for c in LIBRARY:                                   # tolerant lookup ("K=7 r1/2", ...)
        if name.replace(" ", "").lower() in c.name.replace(" ", "").lower():
            return c
    raise KeyError(f"unknown convolutional code '{name}'")


# ---------------------------------------------------------------------------
# Encoder / puncturing
# ---------------------------------------------------------------------------

def mother_encode(bits: np.ndarray, code: ConvCode, terminate: bool = True) -> np.ndarray:
    """Information bits -> (T, n) unpunctured mother-code bits."""
    u = np.asarray(bits, dtype=np.uint8) & 1
    if terminate:
        u = np.concatenate([u, np.zeros(code.K - 1, dtype=np.uint8)])
    taps = code.taps()
    return np.stack([np.convolve(u, taps[j])[:len(u)] % 2 for j in range(code.n)], axis=1).astype(np.uint8)


def puncture_mask(code: ConvCode, T: int, phase: int = 0) -> np.ndarray:
    """(T, n) boolean: which mother-code bits are transmitted (trellis step 0 at column `phase`)."""
    if code.puncture is None:
        return np.ones((T, code.n), dtype=bool)
    P = np.array(code.puncture, dtype=bool)
    return P[:, (phase + np.arange(T)) % P.shape[1]].T


def encode(bits: np.ndarray, code: ConvCode, terminate: bool = True) -> np.ndarray:
    tn = mother_encode(bits, code, terminate)
    return tn[puncture_mask(code, tn.shape[0])]


def depuncture(llrs: np.ndarray, code: ConvCode, phase: int = 0) -> np.ndarray:
    """Transmitted LLR stream -> (T, n) mother-code LLRs with 0 (erasure) at punctured positions.
    phase: puncturing column of the first received bit's trellis step. Trailing bits that do not
    complete a trellis step are dropped."""
    llrs = np.asarray(llrs, dtype=np.float64)
    if code.puncture is None:
        T = len(llrs) // code.n
        return llrs[:T * code.n].reshape(T, code.n)
    per_step = np.array(code.puncture).sum(axis=0)
    cum = np.cumsum(per_step[(phase + np.arange(len(llrs) + 1)) % len(per_step)])
    T = int(np.searchsorted(cum, len(llrs), side="right"))
    mask = puncture_mask(code, T, phase)
    out = np.zeros((T, code.n))
    out[mask] = llrs[:int(mask.sum())]
    return out


# ---------------------------------------------------------------------------
# Viterbi
# ---------------------------------------------------------------------------

_TRELLIS = {}


def _trellis(code: ConvCode):
    key = (code.K, code.gens)
    if key not in _TRELLIS:
        S = 1 << (code.K - 1)
        taps = code.taps()
        sgn = np.empty((S, 2, code.n))                  # +1 for coded 0, -1 for coded 1
        for s in range(S):
            for b in range(2):
                reg = [b] + [(s >> (code.K - 2 - i)) & 1 for i in range(code.K - 1)]
                c = (taps @ np.array(reg)) % 2
                sgn[s, b] = 1 - 2 * c
        ns = np.arange(S)
        ps0 = (ns << 1) & (S - 1)                       # predecessors of next state ns
        ps1 = ps0 | 1
        bit = ns >> (code.K - 2) if code.K > 1 else np.zeros(S, dtype=int)
        _TRELLIS[key] = (S, sgn, ps0, ps1, bit)
    return _TRELLIS[key]


@dataclass
class ViterbiResult:
    bits: np.ndarray               # decoded information bits (tail removed if terminated)
    path_metric: float             # normalised: best metric / sum|LLR| in [-1, 1]
    reencode_mismatch: float       # fraction of non-erased received bits disagreeing with re-encoding
    n_steps: int


def viterbi(llrs: np.ndarray, code: ConvCode, terminated: bool = True, start_known: bool = True,
            phase: int = 0, u_prior: Optional[np.ndarray] = None) -> ViterbiResult:
    """Soft-decision Viterbi over the (depunctured) LLR stream.
    u_prior: optional a-priori LLRs (ln P(0)/P(1)) of the INPUT bits, one per trellis step
    (0 = no information); used by the iterative concatenated decoder to pin bits that the
    outer Reed-Solomon decoder has already corrected."""
    L = depuncture(llrs, code, phase)                   # (T, n)
    T = L.shape[0]
    S, sgn, ps0, ps1, bit = _trellis(code)
    if T == 0:
        return ViterbiResult(np.zeros(0, dtype=np.uint8), -1.0, 1.0, 0)
    bm_all = np.einsum("sbn,tn->tsb", sgn, L) * 0.5     # (T, S, 2) correlation branch metrics
    if u_prior is not None:
        up = np.zeros(T)
        up[:min(T, len(u_prior))] = np.asarray(u_prior, dtype=np.float64)[:T]
        bm_all = bm_all + 0.5 * up[:, None, None] * np.array([1.0, -1.0])[None, None, :]
    pm = np.zeros(S) if not start_known else np.where(np.arange(S) == 0, 0.0, -1e18)
    dec = np.empty((T, S), dtype=np.uint8)
    for t in range(T):
        bm = bm_all[t]
        m0 = pm[ps0] + bm[ps0, bit]
        m1 = pm[ps1] + bm[ps1, bit]
        choose1 = m1 > m0
        pm = np.where(choose1, m1, m0)
        dec[t] = choose1
        pm -= pm.max()                                  # keep metrics bounded
    # traceback
    s = 0 if terminated else int(np.argmax(pm))
    u = np.empty(T, dtype=np.uint8)
    for t in range(T - 1, -1, -1):
        u[t] = bit[s]
        s = ps1[s] if dec[t, s] else ps0[s]
    info = u[:T - (code.K - 1)] if terminated and T >= code.K - 1 else u
    # quality: re-encode the whole decoded input path and compare with the received hard decisions
    tn = mother_encode(u, code, terminate=False)
    mask = puncture_mask(code, T, phase)
    ref = tn[mask]
    rx = np.asarray(llrs, dtype=np.float64)[:len(ref)]
    hard = (rx < 0).astype(np.uint8)
    nz = rx != 0
    mismatch = float(np.mean(hard[nz] != ref[nz])) if np.any(nz) else 1.0
    agree = np.sum(np.abs(rx) * np.where(hard == ref, 1, -1))
    pmn = float(agree / (np.sum(np.abs(rx)) + 1e-12))
    return ViterbiResult(info.astype(np.uint8), pmn, mismatch, T)


def decode_unknown_phase(llrs: np.ndarray, code: ConvCode, terminated: bool = True):
    """Try every puncturing / symbol phase; returns (best ViterbiResult, phase, bit offset)."""
    best = None
    period_bits = code.n if code.puncture is None else int(np.array(code.puncture).sum())
    period_cols = 1 if code.puncture is None else len(code.puncture[0])
    for off in range(period_bits):
        for ph in range(period_cols):
            r = viterbi(llrs[off:], code, terminated=terminated and off == 0, start_known=off == 0, phase=ph)
            key = (r.reencode_mismatch, -r.path_metric)
            if best is None or key < best[0]:
                best = (key, r, ph, off)
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# Syndrome (parity-check) test
# ---------------------------------------------------------------------------

def syndrome_stats(hard: np.ndarray, code: ConvCode, offset: int = 0) -> Tuple[float, int]:
    """(fraction of violated parity checks, number of checks used) for an UNPUNCTURED rate-1/n
    code whose first coded tuple starts at `offset`. Checks whose span is all zeros (padding,
    interleaver flush) are skipped: the all-zero sequence satisfies every code."""
    if code.puncture is not None:
        raise ValueError("syndrome test is defined here for unpunctured codes")
    h = np.asarray(hard[offset:], dtype=np.int64) & 1
    T = len(h) // code.n
    if T < 4 * code.K:
        return 0.5, 0
    c = h[:T * code.n].reshape(T, code.n)
    taps = code.taps().astype(np.int64)
    active = np.convolve(c.any(axis=1).astype(np.int64), np.ones(code.K, dtype=np.int64))[code.K - 1:T] > 0
    if not active.any():
        return 0.5, 0
    viol = []
    for j in range(1, code.n):
        s = (np.convolve(c[:, 0], taps[j]) + np.convolve(c[:, j], taps[0]))[code.K - 1:T] % 2
        viol.append(s[active])
    v = np.concatenate(viol)
    return float(np.mean(v)), int(len(v))


def syndrome_weight(hard: np.ndarray, code: ConvCode, offset: int = 0) -> float:
    """~2*BER when (code, offset) is right, ~0.5 when wrong (see syndrome_stats)."""
    return syndrome_stats(hard, code, offset)[0]


_BASELINE = {}


def random_mismatch_baseline(code: ConvCode, n_bits: int = 2400) -> float:
    """Re-encoding mismatch of Viterbi on random bits (what a wrong hypothesis looks like).
    High-rate punctured codes have few parity bits, so random data already fits them fairly well."""
    if code.name not in _BASELINE:
        rng = np.random.default_rng(12345)
        x = rng.integers(0, 2, n_bits).astype(np.uint8)
        _BASELINE[code.name] = viterbi(1.0 - 2.0 * x, code, terminated=False, start_known=False).reencode_mismatch
    return _BASELINE[code.name]


def unpunctured_codes() -> List[ConvCode]:
    return [c for c in LIBRARY if c.puncture is None]
