"""
FEC-family evidence and blind convolutional-code parameters.

* Family classification (Swaminathan et al. 2017 architecture): uncoded / convolutional /
  block-coded, from measurable tests on the (de-interleaved) hard bits:
    - convolutional: parity-check (syndrome) weight of every unpunctured library code and
      phase; ~2 x BER when right, ~0.5 otherwise
    - block: Hamming syndrome-failure rate at every alignment; Reed-Solomon decoder success
    - uncoded: a library sync word found in clear with a validating CRC
  The scores are turned into probabilities; the final decision is left to decoding + CRC.
* Blind rate / constraint length (Marazin, Gautier & Burel 2011): the GF(2) rank profile of
  the stream reshaped into L-bit rows is full rank until L reaches n*K (rate k/n, constraint
  length K), then the deficiency grows by (n - k) every n columns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.evidence import Evidence
from fec import block_codes, convolutional


@dataclass
class ConvBlindEstimate:
    n: int
    k: int
    K: int
    first_deficient_L: int
    slope: float
    description: str


def blind_conv_params(profile: Dict[int, float]) -> Optional[ConvBlindEstimate]:
    """From {L: normalised deficiency}. Deficiency appears only at L = multiples of n, first at
    L0 = n (m_dual + 1) where m_dual ~ (K - 1)/(n - 1) is the memory of the dual (parity-check)
    code, and grows by (n - k) per n columns. Valid at low BER, non-degenerate rate-k/n codes."""
    d = {L: int(round(v * L)) for L, v in profile.items()}
    defic = [L for L in sorted(d) if d[L] > 0]
    if len(defic) < 3:
        return None
    gaps = np.diff(defic)
    vals, cnt = np.unique(gaps, return_counts=True)
    n = int(vals[np.argmax(cnt)])
    if n < 2 or n > 8:
        return None
    on = [L for L in defic if (L - defic[0]) % n == 0]
    L0, last = on[0], on[-1]
    slope = (d[last] - d[L0]) / max(1, last - L0)          # ~ (n - k) / n
    k = int(round(n * (1 - slope)))
    if not (1 <= k < n):
        return None
    K = int(round((n - 1) * (L0 / n - 1) + 1)) if k == 1 else int(round(L0 / n))
    desc = (f"Rank profile (Marazin-Gautier-Burel): deficient only at multiples of {n} bits, first at "
            f"L = {L0}, slope {slope:.2f} per bit -> rate {k}/{n}, constraint length K ~ {K}")
    return ConvBlindEstimate(n, k, K, L0, float(slope), desc)


@dataclass
class StructureScan:
    conv_code: Optional[convolutional.ConvCode]
    conv_offset: int
    conv_syndrome: float                  # best syndrome weight (0.5 = random)
    block_code: Optional[block_codes.BlockCode]
    block_offset: int
    block_fail: float                     # best syndrome-failure rate (1 = random)
    conv_table: List[Tuple[str, int, float]] = field(default_factory=list)
    n_bits: int = 0
    conv_checks: int = 0                  # parity checks actually evaluated (non-zero spans)
    block_blocks: int = 0                 # non-zero blocks evaluated
    conv_inverted: bool = False           # stream is the complement of a codeword stream

    @property
    def conv_z(self) -> float:
        """Standard deviations below the random syndrome weight 0.5 (~sqrt(N) checks)."""
        if self.conv_code is None:
            return 0.0
        return float((0.5 - self.conv_syndrome) / (0.5 / np.sqrt(max(1, self.conv_checks))))

    @property
    def block_z(self) -> float:
        if self.block_code is None:
            return 0.0
        rnd = 1 - 2.0 ** -(self.block_code.n - self.block_code.k)
        return float((rnd - self.block_fail) / np.sqrt(rnd * (1 - rnd) / max(1, self.block_blocks)))

    # strength in [0, 1]: 0 below ~6 sigma (best-of-many-tests false alarms), 1 above ~16 sigma
    @property
    def conv_strength(self) -> float:
        return float(np.clip((self.conv_z - 6) / 10, 0, 1))

    @property
    def block_strength(self) -> float:
        return float(np.clip((self.block_z - 6) / 10, 0, 1))

    @property
    def strength(self) -> float:
        return max(self.conv_strength, self.block_strength)


def structure_scan(hard: np.ndarray) -> StructureScan:
    """Cheap code-structure tests (a few ms): used for every de-interleaver hypothesis.
    Candidates are compared by significance (sigma below the random expectation), so a test
    run on few active checks cannot win on a lucky low rate."""
    best_c, table, best_cz, inv = (None, 0, 0.5, 0), [], -np.inf, False
    for code in convolutional.unpunctured_codes():
        # Complementing a codeword stream flips every parity check of a code whose generators
        # have different weight parities (w -> 1 - w): a weight near 1 is the inverted stream.
        flips = len({bin(g).count("1") % 2 for g in code.gens}) > 1
        for off in range(code.n):
            w, n = convolutional.syndrome_stats(hard, code, off)
            table.append((code.name, off, w))
            ww, is_inv = (1 - w, True) if flips and w > 0.5 else (w, False)
            z = (0.5 - ww) / (0.5 / np.sqrt(max(n, 1)))
            if n and z > best_cz:
                best_c, best_cz, inv = (code, off, ww, n), z, is_inv
    best_b, best_bz = (None, 0, 1.0, 0), -np.inf
    for code in block_codes.LIBRARY:
        rnd = 1 - 2.0 ** -(code.n - code.k)
        for off in range(code.n):
            f, n = block_codes.syndrome_stats(hard, code, off)
            z = (rnd - f) / np.sqrt(rnd * (1 - rnd) / max(n, 1))
            if n and z > best_bz:
                best_b, best_bz = (code, off, f, n), z
    return StructureScan(best_c[0], best_c[1], best_c[2], best_b[0], best_b[1], best_b[2], table, len(hard),
                         best_c[3], best_b[3], inv)


@dataclass
class FamilyDecision:
    probabilities: Dict[str, float]
    family: str
    evidence: List[Evidence]


def classify_family(scan: StructureScan, rs_ok_fraction: float = 0.0, uncoded_crc_ok: bool = False,
                    uncoded_sync: bool = False) -> FamilyDecision:
    s_conv = scan.conv_strength
    s_block = max(scan.block_strength, rs_ok_fraction)
    # nothing coded found and no sync in clear: uncoded (or scrambled / unknown code) stays plausible
    s_unc = 1.0 if uncoded_crc_ok else (0.5 if uncoded_sync else 0.3 * (1 - max(s_conv, s_block)))
    raw = {"Uncoded": s_unc + 1e-3, "Convolutional code": s_conv + 1e-3, "Block code (Hamming / RS)": s_block + 1e-3}
    tot = sum(raw.values())
    probs = {k: v / tot for k, v in raw.items()}
    fam = max(probs, key=probs.get)
    ev = [Evidence(f"Convolutional parity checks: best {scan.conv_code.name if scan.conv_code else '-'} "
                   f"(phase {scan.conv_offset}) syndrome weight {scan.conv_syndrome:.3f} (0.5 = random; {scan.conv_z:.0f} sigma)",
                   "FEC_SYNDROME", s_conv),
          Evidence(f"Block-code checks: best {scan.block_code.name if scan.block_code else '-'} "
                   f"(offset {scan.block_offset}) syndrome failures {scan.block_fail:.3f} ({scan.block_z:.0f} sigma); RS blocks decoded "
                   f"{rs_ok_fraction:.0%}", "FEC_SYNDROME", s_block),
          Evidence("Sync word + CRC valid without decoding" if uncoded_crc_ok else
                   ("Sync word visible in clear (systematic code or uncoded)" if uncoded_sync else
                    "No sync word in clear"), "SYNC_WORD", s_unc)]
    return FamilyDecision(probs, fam, ev)
