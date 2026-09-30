"""
Transmit-side channel coding for synthetic test vectors (Phase 4 validation).

    payload -> frame (SYNC | LEN | PAYLOAD | CRC-16)          synthetic/framing.py
            -> FEC encode        (convolutional / Reed-Solomon / Hamming / none)
            -> interleave        (block R x C / convolutional B, M / none)
            -> bits to the modulator

sync_outside=True keeps the sync word uncoded in front of the coded + interleaved block
(CCSDS CADU style: ASM | coded frame); otherwise the whole frame, sync included, is coded.
Receivers may only use the truth dict for validation.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from deinterleaving import block as blk
from deinterleaving import convolutional as cvi
from deinterleaving import diagonal as dgi
from deinterleaving import pseudorandom as pri
from fec import block_codes, concatenated, convolutional, ldpc, reed_solomon
from synthetic.framing import SYNC_LEN, build_frame


@dataclass
class CodingSpec:
    fec: Optional[str] = None            # conv library name | RS profile | "Hamming(7,4)" | LDPC library name |
                                         # "CONCAT:<fec.concatenated.PROFILES key>" | None
    interleaver: Optional[str] = None    # "block:RxC" | "conv:B,M" | "diag:RxC[,-1]" | "pr:P[,seed]" | None
    sync_outside: bool = False

    def describe(self) -> str:
        return (f"FEC {self.fec or 'none'}, interleaver {self.interleaver or 'none'}, "
                f"sync {'outside (uncoded)' if self.sync_outside else 'inside the coded block'}")


def fec_encode(bits: np.ndarray, fec: Optional[str]) -> np.ndarray:
    if not fec:
        return bits
    if fec.startswith("CC"):
        return convolutional.encode(bits, convolutional.get_code(fec))
    if fec.startswith("RS"):
        return reed_solomon.encode(bits, reed_solomon.get_profile(fec))
    if fec.startswith("Hamming"):
        return block_codes.encode(bits, block_codes.get_code(fec))
    if fec.startswith("LDPC"):
        return ldpc.encode(bits, ldpc.get_code(fec))
    if fec.startswith("CONCAT:"):
        return concatenated.encode(bits, concatenated.PROFILES[fec.split(":", 1)[1]])
    raise ValueError(f"unknown FEC '{fec}'")


def interleave(bits: np.ndarray, spec: Optional[str]) -> np.ndarray:
    if not spec:
        return bits
    kind, dims = spec.split(":")
    if kind == "block":
        R, C = (int(v) for v in dims.lower().split("x"))
        return blk.interleave(bits, R, C)
    if kind == "conv":
        B, M = (int(v) for v in dims.split(","))
        return cvi.interleave(bits, B, M)
    if kind == "diag":
        rc, *d = dims.split(",")
        R, C = (int(v) for v in rc.lower().split("x"))
        return dgi.interleave(bits, R, C, int(d[0]) if d else 1)
    if kind == "pr":
        P, *seed = dims.split(",")
        P = int(P) if P != "all" else len(bits)
        h = pri.hypotheses(P)[0] if not seed else pri.PRHypothesis(P, pri.PRIMITIVE_TAPS[pri.lfsr_bits(P)][0],
                                                                  int(seed[0]))
        return pri.interleave(bits, h.perm())
    raise ValueError(f"unknown interleaver '{spec}'")


def encode_frame(payload: bytes, coding: CodingSpec, bits_per_symbol: int):
    frame, layout = build_frame(payload, 1)
    if coding.sync_outside:
        head, body = frame[:SYNC_LEN], frame[SYNC_LEN:]
    else:
        head, body = frame[:0], frame
    coded = interleave(fec_encode(body, coding.fec), coding.interleaver)
    bits = np.concatenate([head, coded]).astype(np.uint8)
    pad = (-len(bits)) % bits_per_symbol
    bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    info = {**asdict(coding), "frame_bits_uncoded": int(len(frame)), "coded_bits": int(len(coded)),
            "transmitted_bits": int(len(bits)), "layout": layout.to_dict()}
    return bits, info
