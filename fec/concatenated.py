"""
Concatenated codes: outer Reed-Solomon + outer byte interleaver + inner convolutional code.

Transmit chain (encode):
    data bytes -> RS encoding (k-byte blocks, last one shortened)
               -> outer byte interleaver:
                    "forney:I,M"   convolutional (Forney) byte interleaver, I branches, branch j
                                   delays j*M bytes (DVB-S structure uses RS(204,188), I = 12,
                                   M = 17); flushed with (I-1)*M*I zero bytes
                    "depth:I"      CCSDS-style symbol interleaving of I codewords: the data is split
                                   into I equal (shortened) codewords whose bytes are sent column
                                   by column (byte j of codeword 0, of codeword 1, ...)
                    "none"
               -> bits (MSB first) -> inner convolutional code (terminated)
These reproduce the *structure* of the DVB-S / CCSDS 131.0-B concatenated schemes; the energy
dispersal / pseudo-randomiser, sync-byte handling and CCSDS symbol inversion of those standards
are not modelled.

Decoding (iterative, `decode`):
    1. soft Viterbi of the inner code;
    2. bytes -> outer de-interleaving -> RS decoding of every codeword (errors only);
    3. codewords that fail are retried with ERASURES: the bytes whose decoded bits have the
       lowest channel reliability (sum of |LLR| of the coded bits they influence) are erased
       (reedsolo errors-and-erasures decoding, keeping a 4-symbol detection margin);
    4. feedback: the bits of every successfully decoded codeword are re-interleaved to their
       positions in the inner code's input and given a strong a-priori LLR; Viterbi is run again,
       which also corrects neighbouring bits; steps 2-4 repeat (max 3 iterations) while the
       number of decoded codewords increases.
Blind use (`search_outer`): after an inner convolutional code has been identified, every RS
profile x outer interleaver hypothesis is tried on the Viterbi output; decoded codewords and
the frame CRC decide.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from reedsolo import ReedSolomonError

from deinterleaving import convolutional as cvi
from fec import convolutional as cc
from fec import reed_solomon as rs

PRIOR_LLR = 30.0
ERASURE_MARGIN = 4          # parity symbols kept for error detection in errors-and-erasures decoding


@dataclass(frozen=True)
class ConcatSpec:
    rs_name: str
    outer: str                 # "none" | "forney:I,M" | "depth:I"
    inner: str                 # convolutional library name

    def describe(self) -> str:
        return f"outer {self.rs_name}, outer interleaver {self.outer}, inner {self.inner}"


PROFILES = {
    "DVB-S structure": ConcatSpec("RS(204,188) [DVB, shortened 255,239]", "forney:12,17",
                                  "CC K=7 (171,133) r1/2 [NASA/CCSDS]"),
    "CCSDS structure (I=4)": ConcatSpec("RS(255,223) [CCSDS-size]", "depth:4", "CC K=7 (171,133) r1/2 [NASA/CCSDS]"),
    "CCSDS structure (I=1)": ConcatSpec("RS(255,223) [CCSDS-size]", "none", "CC K=7 (171,133) r1/2 [NASA/CCSDS]"),
}


# ---------------------------------------------------------------------------
# Outer layer (bytes)
# ---------------------------------------------------------------------------

def _rs_codewords(data: bytes, prof: rs.RSProfile, outer: str) -> List[bytes]:
    codec = prof.codec()
    if outer.startswith("depth:"):
        I = int(outer.split(":")[1])
        L = int(np.ceil(len(data) / I))
        data = data + bytes(I * L - len(data))
        return [bytes(codec.encode(bytearray(data[i * L:(i + 1) * L]))) for i in range(I)]
    return [bytes(codec.encode(bytearray(data[i:i + prof.k]))) for i in range(0, len(data), prof.k)]


def outer_interleave(codewords: List[bytes], outer: str) -> np.ndarray:
    if outer.startswith("depth:"):
        return np.frombuffer(b"".join(codewords), dtype=np.uint8).reshape(len(codewords), -1).T.reshape(-1).copy()
    stream = np.frombuffer(b"".join(codewords), dtype=np.uint8).copy()
    if outer.startswith("forney:"):
        I, M = (int(v) for v in outer.split(":")[1].split(","))
        return cvi.interleave(stream, I, M)
    return stream


def outer_deinterleave(stream: np.ndarray, outer: str, n_codewords: Optional[int] = None) -> np.ndarray:
    s = np.asarray(stream, dtype=np.uint8)
    if outer.startswith("depth:"):
        I = int(outer.split(":")[1])
        L = len(s) // I
        return s[:I * L].reshape(L, I).T.reshape(-1)
    if outer.startswith("forney:"):
        I, M = (int(v) for v in outer.split(":")[1].split(","))
        return cvi.deinterleave(s, I, M)
    return s


def outer_positions(n_bytes: int, outer: str) -> np.ndarray:
    """For de-interleaved byte index j, the index of that byte in the interleaved stream."""
    idx = np.arange(n_bytes + (cvi.flush_bits(*[int(v) for v in outer.split(":")[1].split(",")])
                               if outer.startswith("forney:") else 0))
    if outer.startswith("depth:"):
        I = int(outer.split(":")[1])
        L = n_bytes // I
        return np.arange(I * L).reshape(L, I).T.reshape(-1)
    if outer.startswith("forney:"):
        I, M = (int(v) for v in outer.split(":")[1].split(","))
        return cvi.deinterleave(idx.astype(np.int64), I, M)[:n_bytes]
    return idx[:n_bytes]


def encode(bits: np.ndarray, spec: ConcatSpec) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    data = np.packbits(np.concatenate([bits, np.zeros((-len(bits)) % 8, np.uint8)])).tobytes()
    prof = rs.get_profile(spec.rs_name)
    stream = outer_interleave(_rs_codewords(data, prof, spec.outer), spec.outer)
    return cc.encode(np.unpackbits(stream), cc.get_code(spec.inner))


# ---------------------------------------------------------------------------
# Iterative decoding
# ---------------------------------------------------------------------------

@dataclass
class ConcatDecode:
    spec: ConcatSpec
    data_bits: np.ndarray
    codewords: int
    decoded: int
    corrected_symbols: int
    erasure_decodes: int
    iterations: int
    history: List[int] = field(default_factory=list)      # decoded codewords per iteration

    @property
    def ok(self) -> bool:
        return self.codewords > 0 and self.decoded == self.codewords


def _split_codewords(deint: np.ndarray, prof: rs.RSProfile, outer: str, total: Optional[int]):
    """Codeword byte ranges in the de-interleaved stream. All-zero ranges (padding, interleaver
    flush) are dropped: the zero word is a codeword of every RS code and proves nothing."""
    return [(a, b) for a, b in _ranges(deint, prof, outer, total) if np.any(deint[a:b])]


def _ranges(deint: np.ndarray, prof: rs.RSProfile, outer: str, total: Optional[int]):
    n = len(deint) if total is None else min(total, len(deint))
    if outer.startswith("depth:"):
        I = int(outer.split(":")[1])
        L = n // I
        return [(i * L, (i + 1) * L) for i in range(I)] if L > prof.nsym else []
    out, i = [], 0
    while n - i >= prof.n:
        out.append((i, i + prof.n))
        i += prof.n
    if n - i > prof.nsym:
        out.append((i, n))
    return out


def _rs_try(codec, cw: bytes, rel: np.ndarray, nsym: int):
    try:
        msg, full, err = codec.decode(bytearray(cw))
        if not any(full):                 # the zero codeword proves nothing (padding / flush)
            return None, -1, False
        return bytes(msg), len(err), False
    except ReedSolomonError:
        pass
    # Errors-and-erasures: with f erasures and e errors a word is corrected when 2e + f <= nsym.
    # With f close to nsym ANY word decodes to some codeword (miscorrection), so a detection
    # margin of MARGIN parity symbols is kept: f <= nsym - MARGIN and, counted from the actual
    # changes, 2e + f <= nsym - MARGIN.
    order = np.argsort(rel)
    for ne in range(2, nsym - ERASURE_MARGIN + 1, 2):
        era = [int(p) for p in order[:ne]]
        try:
            msg, full, _ = codec.decode(bytearray(cw), erase_pos=era)
        except (ReedSolomonError, ValueError):
            continue
        diff = np.flatnonzero(np.frombuffer(bytes(full), dtype=np.uint8) != np.frombuffer(cw, dtype=np.uint8))
        e = len(set(diff.tolist()) - set(era))
        if 2 * e + ne <= nsym - ERASURE_MARGIN:
            return bytes(msg), len(diff), True
    return None, -1, False


def decode(llrs: np.ndarray, spec: ConcatSpec, total_bytes: Optional[int] = None,
           max_iter: int = 3) -> ConcatDecode:
    code = cc.get_code(spec.inner)
    prof = rs.get_profile(spec.rs_name)
    codec = prof.codec()
    llrs = np.asarray(llrs, dtype=np.float64)
    prior = None
    history: List[int] = []
    best = None
    # channel reliability of each input bit: sum |LLR| over the coded bits it drives
    T = len(llrs) // code.n
    rel_bits = np.convolve(np.abs(llrs[:T * code.n]).reshape(T, code.n).sum(axis=1), np.ones(code.K))[:T]
    for it in range(1, max_iter + 1):
        vr = cc.viterbi(llrs, code, terminated=True, start_known=True, u_prior=prior)
        stream = np.packbits(vr.bits[:len(vr.bits) // 8 * 8])
        rel_bytes = rel_bits[:len(stream) * 8].reshape(-1, 8).min(axis=1)
        deint = outer_deinterleave(stream, spec.outer)
        pos = outer_positions(len(deint), spec.outer)
        rel_d = rel_bytes[np.clip(pos, 0, len(rel_bytes) - 1)]
        ranges = _split_codewords(deint, prof, spec.outer, total_bytes)
        data, n_ok, corr, n_era = bytearray(), 0, 0, 0
        fixed = deint.copy()
        for a, b in ranges:
            msg, c, era = _rs_try(codec, bytes(deint[a:b]), rel_d[a:b], prof.nsym)
            if msg is None:
                data += bytes(deint[a:b - prof.nsym])
                continue
            n_ok += 1
            corr += c
            n_era += int(era)
            data += msg
            fixed[a:b] = np.frombuffer(bytes(codec.encode(bytearray(msg))), dtype=np.uint8)
        history.append(n_ok)
        res = ConcatDecode(spec, np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8)) if data else
                           np.zeros(0, np.uint8), len(ranges), n_ok, corr, n_era, it, list(history))
        if best is None or n_ok > best.decoded:
            best = res
        if n_ok == len(ranges) or (it > 1 and n_ok <= history[-2]):
            break
        # feedback: pin the input bits of every decoded codeword
        ok_mask = np.zeros(len(deint), dtype=bool)
        for a, b in ranges:
            cw = fixed[a:b]
            try:
                codec.decode(bytearray(cw))
                ok_mask[a:b] = True
            except ReedSolomonError:
                pass
        bits_pos = np.clip(pos[ok_mask], 0, len(stream) - 1)
        prior = np.zeros(len(vr.bits) + code.K)
        byte_bits = np.unpackbits(fixed[ok_mask])
        bit_idx = (bits_pos[:, None] * 8 + np.arange(8)[None, :]).ravel()
        prior[bit_idx] = PRIOR_LLR * (1 - 2.0 * byte_bits)
    return best


def search_outer(vit_bits: np.ndarray, inner_name: str) -> List[Tuple[ConcatSpec, int, int]]:
    """Blind outer-layer hypotheses on the inner decoder's output bits (byte aligned at 0):
    RS profile x {none, Forney byte interleaver with I*M = n, depth I = 2..8}.
    Returns [(spec, decoded codewords, codewords)] best first."""
    stream = np.packbits(np.asarray(vit_bits, dtype=np.uint8)[:len(vit_bits) // 8 * 8])
    out = []
    for prof in rs.PROFILES:
        outers = ["none"] + [f"depth:{I}" for I in range(2, 9)]
        outers += [f"forney:{I},{prof.n // I}" for I in range(2, 17) if prof.n % I == 0]
        codec = prof.codec()
        for o in outers:
            deint = outer_deinterleave(stream, o)
            ranges = _split_codewords(deint, prof, o, None)
            if not ranges:
                continue
            ok = 0
            for a, b in ranges:
                try:
                    _, full, _ = codec.decode(bytearray(deint[a:b]))
                    if any(full):             # "corrected" to the zero word = sparse padding, not data
                        ok += 1
                except ReedSolomonError:
                    pass
            if ok:
                out.append((ConcatSpec(prof.name, o, inner_name), ok, len(ranges)))
    # nested RS codes (same fcr) also decode each other: prefer the one with more parity on ties
    out.sort(key=lambda t: (-t[1] / t[2], -t[1], -rs.get_profile(t[0].rs_name).nsym))
    return out
