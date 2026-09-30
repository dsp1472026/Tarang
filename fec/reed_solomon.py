"""
Reed-Solomon codes over GF(2^8) (known-profile decoding, reedsolo backend).

Profiles are (n, k) with primitive polynomial 0x11D and first consecutive root 0
(DVB / generic convention). A message longer than k bytes is split into k-byte blocks;
the last block is shortened (fewer data bytes, same number of parity bytes), which is how
short frames are carried by long RS codes.

Blind use (Ramabadran / Liu-Pan-Lei recommend known hypotheses first): the receiver
tries every profile x 8 bit alignments x {MSB, LSB}-first byte packing and scores each by
decoder success and corrected-symbol count; the final shortened-block length is searched.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from reedsolo import ReedSolomonError, RSCodec


@dataclass(frozen=True)
class RSProfile:
    name: str
    n: int
    k: int
    prim: int = 0x11D
    fcr: int = 0

    @property
    def nsym(self) -> int:
        return self.n - self.k

    @property
    def t(self) -> int:
        return self.nsym // 2

    def codec(self) -> RSCodec:
        return _codec(self.nsym, self.prim, self.fcr)

    def describe(self) -> str:
        return f"RS({self.n},{self.k}) over GF(2^8), p(x)=0x{self.prim:X}, fcr={self.fcr}, corrects {self.t} symbols"


_CODECS = {}


def _codec(nsym, prim, fcr):
    key = (nsym, prim, fcr)
    if key not in _CODECS:
        _CODECS[key] = RSCodec(nsym, nsize=255, prim=prim, fcr=fcr, c_exp=8)
    return _CODECS[key]


PROFILES: List[RSProfile] = [
    RSProfile("RS(255,223) [CCSDS-size]", 255, 223),
    RSProfile("RS(255,239)", 255, 239),
    RSProfile("RS(204,188) [DVB, shortened 255,239]", 204, 188),
    RSProfile("RS(255,247)", 255, 247),
]
BY_NAME = {p.name: p for p in PROFILES}


def get_profile(name: str) -> RSProfile:
    if name in BY_NAME:
        return BY_NAME[name]
    for p in PROFILES:
        if p.name.startswith(name.split(" ")[0]):
            return p
    raise KeyError(f"unknown RS profile '{name}'")


def encode_bytes(data: bytes, prof: RSProfile) -> bytes:
    rs = prof.codec()
    out = bytearray()
    for i in range(0, len(data), prof.k):
        out += rs.encode(bytearray(data[i:i + prof.k]))
    return bytes(out)


def encode(bits: np.ndarray, prof: RSProfile) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    data = np.packbits(np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])).tobytes()
    return np.unpackbits(np.frombuffer(encode_bytes(data, prof), dtype=np.uint8))


@dataclass
class RSDecode:
    ok: bool                      # every block decoded
    data_bits: np.ndarray
    blocks: int
    failed_blocks: int
    corrected_symbols: int
    bit_offset: int = 0
    lsb_first: bool = False
    last_block_len: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        if self.blocks == 0:
            return 0.0
        return (self.blocks - self.failed_blocks) / self.blocks


def _pack(bits: np.ndarray, lsb_first: bool) -> bytes:
    n = len(bits) // 8 * 8
    b = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, 8)
    if lsb_first:
        b = b[:, ::-1]
    return np.packbits(b.ravel()).tobytes()


def _decode_block(rs, block: bytes):
    try:
        msg, full, errata = rs.decode(bytearray(block))
        if not any(full):                 # decoding to the zero codeword proves nothing (padding)
            return None, -1
        return bytes(msg), len(errata)
    except ReedSolomonError:
        return None, -1


def decode(bits: np.ndarray, prof: RSProfile, bit_offset: int = 0, lsb_first: bool = False,
           total_bytes: Optional[int] = None) -> RSDecode:
    """Decode consecutive codewords. total_bytes = coded length if known; otherwise the last
    (shortened) block length is searched over the remaining bytes (fewest corrections wins)."""
    data = _pack(np.asarray(bits)[bit_offset:], lsb_first)
    if total_bytes is not None:
        data = data[:total_bytes]
    rs = prof.codec()
    out, fails, corr, blocks = bytearray(), 0, 0, 0
    i = 0
    last_len = 0
    while len(data) - i >= prof.n:
        m, c = _decode_block(rs, data[i:i + prof.n])
        blocks += 1
        if m is None:
            fails += 1
            m = data[i:i + prof.k]
        else:
            corr += c
        out += m
        i += prof.n
        last_len = prof.n
    rem = len(data) - i
    if rem > prof.nsym:
        if total_bytes is not None or (blocks and fails == blocks):
            cands = [rem]                              # known length / wrong hypothesis: no length search
        else:
            cands = range(rem, prof.nsym, -1)
        best = None
        for L in cands:
            m, c = _decode_block(rs, data[i:i + L])
            if m is not None and (best is None or c < best[1]):
                best = (m, c, L)
                if c == 0:
                    break
        blocks += 1
        if best is None:
            fails += 1
            out += data[i:i + rem - prof.nsym]
        else:
            out += best[0]
            corr += best[1]
            last_len = best[2]
    bits_out = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8)) if out else np.zeros(0, np.uint8)
    return RSDecode(blocks > 0 and fails == 0, bits_out, blocks, fails, corr, bit_offset, lsb_first, last_len)


def search(bits: np.ndarray, profiles: Optional[List[RSProfile]] = None) -> List[Tuple[RSProfile, RSDecode]]:
    """All profiles x 8 bit offsets x MSB/LSB packing, best first (decoded blocks, then fewest corrections)."""
    res = []
    for prof in profiles or PROFILES:
        if len(bits) < 8 * (prof.nsym + 1):
            continue
        for off in range(8):
            for lsb in (False, True):
                d = decode(bits, prof, off, lsb)
                res.append((prof, d))
    res.sort(key=lambda pd: (-pd[1].score, pd[1].corrected_symbols, pd[1].bit_offset, pd[1].lsb_first))
    return res
