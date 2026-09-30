"""
Frame / protocol analysis after sync detection (L11).

For every sync occurrence the frame layout is hypothesised and validated by CRC:
    length field : 16-bit big-endian, 16-bit little-endian, 8-bit, or none (fixed length =
                   spacing between syncs / to the end of the stream)
    CRC          : every catalogue CRC, stored big- or little-endian after the payload,
                   covering [length + payload], [payload] or [sync + length + payload]
The layout that validates the most frames wins; frames are then parsed with it. Without a
validating layout the frames are still reported (CRC 'not verified'), never claimed valid.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from bitstream.sync_search import SyncCandidate
from core.evidence import Evidence
from protocol.crc import CATALOGUE, CRCSpec, compute


MAX_FALSE_ALARMS = 1e-2       # expected chance validations tolerated over the whole layout search


@dataclass(frozen=True)
class Layout:
    length_field: str               # "u16be" | "u16le" | "u8" | "none"
    crc: Optional[str]              # CRC spec name or None
    crc_endian: str = "be"
    crc_coverage: str = "len+payload"   # "len+payload" | "payload" | "sync+len+payload"

    def describe(self, sync_bits: int) -> str:
        lf = {"u16be": "LEN 16 bit big-endian", "u16le": "LEN 16 bit little-endian", "u8": "LEN 8 bit",
              "none": "no length field (fixed length)"}[self.length_field]
        c = f"{self.crc} ({self.crc_endian.upper()}) over {self.crc_coverage}" if self.crc else "no CRC found"
        return f"SYNC {sync_bits} bit | {lf} | PAYLOAD | {c}"


@dataclass
class FrameRecord:
    index: int
    start_bit: int
    sync_name: str
    sync_errors: int
    length: Optional[int]
    payload: bytes
    crc_name: Optional[str]
    crc_received: Optional[int]
    crc_computed: Optional[int]
    end_bit: int

    @property
    def crc_ok(self) -> Optional[bool]:
        if self.crc_name is None:
            return None
        return self.crc_received == self.crc_computed and not _trivial(self.payload, self.crc_received)

    @property
    def text(self) -> str:
        return self.payload.decode("ascii", errors="replace")

    @property
    def printable_ratio(self) -> float:
        if not self.payload:
            return 0.0
        return sum(32 <= b < 127 or b in (9, 10, 13) for b in self.payload) / len(self.payload)


@dataclass
class FrameAnalysis:
    layout: Optional[Layout]
    frames: List[FrameRecord]
    sync: Optional[SyncCandidate]
    evidence: List[Evidence] = field(default_factory=list)
    layouts_tested: int = 0

    @property
    def valid_frames(self) -> int:
        return sum(1 for f in self.frames if f.crc_ok)

    @property
    def payload(self) -> bytes:
        good = [f for f in self.frames if f.crc_ok] or self.frames
        return good[0].payload if good else b""


def _trivial(payload: bytes, crc_received: Optional[int]) -> bool:
    """An all-zero (or empty) payload with a zero CRC 'validates' under every CRC with zero init and
    xorout: an all-zero decoder output (e.g. an LDPC decoder converging to the zero codeword) is
    not evidence of a frame."""
    return crc_received == 0 and not any(payload)


def _bytes(bits: np.ndarray) -> bytes:
    n = len(bits) // 8 * 8
    return np.packbits(np.asarray(bits[:n], dtype=np.uint8)).tobytes()


def _parse_one(bits: np.ndarray, start: int, L_sync: int, layout: Layout, spec: Optional[CRCSpec],
               fixed_len_bits: Optional[int]) -> Optional[Tuple[Optional[int], bytes, Optional[int], Optional[int], int]]:
    pos = start + L_sync
    lf_bits = {"u16be": 16, "u16le": 16, "u8": 8, "none": 0}[layout.length_field]
    nb = spec.nbytes if spec else 0
    if layout.length_field == "none":
        if fixed_len_bits is None:
            return None
        body_bits = fixed_len_bits - L_sync - 8 * nb
        if body_bits < 8:
            return None
        length = body_bits // 8
        hdr = b""
    else:
        if pos + lf_bits > len(bits):
            return None
        hdr = _bytes(bits[pos:pos + lf_bits])
        length = (int.from_bytes(hdr, "big") if layout.length_field in ("u16be", "u8")
                  else int.from_bytes(hdr, "little"))
        if pos + lf_bits + 8 * (length + nb) > len(bits):
            return None
    p0 = pos + lf_bits
    payload = _bytes(bits[p0:p0 + 8 * length])
    end = p0 + 8 * (length + nb)
    if spec is None:
        return length if lf_bits else None, payload, None, None, end
    crc_bytes = _bytes(bits[p0 + 8 * length:end])
    rx = int.from_bytes(crc_bytes, "big" if layout.crc_endian == "be" else "little")
    cover = {"len+payload": hdr + payload, "payload": payload,
             "sync+len+payload": _bytes(bits[start:start + L_sync]) + hdr + payload}[layout.crc_coverage]
    if layout.crc_coverage == "sync+len+payload" and L_sync % 8:
        return None
    return (length if lf_bits else None), payload, rx, compute(spec, cover), end


def analyze(bits: np.ndarray, sync: Optional[SyncCandidate], sync_word: Optional[np.ndarray] = None) -> FrameAnalysis:
    bits = np.asarray(bits, dtype=np.uint8) & 1
    if sync is None or not sync.detected:
        return FrameAnalysis(None, [], sync, [Evidence("No sync word detected: frame boundaries unknown",
                                                          "FRAMING", 0.0)])
    if sync.inverted:
        bits = bits ^ 1
    L = sync.length
    occ = sync.occurrences or [sync.offset]
    fixed = None
    if sync.period:
        fixed = sync.period
    elif len(occ) == 1:
        fixed = len(bits) - occ[0]

    layouts = []
    for lf in ("u16be", "u16le", "u8", "none"):
        for spec in CATALOGUE:
            for en in ("be", "le"):
                for cov in ("len+payload", "payload", "sync+len+payload"):
                    if lf == "none" and cov == "len+payload":
                        continue
                    layouts.append((Layout(lf, spec.name, en, cov), spec))
    # A CRC of w bits matches random data with probability 2^-w, and hundreds of layouts are
    # tried, so a layout is accepted only if chance cannot explain it: expected number of
    # random validations (layouts of that width x 2^-w per frame ^ frames) < MAX_FALSE_ALARMS.
    n_width = {}
    for _, spec in layouts:
        n_width[spec.width] = n_width.get(spec.width, 0) + 1
    best = None
    for lay, spec in layouts:
        ok = 0
        for s in occ:
            r = _parse_one(bits, s, L, lay, spec, fixed)
            if r and r[2] == r[3] and not _trivial(r[1], r[2]):
                ok += 1
        if not ok:
            continue
        p_false = n_width[spec.width] * (2.0 ** -spec.width) ** ok
        if p_false > MAX_FALSE_ALARMS:
            continue
        if best is None or ok > best[0] or (ok == best[0] and spec.width > best[2].width):
            best = (ok, lay, spec)
            if ok == len(occ) and spec.width >= 16:
                break
    ev = []
    if best:
        lay, spec = best[1], best[2]
    else:
        lay, spec = Layout("u16be", None), None
    frames = []
    for i, s in enumerate(occ):
        r = _parse_one(bits, s, L, lay, spec, fixed)
        if r is None and lay.length_field != "none":
            r = _parse_one(bits, s, L, Layout("none", None), None, (occ[i + 1] - s) if i + 1 < len(occ) else fixed)
        if r is None:
            continue
        length, payload, rx, cc, end = r
        err = int(np.count_nonzero(bits[s:s + L] != (sync_word if sync_word is not None else bits[s:s + L])))
        frames.append(FrameRecord(i, s, sync.name, err, length, payload, spec.name if spec else None, rx, cc, end))
    fa = FrameAnalysis(lay if best else None, frames, sync, ev, len(layouts))
    if best:
        ev.append(Evidence(f"Frame layout {lay.describe(L)} validated by CRC on {fa.valid_frames}/{len(frames)} "
                           f"frame(s) ({len(layouts)} layouts tested)", "CRC", fa.valid_frames / max(1, len(frames))))
    else:
        ev.append(Evidence(f"Sync '{sync.name}' found at {len(occ)} position(s) but no length/CRC layout from the "
                           f"catalogue validates; frames reported unverified", "FRAMING", 0.2))
    if sync.period:
        ev.append(Evidence(f"Sync recurs every {sync.period} bits ({len(occ)} occurrences): frame period",
                           "SYNC_PERIOD", 0.8))
    return fa
