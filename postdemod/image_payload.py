"""
Image / file payload recovery (L11, any modulation family).

A received file is carried in a byte frame with a CCSDS attached sync marker:

    SYNC 0x1ACFFC1D | LENGTH (u32 big-endian) | PAYLOAD | CRC32      synthetic/image_framing.py

sent either uncoded or with the K=7 (171,133) rate-1/2 convolutional code (synthetic/tx_image.py).
The generic L8-L11 search does not cover this layout (it tries 8/16-bit length fields) and would
only report binary payloads as 'no decisive text', so the frame is searched for directly here:

  1. Ambiguity hypotheses. PSK / QAM receivers resolve the carrier-phase ambiguity with the
     project's PN127 sync word, which a file frame does not carry, so every M-fold rotation with
     and without conjugation is demapped. FSK: the receiver's bits and their inversion.
  2. Per hypothesis: the ASM is correlated over the raw hard bits (every bit offset, both
     polarities); then a Viterbi decode of a prefix window (both trellis phases) is searched,
     and only a hit there is decoded over the whole stream.
  3. Frame parse: LENGTH must fit in the stream; CRC-32 over SYNC + LENGTH + PAYLOAD (this project's
     synthetic/image_framing.py) or over the PAYLOAD alone (the Space Downlink Simulator,
     Image Decoding/c.py, which also sends a 0xAA preamble before the sync).
  3b. Bit labelling: besides the receivers' own Gray labels, the labels of other transmitters are
     tried on hard decisions (c.py: PSK point = exp(j 2 pi gray(i) / M), QAM axis level = gray(i),
     FSK tone = natural binary index), for every rotation / conjugation.
  4. Payload identification: encoded image (PNG, JPEG, GIF, BMP, TIFF, WebP; checked by decoding
     it with Pillow) or raw pixels behind a 5-byte header W(u16) H(u16) C(u8), C in {1, 3, 4},
     whose size must equal the length field exactly.

Acceptance: a CRC-valid frame; or, when the CRC does not match (another transmitter's checksum
convention), a sync hit with <= 2 bit errors AND a payload whose image structure is internally
consistent (it decodes completely / the raw header predicts the exact length). The second case
is reported as 'CRC not verified'.
"""

import io
import zlib
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from core.evidence import Evidence
from demodulation.psk import rotate
from dsp.constellations import get_constellation
from fec import convolutional

SYNC_WORD = bytes.fromhex("1ACFFC1D")
SYNC_BITS = np.unpackbits(np.frombuffer(SYNC_WORD, dtype=np.uint8))
SYNC_MAX_ERRORS = 2                   # of 32 bits
HEADER_BYTES = 8                      # SYNC + LENGTH
CRC_BYTES = 4
VITERBI_PREFIX_LLRS = 32768           # coded bits searched for the sync before a full decode
SYNC_PREFIX_BITS = 1 << 17
CRC_TAIL_BITS = 7                     # CRC bits a symbol-packing transmitter may drop            # bits searched for the sync under the extra label hypotheses
CONV_CODE = convolutional.get_code("CC K=7 (171,133) r1/2 [NASA/CCSDS]")
RAW_HEADER_BYTES = 5                  # W u16, H u16, C u8
MAX_RAW_DIM = 16384

MAGIC = (("PNG", b"\x89PNG\r\n\x1a\n"), ("JPEG", b"\xff\xd8\xff"), ("GIF", b"GIF8"), ("BMP", b"BM"),
         ("TIFF", b"II*\x00"), ("TIFF", b"MM\x00*"), ("WEBP", b"RIFF"))
EXT = {"PNG": ".png", "JPEG": ".jpg", "GIF": ".gif", "BMP": ".bmp", "TIFF": ".tif", "WEBP": ".webp"}


@dataclass
class ImagePayload:
    format: str                        # PNG / JPEG / ... / RAW (header + pixels) / BINARY (not an image)
    payload: bytes                     # the frame payload exactly as received
    crc_ok: bool
    crc_received: int
    crc_computed: int
    fec: str                           # "none" or the convolutional code name
    hypothesis: str                    # carrier / polarity hypothesis that carried the frame
    sync_bit: int                      # sync position in the (decoded) bit stream
    sync_errors: int
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    pixels: Optional[np.ndarray] = None    # (H, W, C) uint8 for display, None if not an image
    evidence: List[Evidence] = field(default_factory=list)

    @property
    def is_image(self) -> bool:
        return self.pixels is not None

    @property
    def extension(self) -> str:
        return EXT.get(self.format, ".png" if self.is_image else ".bin")

    def describe(self) -> str:
        what = (f"{self.format} image {self.width}x{self.height}" + (f"x{self.channels}" if self.format == "RAW" else "")
                if self.is_image else f"{len(self.payload)}-byte binary file")
        return (f"{what}, {len(self.payload)} bytes; CRC-32 {'VALID' if self.crc_ok else 'not verified'}; "
                f"FEC {self.fec}; {self.hypothesis}")

    def save(self, path: str) -> str:
        """Encoded formats are written byte-exact; RAW pixels are written as PNG (or .bin if not an image)."""
        if self.format == "RAW" and self.pixels is not None:
            from PIL import Image
            px = self.pixels[:, :, 0] if self.pixels.shape[2] == 1 else self.pixels
            Image.fromarray(px).save(path, format="PNG")
        else:
            with open(path, "wb") as fh:
                fh.write(self.file_bytes)
        return path

    @property
    def file_bytes(self) -> bytes:
        """The encoded image file itself: c.py's compressed mode puts a 5-byte W/H/C=0 header in front
        of the PNG, which must not be written into the saved .png (it made the file unreadable)."""
        for _, magic in MAGIC:
            if self.payload[:len(magic)] == magic:
                return self.payload
            if self.payload[RAW_HEADER_BYTES:RAW_HEADER_BYTES + len(magic)] == magic:
                return self.payload[RAW_HEADER_BYTES:]
        return self.payload


# ---------------------------------------------------------------------------
# Payload identification
# ---------------------------------------------------------------------------

def identify_image(payload: bytes) -> Tuple[str, Optional[np.ndarray]]:
    """(format, pixels) for an encoded or raw-header image; ("BINARY", None) otherwise.
    Raw pixels: 5-byte header W(u16) H(u16) C(u8), C in {1, 3, 4}. C = 0: the header is followed by an
    encoded image (c.py 'Compress payload (PNG)'), whose decoded size must equal W x H."""
    if len(payload) > RAW_HEADER_BYTES + 8 and payload[4] == 0:
        w, h = int.from_bytes(payload[0:2], "big"), int.from_bytes(payload[2:4], "big")
        fmt, px = identify_image(payload[RAW_HEADER_BYTES:])
        if px is not None and px.shape[1] == w and px.shape[0] == h:
            return fmt, px
    for name, magic in MAGIC:
        if payload[:len(magic)] == magic and (name != "WEBP" or payload[8:12] == b"WEBP"):
            try:
                from PIL import Image
                im = Image.open(io.BytesIO(payload))
                im.load()                                   # full decode: truncated / corrupt data fails
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
                return name, np.asarray(im)
            except Exception:                               # noqa: BLE001 - magic without a valid image
                return "BINARY", None
    if len(payload) > RAW_HEADER_BYTES:
        for order in ("big", "little"):
            w = int.from_bytes(payload[0:2], order)
            h = int.from_bytes(payload[2:4], order)
            c = payload[4]
            if (c in (1, 3, 4) and 0 < w <= MAX_RAW_DIM and 0 < h <= MAX_RAW_DIM
                    and RAW_HEADER_BYTES + w * h * c == len(payload)):
                px = np.frombuffer(payload, dtype=np.uint8, offset=RAW_HEADER_BYTES).reshape(h, w, c)
                return "RAW", px
    return "BINARY", None


# ---------------------------------------------------------------------------
# Frame search
# ---------------------------------------------------------------------------

def _find_sync(bits: np.ndarray) -> Tuple[int, int]:
    """(bit index, errors) of the best ASM match, either polarity; errors < 0 means inverted bits."""
    if len(bits) < len(SYNC_BITS):
        return -1, len(SYNC_BITS)
    b = bits.astype(np.int8) * 2 - 1
    ref = (SYNC_BITS.astype(np.int8) * 2 - 1)
    c = np.correlate(b.astype(np.int32), ref.astype(np.int32), "valid")
    i = int(np.argmax(np.abs(c)))
    errors = (len(ref) - abs(int(c[i]))) // 2
    return i, (errors if c[i] > 0 else -errors - 1)       # -1 - errors keeps 'inverted with 0 errors' distinct


def _parse(bits: np.ndarray, start: int) -> Optional[Tuple[bytes, int, int, int]]:
    """(payload, crc received, crc over SYNC+LEN+PAYLOAD, crc over PAYLOAD) of the frame at bit `start`."""
    n_avail = (len(bits) - start) // 8
    if n_avail < HEADER_BYTES + CRC_BYTES:
        return None
    head = np.packbits(bits[start:start + 8 * HEADER_BYTES]).tobytes()
    n = int.from_bytes(head[4:8], "big")
    need = 8 * (HEADER_BYTES + n + CRC_BYTES)
    have = len(bits) - start
    if n == 0 or need - have > CRC_TAIL_BITS:
        return None
    seg = bits[start:start + need]
    if have < need:
        # the stream ends inside the CRC (a burst that ends with its last symbol, or a transmitter
        # that dropped the bits not filling a symbol): pad; the CRC check then tolerates those bits
        seg = np.concatenate([seg, np.zeros(need - have, dtype=seg.dtype)])
    frame = np.packbits(seg).tobytes()
    body = frame[:HEADER_BYTES + n]
    payload = body[HEADER_BYTES:]
    return (payload, int.from_bytes(frame[-CRC_BYTES:], "big"), zlib.crc32(body) & 0xFFFFFFFF,
            zlib.crc32(payload) & 0xFFFFFFFF)


def llr_hypotheses(family: str, r) -> Iterator[Tuple[str, np.ndarray]]:
    """(name, LLRs; positive = bit 0) for each carrier-phase / polarity hypothesis."""
    L0 = np.asarray(r.llrs, dtype=np.float64)
    if family == "FSK" or getattr(r, "symbols", None) is None or not len(r.symbols):
        yield "receiver bits", L0
        yield "receiver bits inverted", -L0
        return
    yield "receiver bits", L0
    const = get_constellation(r.modulation)
    y = np.asarray(r.symbols)
    nv = 10 ** (-max(float(getattr(r, "esn0_est_db", 20.0)), 0.0) / 10)
    m = const.phase_symmetry
    for conj in (False, True):
        for k in range(m):
            yr = rotate(np.conj(y) if conj else y, k, m)
            yield f"rotation +{360 * k // m} deg" + (", conjugated" if conj else ""), const.demap_soft(yr, nv)


def _gray(k):
    k = np.asarray(k, dtype=np.int64)
    return k ^ (k >> 1)


def _gray_inv(g):
    g = np.asarray(g, dtype=np.int64)
    k = g.copy()
    s = g >> 1
    while np.any(s):
        k ^= s
        s >>= 1
    return k


LABELINGS = (("inverse-Gray", _gray_inv), ("Gray", _gray), ("natural", lambda p: np.asarray(p, dtype=np.int64)))


def _bits_of(labels, k):
    labels = np.asarray(labels, dtype=np.int64)
    return ((labels[:, None] >> np.arange(k - 1, -1, -1)[None, :]) & 1).astype(np.uint8).ravel()


def label_hypotheses(family: str, r) -> Iterator[Tuple[str, np.ndarray]]:
    """(name, hard bits) under other transmitters' bit labels (see 3b), every rotation / conjugation."""
    if family == "FSK":
        pos = getattr(r, "positions", None)
        if pos is None or not len(pos):
            return
        pos = np.asarray(pos, dtype=np.int64)
        M = 1 << int(np.ceil(np.log2(max(int(pos.max()) + 1, 2))))
        k = int(np.log2(M))
        for rev in (False, True):
            p = (M - 1 - pos) if rev else pos
            for lname, f in LABELINGS:
                yield f"FSK tones, {lname} labels" + (", reversed" if rev else ""), _bits_of(f(p), k)
        return
    y = np.asarray(getattr(r, "symbols", []))
    if not len(y):
        return
    mod = "QPSK" if r.modulation == "4QAM" else r.modulation
    const = get_constellation(mod)
    M = len(const.points)
    k = const.bits_per_symbol
    if mod.endswith("PSK"):
        if M <= 4:
            return                                          # BPSK / QPSK: every Gray labelling is a rotation
        th0 = float(np.min(np.mod(np.angle(const.points), 2 * np.pi / M)))
        for conj in (False, True):
            yy = np.conj(y) if conj else y
            p0 = np.round((np.angle(yy) - th0) / (2 * np.pi / M)).astype(np.int64)
            for rot in range(M):
                p = np.mod(p0 + rot, M)
                for lname, f in LABELINGS:
                    yield (f"{mod} rotation +{360 * rot // M} deg{', conjugated' if conj else ''}, {lname} labels",
                           _bits_of(f(p), k))
        return
    side = int(round(np.sqrt(M)))
    lv = np.unique(np.round(const.points.real, 6))
    for conj in (False, True):
        for rot in range(4):
            yy = (np.conj(y) if conj else y) * np.exp(1j * np.pi / 2 * rot)
            pi = np.argmin(np.abs(yy.real[:, None] - lv[None, :]), axis=1)
            pq = np.argmin(np.abs(yy.imag[:, None] - lv[None, :]), axis=1)
            for lname, f in LABELINGS:
                bi, bq = f(pi), f(pq)
                for iq in (True, False):
                    lab = (bi * side + bq) if iq else (bq * side + bi)
                    yield (f"{mod} rotation +{90 * rot} deg{', conjugated' if conj else ''}, {lname} labels, "
                           f"{'I' if iq else 'Q'} first", _bits_of(lab, k))


def _accept(bits: np.ndarray, idx: int, err: int, fec: str, name: str) -> Optional[ImagePayload]:
    inverted = err < 0
    errors = -err - 1 if inverted else err
    if idx < 0 or errors > SYNC_MAX_ERRORS:
        return None
    b = (1 - bits) if inverted else bits
    parsed = _parse(b, idx)
    if parsed is None:
        return None
    payload, crc_rx, crc_full, crc_pay = parsed
    fmt, px = identify_image(payload)
    crc_ok = crc_rx in (crc_full, crc_pay)
    crc_c = crc_pay if crc_rx == crc_pay else crc_full
    coverage = "PAYLOAD" if crc_rx == crc_pay else "SYNC+LENGTH+PAYLOAD"
    tail_note = ""
    if not crc_ok:
        # A transmitter that packs the frame into k-bit symbols and drops the remainder (c.py does
        # nb = len(bits) // k * k) never sends the CRC's last (len mod k) < 8 bits. The other >= 25
        # bits must match (chance agreement 2^-25).
        for cov, cc in (("PAYLOAD", crc_pay), ("SYNC+LENGTH+PAYLOAD", crc_full)):
            if (crc_rx ^ cc) >> CRC_TAIL_BITS == 0:
                crc_ok, crc_c, coverage = True, cc, cov
                tail_note = f"; top {32 - CRC_TAIL_BITS}+ bits match, the last bits were not transmitted " \
                            f"(frame length not a whole number of symbols)"
                break
    if not crc_ok and px is None:
        return None                                        # neither checksum nor image structure confirms it
    hyp = name + (", bits inverted" if inverted else "")
    ev = [Evidence(f"CCSDS ASM 0x1ACFFC1D at bit {idx} ({errors}/32 bit errors, {hyp}; FEC {fec}); "
                   f"LENGTH {len(payload)} bytes fits the stream", "IMAGE_FRAME", 1.0 - errors / 32)]
    if crc_ok:
        ev.append(Evidence(f"CRC-32 over {coverage} valid (0x{crc_c:08X}){tail_note}", "CRC", 1.0))
    else:
        ev.append(Evidence(f"CRC-32 mismatch (received 0x{crc_rx:08X}; over SYNC+LENGTH+PAYLOAD 0x{crc_full:08X}, over "
                           f"PAYLOAD 0x{crc_pay:08X}): bit errors in the frame; payload shown on its image structure only",
                           "CRC", 0.3))
    if px is not None:
        how = ("raw pixels: 5-byte header W/H/C predicts the exact payload length" if fmt == "RAW"
               else "file signature present and the image decodes completely")
        ev.append(Evidence(f"Payload is a {fmt} image {px.shape[1]}x{px.shape[0]} ({how})", "IMAGE", 0.9))
    return ImagePayload(fmt, payload, crc_ok, crc_rx, crc_c, fec, hyp, idx, errors,
                        px.shape[1] if px is not None else None, px.shape[0] if px is not None else None,
                        px.shape[2] if px is not None else None, px, ev)


def find_image(family: str, r, progress: Optional[Callable[[str], None]] = None,
               check_cancel: Optional[Callable[[], None]] = None) -> Optional[ImagePayload]:
    """
    Search a receiver result (PSK / QAM / FSK) for a SYNC | LENGTH | PAYLOAD | CRC32 file frame.
    Returns the first CRC-valid frame, else the best structurally confirmed image, else None.
    """
    best: Optional[ImagePayload] = None

    def better(c):
        return c is not None and (best is None or (c.crc_ok, c.is_image, -c.sync_errors) >
                                  (best.crc_ok, best.is_image, -best.sync_errors))

    for name, L in llr_hypotheses(family, r):
        if check_cancel:
            check_cancel()
        if progress:
            progress(name)
        hard = (L < 0).astype(np.uint8)
        c = _accept(hard, *_find_sync(hard), "none", name)
        if better(c):
            best = c
        if best is not None and best.crc_ok:
            return best
        for off in (0, 1):                                 # which coded bit starts a trellis step
            if len(L) - off < 2 * 64:
                continue
            pre = convolutional.viterbi(L[off:off + VITERBI_PREFIX_LLRS], CONV_CODE, terminated=False,
                                        start_known=False)
            idx, err = _find_sync(pre.bits)
            if idx < 0 or (-err - 1 if err < 0 else err) > SYNC_MAX_ERRORS:
                continue
            if check_cancel:
                check_cancel()
            full = pre if len(L) - off <= VITERBI_PREFIX_LLRS else convolutional.viterbi(
                L[off:], CONV_CODE, terminated=False, start_known=False)
            c = _accept(full.bits, *_find_sync(full.bits[:max(idx + 64, 64)]), CONV_CODE.name,
                        f"{name}, Viterbi phase {off}")
            if better(c):
                best = c
            if best is not None and best.crc_ok:
                return best
    # other transmitters' bit labels, hard decisions, uncoded (sync searched in a prefix: it opens the frame)
    for name, hard in label_hypotheses(family, r):
        if check_cancel:
            check_cancel()
        idx, err = _find_sync(hard[:SYNC_PREFIX_BITS])
        if idx < 0 or (-err - 1 if err < 0 else err) > SYNC_MAX_ERRORS:
            continue
        if progress:
            progress(name)
        c = _accept(hard, idx, err, "none", name)
        if better(c):
            best = c
        if best is not None and best.crc_ok:
            return best
    return best
