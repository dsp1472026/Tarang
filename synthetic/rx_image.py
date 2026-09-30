"""
Image receiver: .iq -> the project's existing receivers -> soft LLRs -> Viterbi -> SYNC search
-> LENGTH / PAYLOAD / CRC32 -> image file.

    BPSK / QPSK   demodulation/psk_receiver.receive_psk   (modulation and sps given)
    16QAM         demodulation/qam_receiver.receive_qam   (modulation and sps given)
    4FSK          demodulation/fsk_receiver.receive_fsk   (blind; one rate hypothesis)

Carrier ambiguity. The receivers resolve the PSK/QAM phase ambiguity with the project's PN127
sync word, which an image frame does not carry, so their symbols come out in an arbitrary one of
the M-fold rotations (QPSK/16QAM needed 90 degrees on the test vectors). Every rotation, with and
without conjugation (spectral inversion), is demapped and decoded; for 4FSK the receiver's bits
and their inversion. SYNC decides among the hypotheses and CRC32 confirms: the first hypothesis
whose SYNC peak is strong enough and whose CRC validates is taken, else the strongest SYNC peak
is reported with crc_ok = False.

SYNC search: sliding +-1 correlation with the 32-bit ASM over every bit position of the decoded
stream (so every one of the 8 byte alignments), both polarities (the sign of the peak).

The FSK receiver is limited to its best symbol-rate hypothesis: with FEC-coded data it never
finds its own CRC frame and would otherwise backtrack through 10 rate candidates (147 s instead
of 4.5 s for a 700-byte frame).
"""

import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from demodulation.fsk_receiver import FSKReceiverConfig, receive_fsk
from demodulation.psk import rotate
from demodulation.psk_receiver import PSKReceiverConfig, receive_psk
from demodulation.qam_receiver import QAMReceiverConfig, receive_qam
from dsp.constellations import get_constellation
from fec import convolutional
from synthetic.bitops import bits_to_bytes, bytes_to_bits
from synthetic.image_framing import HEADER_BYTES, SYNC_WORD, build_frame, parse_frame
from synthetic.iq_io import read_iq

CONV_CODE = convolutional.get_code("CC K=7 (171,133) r1/2 [NASA/CCSDS]")
SYNC_MIN_CORRELATION = 28         # of 32: at most 2 SYNC bit errors
MAGIC = (("JPEG", bytes.fromhex("FFD8FF")),
         ("PNG", bytes.fromhex("89504E470D0A1A0A")),
         ("GIF", b"GIF8"),
         ("BMP", b"BM"))


def detect_format(data: bytes) -> str:
    """Image format from magic bytes: JPEG, PNG, GIF, BMP, else UNKNOWN."""
    for name, magic in MAGIC:
        if bytes(data[:len(magic)]) == magic:
            return name
    return "UNKNOWN"


def find_sync(bits: np.ndarray) -> Tuple[int, int]:
    """(bit index, signed correlation) of the strongest ASM match; sign < 0 means inverted bits."""
    ref = bytes_to_bits(SYNC_WORD).astype(np.int32) * 2 - 1
    b = np.asarray(bits, dtype=np.int32) * 2 - 1
    if len(b) < len(ref):
        return -1, 0
    c = np.correlate(b, ref, "valid")
    i = int(np.argmax(np.abs(c)))
    return i, int(c[i])


def load_iq(iq_path: str, fallback_rate: float) -> Tuple[np.ndarray, float, str]:
    """(samples, sample rate, datatype): cs16 vs cf32 from the sidecar, else the _f32 filename rule."""
    f = read_iq(iq_path)
    return np.asarray(f.samples, dtype=np.complex128), float(f.sample_rate or fallback_rate), f.datatype


def demodulate(x: np.ndarray, fs: float, modulation: str, sps: int):
    """Run the project's receiver for this modulation; returns its result object."""
    if modulation in ("BPSK", "QPSK"):
        return receive_psk(x, fs, PSKReceiverConfig(modulation=modulation, sps=float(sps)))
    if modulation == "16QAM":
        return receive_qam(x, fs, QAMReceiverConfig(modulation=modulation, sps=float(sps)))
    if modulation == "4FSK":
        return receive_fsk(x, fs, FSKReceiverConfig(max_rate_candidates=1))
    raise ValueError(f"unsupported modulation {modulation!r}")


def llr_hypotheses(r, modulation: str) -> Iterator[Tuple[str, np.ndarray]]:
    """(name, LLRs) for every carrier-ambiguity hypothesis; the receiver's own output first."""
    if modulation == "4FSK":
        L = np.asarray(r.llrs, dtype=np.float64)
        yield "receiver bits", L
        yield "inverted", -L
        return
    const = get_constellation(modulation)
    y = np.asarray(r.symbols)
    nv = 10 ** (-max(float(r.esn0_est_db), 0.0) / 10)
    m = const.phase_symmetry
    for conj in (False, True):
        for k in range(m):
            yr = rotate(np.conj(y) if conj else y, k, m)
            yield f"rotation +{360 * k // m} deg" + (", conjugated" if conj else ""), const.demap_soft(yr, nv)


def channel_bit_errors(L: np.ndarray, off: int, sync_index: int, inverted: bool, payload: bytes,
                       fec: bool) -> int:
    """
    Channel bit errors over the frame span once the frame is known (CRC valid): the frame is
    rebuilt (and re-encoded with FEC) and compared with the receiver's hard decisions. (171, 133)
    both have odd weight, so inverted information bits mean inverted coded bits as well.
    """
    ref = bytes_to_bits(build_frame(payload))
    start = sync_index
    if fec:
        ref = convolutional.encode(ref, CONV_CODE, terminate=False)
        start = off + 2 * sync_index
    hard = (np.asarray(L[start:start + len(ref)]) < 0).astype(np.uint8)
    ref = ref[:len(hard)]
    return int(np.count_nonzero(hard != ((1 - ref) if inverted else ref)))


def _try(bits: np.ndarray) -> Dict[str, Any]:
    """SYNC search + frame parse on one decoded bit stream."""
    i, c = find_sync(bits)
    out = {"sync_index": i, "sync_corr": c, "crc_ok": False, "payload": b""}
    if abs(c) >= SYNC_MIN_CORRELATION:
        b = bits[i:] if c > 0 else 1 - bits[i:]
        out["payload"], out["crc_ok"] = parse_frame(bits_to_bytes(b))
    return out


def receive_image(iq_path: str,
                  modulation: str,
                  output_image_path: str,
                  fec: bool = True,
                  sps: int = 8,
                  baud: int = 1200) -> Dict[str, Any]:
    """
    Demodulate iq_path, recover the framed payload and write it to output_image_path (only when
    CRC32 validates). Returns the reception record; crc_ok = False means nothing was written.
    """
    t0 = time.time()
    modulation = modulation.upper()
    x, fs, dtype = load_iq(iq_path, float(baud * sps))
    r = demodulate(x, fs, modulation, sps)

    best: Optional[Dict[str, Any]] = None
    tried: List[str] = []
    fec_decoded = False
    n_llr = 0
    for name, L in llr_hypotheses(r, modulation):
        n_llr = len(L)
        streams = []
        if fec:
            for off in (0, 1):                  # which received bit starts a trellis step
                v = convolutional.viterbi(L[off:], CONV_CODE, terminated=False, start_known=False)
                streams.append((f"{name}, Viterbi offset {off}", v.bits, True, off))
        else:
            streams.append((name, (L < 0).astype(np.uint8), False, 0))
        for label, bits, decoded, off in streams:
            res = {**_try(bits), "hypothesis": label, "fec_decoded": decoded, "bits_after_fec": int(len(bits))}
            if res["crc_ok"]:
                res["bit_errors"] = channel_bit_errors(L, off, res["sync_index"], res["sync_corr"] < 0,
                                                       res["payload"], decoded)
            tried.append(label)
            if best is None or (res["crc_ok"], abs(res["sync_corr"])) > (best["crc_ok"], abs(best["sync_corr"])):
                best = res
            if res["crc_ok"]:
                break
        if best["crc_ok"]:
            fec_decoded = best["fec_decoded"]
            break

    if fec and not best["crc_ok"]:              # Viterbi found nothing: fall back to the raw hard bits
        for name, L in llr_hypotheses(r, modulation):
            res = {**_try((L < 0).astype(np.uint8)), "hypothesis": name + ", hard bits (no FEC)",
                   "fec_decoded": False, "bits_after_fec": int(len(L))}
            if res["crc_ok"]:
                res["bit_errors"] = channel_bit_errors(L, 0, res["sync_index"], res["sync_corr"] < 0,
                                                       res["payload"], False)
            tried.append(res["hypothesis"])
            if (res["crc_ok"], abs(res["sync_corr"])) > (best["crc_ok"], abs(best["sync_corr"])):
                best = res
            if res["crc_ok"]:
                break

    payload = best["payload"] if best["crc_ok"] else b""
    if best["crc_ok"]:
        os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
        with open(output_image_path, "wb") as fh:
            fh.write(payload)
    return {"output_image_path": output_image_path if best["crc_ok"] else None,
            "crc_ok": bool(best["crc_ok"]),
            "fec_decoded": bool(fec_decoded and best["crc_ok"]),
            "sync_found": abs(best["sync_corr"]) >= SYNC_MIN_CORRELATION,
            "sync_offset_bits": int(best["sync_index"]),
            "sync_byte_alignment": int(best["sync_index"] % 8) if best["sync_index"] >= 0 else None,
            "sync_correlation": int(best["sync_corr"]),
            "polarity": "inverted" if best["sync_corr"] < 0 else "normal",
            "ambiguity_hypothesis": best["hypothesis"],
            "hypotheses_tried": len(tried),
            "detected_format": detect_format(payload) if best["crc_ok"] else "UNKNOWN",
            "payload_bytes": len(payload),
            "frame_header_bytes": HEADER_BYTES,
            "bits_before_fec": int(n_llr),
            "bits_after_fec": int(best["bits_after_fec"]),
            "bit_error_count": best.get("bit_errors"),     # channel errors over the frame (CRC-valid only)
            "iq_datatype": dtype, "sample_rate_hz": fs,
            "elapsed_seconds": round(time.time() - t0, 3)}
