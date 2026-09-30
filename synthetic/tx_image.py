"""
Image transmitter: file bytes -> frame -> convolutional FEC -> BPSK / QPSK / 16QAM / 4FSK
-> AWGN -> .iq (cs16 + cf32, with the usual JSON sidecar) and a 48 kHz passband .wav.

    bytes -> SYNC | LENGTH | PAYLOAD | CRC32          synthetic/image_framing.py
          -> K=7 (171,133) rate 1/2, zero-tail       fec/convolutional.py  (fec=True)
          -> bits, MSB first                          synthetic/bitops.py
          -> symbols + RRC (beta 0.35, span 10)       synthetic/modulators.py  (BPSK/QPSK/16QAM)
             or continuous-phase 4FSK, h = 0.5       synthetic/modulators.py  (4FSK, no RRC)
          -> AWGN at snr_db = Es/N0 per symbol       (generator.py convention)

The whole frame, SYNC included, is FEC-coded, so the receiver finds SYNC after Viterbi.
"""

import os
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from fec import convolutional
from synthetic.bitops import bytes_to_bits
from synthetic.image_framing import SYNC_WORD, build_frame
from synthetic.iq_io import write_iq
from synthetic.modulators import map_bpsk, map_cpfsk, map_qam, map_qpsk, rrc_shape

MODULATIONS = ("BPSK", "QPSK", "16QAM", "4FSK")
CONV_CODE = convolutional.get_code("CC K=7 (171,133) r1/2 [NASA/CCSDS]")
RRC_BETA, RRC_SPAN = 0.35, 10
FSK_ORDER, FSK_H = 4, 0.5
WAV_RATE, WAV_CARRIER_HZ = 48_000, 6_000.0


def fec_encode(bits: np.ndarray) -> np.ndarray:
    """K=7, rate 1/2, generators (171, 133) octal, zero-tail terminated (fec/convolutional.py)."""
    return convolutional.encode(bits, CONV_CODE, terminate=True)


def modulate(bits: np.ndarray, modulation: str, sps: int) -> np.ndarray:
    """Bits -> complex baseband at sps samples/symbol (RRC-shaped, or CPFSK for 4FSK)."""
    m = modulation.upper()
    if m == "BPSK":
        return rrc_shape(map_bpsk(bits), sps, RRC_BETA, RRC_SPAN)
    if m == "QPSK":
        return rrc_shape(map_qpsk(bits), sps, RRC_BETA, RRC_SPAN)
    if m == "16QAM":
        return rrc_shape(map_qam(bits, 16), sps, RRC_BETA, RRC_SPAN)
    if m == "4FSK":
        return map_cpfsk(bits, FSK_ORDER, FSK_H, sps)
    raise ValueError(f"modulation must be one of {MODULATIONS}, not {modulation!r}")


def bits_per_symbol(modulation: str) -> int:
    """Bits carried per symbol for the supported modulations."""
    return {"BPSK": 1, "QPSK": 2, "16QAM": 4, "4FSK": 2}[modulation.upper()]


def add_awgn(x: np.ndarray, modulation: str, sps: int, snr_db: float, seed: int) -> np.ndarray:
    """
    AWGN at Es/N0 = snr_db. RRC signals: unit-energy pulse, unit-power symbols -> Es = 1, so the
    per-sample noise variance is N0. 4FSK: unit amplitude over sps samples -> Es = sps.
    """
    es = float(sps) if modulation.upper() == "4FSK" else 1.0
    n0 = es * 10 ** (-snr_db / 10.0)
    rng = np.random.default_rng(seed)
    return x + np.sqrt(n0 / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))


def write_wav(path: str, x: np.ndarray, fs: float) -> None:
    """Resample the baseband to 48 kHz, shift to a 6 kHz real carrier, write 16-bit mono PCM."""
    ratio = Fraction(WAV_RATE, int(round(fs))).limit_denominator(1000)
    y = resample_poly(x, ratio.numerator, ratio.denominator)
    t = np.arange(len(y)) / WAV_RATE
    pb = np.real(y * np.exp(2j * np.pi * WAV_CARRIER_HZ * t))
    pcm = np.round(pb / (np.max(np.abs(pb)) + 1e-12) * 0.9 * 32767).astype(np.int16)
    wavfile.write(path, WAV_RATE, pcm)


def transmit_image(image_path: str,
                   modulation: str,
                   output_dir: str,
                   fec: bool = True,
                   sps: int = 8,
                   baud: int = 1200,
                   snr_db: float = 25.0,
                   seed: int = 26147) -> Dict[str, Any]:
    """
    Frame, FEC-encode, modulate and write image_path as {tag}.iq, {tag}_f32.iq and {tag}.wav
    in output_dir, tag = img_<image stem>_<modulation>. Returns the transmission record.
    """
    modulation = modulation.upper()
    if modulation not in MODULATIONS:
        raise ValueError(f"modulation must be one of {MODULATIONS}, not {modulation!r}")
    with open(image_path, "rb") as fh:
        payload = fh.read()
    frame = build_frame(payload)
    bits = bytes_to_bits(frame)
    if fec:
        bits = fec_encode(bits)
    x = add_awgn(modulate(bits, modulation, sps), modulation, sps, snr_db, seed)

    fs = float(baud * sps)
    os.makedirs(output_dir, exist_ok=True)
    tag = f"img_{Path(image_path).stem}_{modulation.lower()}"
    stem = os.path.join(output_dir, tag)
    truth = {"generator": "synthetic.tx_image v1", "modulation": modulation, "fec": CONV_CODE.name if fec else None,
             "payload_file": os.path.basename(image_path), "payload_bytes": len(payload),
             "frame_bytes": len(frame), "transmitted_bits": int(len(bits)), "symbol_rate_hz": float(baud),
             "samples_per_symbol": sps, "snr_db_esn0": snr_db, "seed": seed}
    write_iq(stem, x, fs, 0.0, truth)
    write_wav(stem + ".wav", x, fs)
    bps = bits_per_symbol(modulation)
    return {"image_path": image_path, "modulation": modulation, "fec": bool(fec), "sps": sps, "baud": baud,
            "snr_db": snr_db, "payload_bytes": len(payload), "frame_bytes": len(frame),
            "coded_bits": int(len(bits)), "symbol_count": int(-(-len(bits) // bps)),
            "iq_cs16_path": stem + ".iq", "iq_cf32_path": stem + "_f32.iq", "wav_path": stem + ".wav",
            "sync_word_hex": SYNC_WORD.hex().upper(),
            "crc32_hex": f"{zlib.crc32(frame[:-4]) & 0xFFFFFFFF:08X}"}
