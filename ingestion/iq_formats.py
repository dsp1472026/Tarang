"""
Raw interleaved IQ sample formats (SDR naming) and blind format detection.

    key    aliases                         bytes / complex sample   scaling to [-1, 1)
    cu8    CU8, uint8, u8 (RTL-SDR)        2   (I u8, Q u8)          (v - 127.5) / 128
    cs8    CS8, CI8, SC8, int8, i8         2   (I s8, Q s8)          v / 128
    cs16   CS16, CI16, SC16, int16, i16    4   (I s16, Q s16)        v / 32768
    cf32   CF32, FC32, float32, complex64  8   (I f32, Q f32)        as stored
    cf64   CF64, FC64, float64, complex128 16  (I f64, Q f64)        as stored

Multi-byte formats take an '_le' / '_be' suffix (default little-endian), so the
legacy sidecar names 'ci16_le' and 'cf32_le' still resolve.

Blind detection (detect_iq_datatype) looks at byte structure rather than value
ranges, so it does not depend on the recording's level:
  1. float:  cf32 / cf64 (either endianness) decoded values must be finite, not
     denormal and within a sane dynamic range; any other byte layout read as a
     float puts sample bits into the exponent and spreads the magnitudes over
     tens of decades.
  2. 8 vs 16 bit:  in int16 data the low byte of each value is ~uniformly random
     (or, at low level, the high byte is the sign extension of the low byte);
     in 8-bit data the even and odd byte lanes are both slowly varying samples.
     Byte-to-byte steps are measured modulo 256, so this is offset-independent.
  3. cu8 vs cs8:  reading with the wrong offset makes every zero crossing a
     ~255-step jump; the interpretation with the smaller mean step wins.
"""

import re
from typing import Dict, Optional, Tuple

import numpy as np

# key -> (numpy scalar type, bytes per complex sample, offset, scale)
FORMATS: Dict[str, Tuple[str, int, float, float]] = {
    "cu8": ("u1", 2, 127.5, 128.0),
    "cs8": ("i1", 2, 0.0, 128.0),
    "cs16": ("i2", 4, 0.0, 32768.0),
    "cf32": ("f4", 8, 0.0, 1.0),
    "cf64": ("f8", 16, 0.0, 1.0),
}

# GUI labels, in display order (first token of each resolves via normalize_datatype)
DISPLAY_NAMES = {
    "cu8": "CU8 (uint8)",
    "cs8": "CS8 / CI8 (int8)",
    "cs16": "CS16 / CI16 / SC16 (int16)",
    "cf32": "CF32 / FC32 (float32)",
    "cf64": "CF64 / FC64 (float64)",
}

# Names used by ingestion.format_detector / IQReader (and their tests)
LEGACY_NAMES = {"cu8": "uint8", "cs8": "int8", "cs16": "int16", "cf32": "float32", "cf64": "complex128"}

_ALIASES = {
    "cu8": "cu8", "uint8": "cu8", "u8": "cu8", "rtl": "cu8",
    "cs8": "cs8", "ci8": "cs8", "sc8": "cs8", "int8": "cs8", "i8": "cs8", "s8": "cs8",
    "cs16": "cs16", "ci16": "cs16", "sc16": "cs16", "int16": "cs16", "i16": "cs16", "s16": "cs16",
    "cf32": "cf32", "fc32": "cf32", "float32": "cf32", "complex64": "cf32", "f32": "cf32",
    "cf64": "cf64", "fc64": "cf64", "float64": "cf64", "complex128": "cf64", "f64": "cf64",
}


def normalize_datatype(name: Optional[str]) -> Optional[Tuple[str, str]]:
    """'CS16 / CI16 (int16)', 'ci16_le', 'int16', 'cf32_be', ... -> ('cs16', 'little').
    None / 'auto' / '' -> None."""
    if name is None:
        return None
    tok = re.split(r"[\s/(]+", str(name).strip().lower())[0]
    if tok in ("", "auto"):
        return None
    endian = "little"
    m = re.fullmatch(r"(.+?)[_-]?(le|be)", tok)
    if m and m.group(1) in _ALIASES:
        tok, endian = m.group(1), "big" if m.group(2) == "be" else "little"
    if tok not in _ALIASES:
        raise ValueError(f"unsupported IQ datatype '{name}'. Supported: CU8, CS8/CI8, CS16/CI16/SC16, "
                         f"CF32/FC32, CF64/FC64 (optionally _le/_be)")
    return _ALIASES[tok], endian


def datatype_tag(key: str, endian: str = "little") -> str:
    """Canonical tag as written in sidecars: 'cu8', 'cs8', 'cs16_le', 'cf32_be', ..."""
    return key if FORMATS[key][0][1] == "1" else f"{key}_{'le' if endian == 'little' else 'be'}"


def bytes_per_complex_sample(name: str) -> int:
    return FORMATS[normalize_datatype(name)[0]][1]


def decode_iq_bytes(raw: bytes, key: str, endian: str = "little") -> np.ndarray:
    """Interleaved I,Q bytes -> complex64 in [-1, 1) (integers) or as stored (floats).
    A trailing partial complex sample is dropped."""
    code, bps, offset, scale = FORMATS[key]
    dt = np.dtype(("<" if endian == "little" else ">") + code)
    n = len(raw) // bps * bps
    v = np.frombuffer(raw[:n], dtype=dt)
    if key in ("cf32", "cf64"):
        v = v.astype(np.float64)
    else:
        v = (v.astype(np.float64) - offset) / scale
    return (v[0::2] + 1j * v[1::2]).astype(np.complex64)


def encode_iq(x: np.ndarray, key: str, endian: str = "little", peak: float = 0.95) -> bytes:
    """complex -> interleaved bytes. Integer formats are peak-scaled to `peak` of full scale."""
    code, _, offset, scale = FORMATS[key]
    x = np.asarray(x, dtype=np.complex128)
    v = np.empty(2 * len(x))
    v[0::2], v[1::2] = x.real, x.imag
    if key not in ("cf32", "cf64"):
        v = v * (peak * (scale - 1) / (np.max(np.abs(v)) + 1e-300))
        lo, hi = (0, 255) if key == "cu8" else (-scale, scale - 1)
        v = np.clip(np.round(v + offset), lo, hi)
    return v.astype(("<" if endian == "little" else ">") + code).tobytes()


# ---------------------------------------------------------------------------
# Blind detection
# ---------------------------------------------------------------------------

def _drop_zero_blocks(b: np.ndarray, block: int = 16) -> np.ndarray:
    """Remove all-zero 16-byte-aligned blocks (silence / padding) keeping lane alignment."""
    n = len(b) // block * block
    blocks = b[:n].reshape(-1, block)
    return blocks[np.any(blocks != 0, axis=1)].ravel()


def _float_plausibility(b: np.ndarray, code: str) -> float:
    """Fraction-style score in [0, 1] that bytes b are IEEE floats of this type/endianness."""
    itemsize = int(code[-1])
    n = len(b) // (2 * itemsize) * (2 * itemsize)
    if n < 64 * itemsize:
        return 0.0
    with np.errstate(all="ignore"):
        v = np.frombuffer(b[:n].tobytes(), dtype=np.dtype(code)).astype(np.float64)
        if not np.all(np.isfinite(v)):
            return 0.0
        a = np.abs(v[v != 0])
        if len(a) < 32:
            return 0.0
        tiny = np.finfo(np.float32 if itemsize == 4 else np.float64).tiny
        if np.mean(a < tiny) > 0.001:                      # denormals: integer bits in the exponent
            return 0.0
        med, hi = np.percentile(a, [50, 99.9])
        if not (1e-15 < med < 1e15):
            return 0.0
        span = np.log10(hi / med)                        # decades from the median to the 99.9th percentile
    # Real IQ peaks sit within ~1-3 decades of the median (the lower tail is not used: exact
    # rotations leave ~1e-17 residues); a wrong layout spreads exponents over tens of decades.
    return float(np.clip((6.0 - span) / 2.0, 0.0, 1.0))


def _mean_step_mod256(lane: np.ndarray) -> float:
    d = (lane[1:].astype(np.int16) - lane[:-1].astype(np.int16) + 128) % 256 - 128
    return float(np.mean(np.abs(d))) if len(d) else 0.0


def _tv_from_uniform(lane: np.ndarray) -> float:
    """Total-variation distance between a byte lane's histogram and uniform over 256 values."""
    if len(lane) == 0:
        return 1.0
    h = np.bincount(lane, minlength=256) / len(lane)
    return float(0.5 * np.sum(np.abs(h - 1 / 256)))


def _roughness(v: np.ndarray) -> float:
    """Mean |step| / std of the I and Q streams (interleaved v): ~1.13 for white noise, << 1 when
    oversampled. Constant streams are skipped."""
    v = np.asarray(v, dtype=np.float64)
    out = []
    for comp in (v[0::2], v[1::2]):
        sd = np.std(comp)
        if sd > 0 and len(comp) > 2:
            out.append(np.mean(np.abs(np.diff(comp))) / sd)
    return float(np.mean(out)) if out else 1.13


def detect_iq_datatype(raw: bytes) -> dict:
    """
    Blind format decision from raw sample bytes (header already skipped).
    Returns {'key', 'endian', 'confidence' (0-100), 'scores', 'reason'}.
    """
    b = _drop_zero_blocks(np.frombuffer(raw, dtype=np.uint8))
    if len(b) < 256:
        return {"key": "cs16", "endian": "little", "confidence": 30.0, "scores": {},
                "reason": "too few non-zero bytes to decide; defaulting to CS16"}

    # 1. floats
    fscores = {(k, e): _float_plausibility(b, ("<" if e == "little" else ">") + FORMATS[k][0])
               for k in ("cf32", "cf64") for e in ("little", "big")}
    (fk, fe), fbest = max(fscores.items(), key=lambda kv: kv[1])
    scores = {datatype_tag(k, e): round(s, 3) for (k, e), s in fscores.items()}
    if fbest >= 0.5:
        return {"key": fk, "endian": fe, "confidence": round(60 + 40 * fbest, 1), "scores": scores,
                "reason": f"{fk.upper()} {fe}-endian: all values finite, normal, within a plausible dynamic range"}

    # 2. 8-bit vs 16-bit (per-byte-lane randomness; 4-byte groups = one CS16 sample = two 8-bit samples)
    n4 = len(b) // 4 * 4
    lanes = [b[j:n4:4] for j in range(4)]
    step = [_mean_step_mod256(l) for l in lanes]             # uniform random bytes -> ~64
    # skip a constant component (e.g. Q == 0 for noiseless BPSK) so it does not dilute the lanes
    comps = [c for c in (0, 2) if np.ptp(lanes[c]) or np.ptp(lanes[c + 1])] or [0, 2]
    lo_le = float(np.mean([step[c] for c in comps]))
    hi_le = float(np.mean([step[c + 1] for c in comps]))
    # low-level int16: high byte = sign extension of the low byte
    pairs = b[:n4].reshape(-1, 2)
    signext_le = float(np.mean(pairs[:, 1] == np.where(pairs[:, 0] >= 128, 255, 0)))
    signext_be = float(np.mean(pairs[:, 0] == np.where(pairs[:, 1] >= 128, 255, 0)))
    s16 = {"little": max(signext_le, 0.0), "big": max(signext_be, 0.0)}
    for e, lo, hi in (("little", lo_le, hi_le), ("big", hi_le, lo_le)):
        if lo > 40 and lo > 1.6 * hi:
            s16[e] = max(s16[e], min(1.0, (lo / max(hi, 1e-9) - 1.6) / 2 + 0.6))
    # Smoothness: the right word size gives the smoother sample stream (8-bit data read as int16
    # sees every other 8-bit sample of one component; int16 read as 8-bit sees random low bytes).
    rough8 = min(_roughness(b[:n4].view(np.int8)), _roughness(b[:n4].astype(np.float64) - 127.5))
    for e in ("little", "big"):
        r16 = _roughness(b[:n4].view("<i2" if e == "little" else ">i2"))
        if r16 < 0.8 * rough8:
            s16[e] = max(s16[e], float(np.clip(0.6 + (0.8 - r16 / rough8), 0.6, 1.0)))
    # Marginals: the low byte of int16 samples is ~uniform over 0..255; an 8-bit sample lane
    # follows the signal's amplitude distribution (Gaussian, arcsine, lattice), so both lanes of
    # 8-bit data sit equally far from uniform.
    tv = [_tv_from_uniform(lanes[j]) for j in range(4)]
    tv_noise = 0.5 * np.sqrt(2 * 256 / (np.pi * max(len(lanes[0]), 1)))
    for e, lo_tv, hi_tv in (("little", min(tv[0], tv[2]), min(tv[1], tv[3])),
                            ("big", min(tv[1], tv[3]), min(tv[0], tv[2]))):
        if lo_tv < 1.5 * tv_noise + 0.02 and hi_tv > lo_tv + 0.12:
            s16[e] = max(s16[e], float(np.clip(0.6 + (hi_tv - lo_tv - 0.12), 0.6, 0.95)))
    e16 = max(s16, key=s16.get)
    scores.update({datatype_tag("cs16", e): round(s, 3) for e, s in s16.items()})
    if s16[e16] >= 0.6:
        return {"key": "cs16", "endian": e16, "confidence": round(60 + 40 * s16[e16], 1), "scores": scores,
                "reason": (f"CS16 {e16}-endian: " + ("high byte is the sign extension of the low byte"
                           if max(signext_le, signext_be) >= 0.6 else
                           f"low bytes look random (mean step {max(lo_le, hi_le):.0f}/256) while "
                           f"high bytes vary slowly ({min(lo_le, hi_le):.0f})"))}

    # 3. cu8 vs cs8: wrong offset -> ~255 jumps at zero crossings
    n2 = len(b) // 2 * 2
    iq = b[:n2].reshape(-1, 2).astype(np.float64)
    step_u = float(np.mean(np.abs(np.diff(iq, axis=0))))
    step_s = float(np.mean(np.abs(np.diff(iq.astype(np.uint8).view(np.int8).astype(np.float64), axis=0))))
    key = "cu8" if step_u < step_s else "cs8"
    ratio = max(step_u, step_s) / max(min(step_u, step_s), 1e-9)
    scores.update({"cu8": round(1 / (1 + step_u), 4), "cs8": round(1 / (1 + step_s), 4)})
    conf = float(np.clip(50 + 50 * (ratio - 1), 50, 99))
    return {"key": key, "endian": "little", "confidence": round(conf, 1), "scores": scores,
            "reason": f"8-bit ({key.upper()}): mean sample step {min(step_u, step_s):.1f} vs "
                      f"{max(step_u, step_s):.1f} with the other offset"}
