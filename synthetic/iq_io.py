"""
IQ file I/O with a JSON sidecar.

On-disk format (same as the existing Test-Data files):
  <stem>.iq      interleaved I/Q, int16 little-endian, peak-scaled to 0.95 full scale
  <stem>_f32.iq  interleaved I/Q, float32 little-endian, unscaled
  <stem>.json    sidecar: {"format": {...}, "truth": {...}}

Demodulators may only consume the "format" block (datatype, sample rate,
centre frequency). The "truth" block is for validation only.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


def write_iq(stem: str, samples: np.ndarray, sample_rate: Optional[float],
             center_frequency: Optional[float], truth: dict) -> dict:
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    samples = np.asarray(samples, dtype=np.complex128)

    scale = 0.95 * 32767.0 / (np.max(np.abs(np.r_[samples.real, samples.imag])) + 1e-12)
    i16 = np.empty(2 * len(samples), dtype="<i2")
    i16[0::2] = np.round(samples.real * scale)
    i16[1::2] = np.round(samples.imag * scale)
    i16.tofile(stem + ".iq")

    f32 = np.empty(2 * len(samples), dtype="<f4")
    f32[0::2] = samples.real
    f32[1::2] = samples.imag
    f32.tofile(stem + "_f32.iq")

    base = os.path.basename(stem)
    meta = {
        "format": {
            "files": {base + ".iq": "ci16_le", base + "_f32.iq": "cf32_le"},
            "interleaving": "IQIQ",
            "sample_rate_hz": sample_rate,
            "center_frequency_hz": center_frequency,
            "num_samples": int(len(samples)),
        },
        "truth": truth,
    }
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


@dataclass
class IQFile:
    samples: np.ndarray              # complex64
    sample_rate: Optional[float]     # None -> unknown (never invented)
    center_frequency: Optional[float]
    datatype: str
    sample_rate_source: str          # "KNOWN" (metadata/user) or "UNKNOWN"


def _sidecar_candidates(path: str):
    stem = path[:-3] if path.endswith(".iq") else os.path.splitext(path)[0]
    if stem.endswith("_f32"):
        stem = stem[:-4]
    return stem, [stem + ".json", stem + "_ground_truth.json", stem + "_metadata.json"]


def read_iq(path: str, datatype: Optional[str] = None, sample_rate: Optional[float] = None,
            byte_offset: int = 0, endianness: Optional[str] = None) -> IQFile:
    """
    Load an interleaved IQ file.

    datatype: any name ingestion.iq_formats understands - CU8, CS8/CI8, CS16/CI16/SC16,
    CF32/FC32, CF64/FC64 (optionally '_le'/'_be'; legacy 'ci16_le', 'cf32_le', 'int16',
    'complex64', ...). None -> sidecar, else the '_f32' filename convention, else CS16.
    endianness: 'little' | 'big' overrides the suffix. byte_offset: header bytes to skip.
    Sample rate comes from the caller, else the sidecar's format block, else legacy
    Test-Data metadata; otherwise it stays unknown.
    """
    from ingestion.iq_formats import datatype_tag, decode_iq_bytes, normalize_datatype

    _, sidecars = _sidecar_candidates(path)
    fmt, legacy_fs = {}, None
    for sc in sidecars:
        if os.path.exists(sc):
            with open(sc) as f:
                meta = json.load(f)
            if "format" in meta:
                fmt = meta["format"]
            elif "config" in meta:   # legacy *_ground_truth.json: only the timebase is read
                legacy_fs = meta["config"].get("baseband_fs_hz")
            elif "sample_rate" in meta:
                legacy_fs = meta["sample_rate"]
            elif "baseband_sample_rate" in meta:     # all_modulations_output/*_metadata.json
                legacy_fs = meta["baseband_sample_rate"]
            break

    if datatype is None:
        datatype = fmt.get("files", {}).get(os.path.basename(path)) or fmt.get("datatype")
    if datatype is None:
        datatype = "cf32_le" if path.endswith("_f32.iq") else "cs16_le"
    key, endian = normalize_datatype(datatype)
    if endianness in ("little", "big"):
        endian = endianness

    with open(path, "rb") as fh:
        fh.seek(int(byte_offset or 0))
        x = decode_iq_bytes(fh.read(), key, endian)

    fs = sample_rate or fmt.get("sample_rate_hz") or legacy_fs
    return IQFile(x, float(fs) if fs else None, fmt.get("center_frequency_hz"), datatype_tag(key, endian),
                  "KNOWN" if fs else "UNKNOWN")


def read_truth(path: str) -> dict:
    """Validation only: return the truth block (new format) or legacy ground truth."""
    _, sidecars = _sidecar_candidates(path)
    for sc in sidecars:
        if os.path.exists(sc):
            with open(sc) as f:
                meta = json.load(f)
            if "truth" in meta:
                return meta["truth"]
            if "scheme" in meta:
                return {"legacy": True, "modulation": meta["scheme"], "payload_text": meta["message"],
                        "bits": meta["bits"], "config": meta["config"]}
            if "input_text" in meta:                  # all_modulations_output/*_metadata.json
                return {"legacy": True, "modulation": meta["modulation"], "payload_text": meta["input_text"],
                        "config": {"sps": meta.get("samples_per_symbol"), "baud": meta.get("baud_rate")}}
    return {}
