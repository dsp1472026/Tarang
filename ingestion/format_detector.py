import os
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np

from core.signal_buffer import SignalBuffer
from ingestion.iq_reader import IQReader
from ingestion.wav_reader import WAVReader
from ingestion.sigmf_reader import SigMFReader
from ingestion.iq_formats import FORMATS, LEGACY_NAMES, detect_iq_datatype, normalize_datatype

logger = logging.getLogger("ingestion.format_detector")


@dataclass
class IQDetectionResult:
    format_name: str          # 'uint8' (CU8), 'int8' (CS8), 'int16' (CS16), 'float32' (CF32), 'complex128' (CF64)
    confidence: float         # Score from 0.0 to 100.0 (%)
    bytes_per_sample: int     # 2, 4, 8, 16
    method: str               # 'Metadata', 'SigMF', 'FilenameHint', 'StatisticalAnalysis'
    scores: Dict[str, float]  # Candidate -> score
    is_low_confidence: bool   # True if confidence < 80.0
    warnings: List[str]
    endianness: str = "little"
    header_offset: int = 0    # bytes to skip before the first sample


class FormatDetector:
    """
    Auto-detects file format by header magic bytes, extension, sidecar metadata,
    and memory-mapped multi-region 5-step statistical IQ sample format detection with confidence scoring.
    Handles files of any size (up to multi-GB) within < 5s and < 500 MB peak RAM.
    """

    @staticmethod
    def detect_format(file_path: str) -> str:
        """
        Returns 'WAV', 'SigMF', or 'IQ' based on header magic bytes and extension.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        # Check RIFF/WAVE header
        try:
            with open(file_path, "rb") as f:
                header = f.read(12)
                if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                    return "WAV"
        except Exception:
            pass

        if ext in (".sigmf-meta", ".sigmf-data"):
            return "SigMF"
        elif ext == ".wav":
            return "WAV"
        else:
            return "IQ"

    @staticmethod
    def read_chunk(file_path: str, offset_bytes: int, chunk_bytes: int, np_dtype: Any) -> np.ndarray:
        """
        Memory-mapped chunked reading helper. Never loads entire file into memory.
        """
        if not os.path.exists(file_path):
            return np.array([], dtype=np_dtype)

        file_size = os.path.getsize(file_path)
        if offset_bytes >= file_size or chunk_bytes <= 0:
            return np.array([], dtype=np_dtype)

        itemsize = np.dtype(np_dtype).itemsize
        available_bytes = file_size - offset_bytes
        read_bytes = min(chunk_bytes, available_bytes)
        num_items = read_bytes // itemsize

        if num_items <= 0:
            return np.array([], dtype=np_dtype)

        try:
            mm = np.memmap(file_path, dtype=np_dtype, mode="r", offset=offset_bytes, shape=(num_items,))
            data = np.array(mm)  # Copy requested chunk slice to RAM
            del mm               # Explicitly release memmap file handle
            return data
        except Exception as e:
            logger.warning(f"memmap read failed at offset {offset_bytes}: {e}")
            return np.array([], dtype=np_dtype)

    @classmethod
    def detect_header_offset(cls, file_path: str) -> Tuple[int, Optional[str]]:
        """
        Detects ASCII / metadata headers at file start and returns (header_bytes_to_skip, header_description).
        """
        try:
            with open(file_path, "rb") as f:
                header_bytes = f.read(4096)

            if len(header_bytes) == 0:
                return 0, None

            # RIFF WAV Check
            if len(header_bytes) >= 12 and header_bytes[:4] == b"RIFF" and header_bytes[8:12] == b"WAVE":
                return 0, "WAV Header"

            # A text header is a leading run of printable bytes made of lines. A bare printable-byte
            # density test is not enough: CU8 samples sit around 127, so most of their bytes are printable.
            with open(file_path, "rb") as f:
                f_head = f.read(65536)
            run = 0
            for byte in f_head:
                if (32 <= byte <= 126) or byte in (9, 10, 13):
                    run += 1
                else:
                    break
            head = f_head[:run]
            if run >= 16 and b"\n" in head and (chr(f_head[0]).isalpha() or chr(f_head[0]) in "#{%;/<["):
                header_offset = head.rindex(b"\n") + 1
                logger.info(f"[FORMAT DETECTOR] Skipped {header_offset} bytes of ASCII metadata header at file start.")
                return header_offset, "ASCII Header"

        except Exception as e:
            logger.debug(f"Header check error: {e}")

        return 0, None

    @classmethod
    def auto_detect_iq_format(
        cls,
        file_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> IQDetectionResult:
        """
        IQ sample-type detection: CU8, CS8, CS16, CF32, CF64 (format_name uses the legacy
        names uint8 / int8 / int16 / float32 / complex128; ingestion.iq_formats maps both).

        1. Sidecar metadata (SigMF core:datatype, JSON 'datatype' or the generator's format block)
        2. Filename hints (_cu8, _cs8/_ci8, _cs16/_ci16/_sc16, _cf32/_fc32, _cf64/_fc64)
        3. Text-header skip, then blind byte-structure detection on up to three 256 KB regions
           (beginning / middle / end), majority vote (ingestion.iq_formats.detect_iq_datatype).
        """
        start_time = time.time()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        file_size = os.path.getsize(file_path)
        warnings: List[str] = []

        def result(key, endian, conf, method, scores, warn, offset=0):
            name = LEGACY_NAMES[key]
            bps = FORMATS[key][1]
            rem = (file_size - offset) % bps
            if rem:
                warn = warn + [f"Trailing {rem} bytes ignored (partial sample for {name})."]
            if conf < 80.0:
                warn = warn + ["Auto-detection confidence is below 80%. Please verify sample format."]
            return IQDetectionResult(name, round(float(conf), 1), bps, method, scores, conf < 80.0, warn,
                                     endianness=endian, header_offset=offset)

        # 1. Metadata sidecars
        base_path = os.path.splitext(file_path)[0]
        stem = base_path[:-4] if base_path.endswith("_f32") else base_path
        for m_path in (file_path + ".sigmf-meta", base_path + ".sigmf-meta", base_path + ".json",
                       stem + ".json", stem + "_metadata.json"):
            if not os.path.exists(m_path) or m_path == file_path:
                continue
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                dt = (meta.get("global", {}).get("core:datatype") or meta.get("datatype")
                      or meta.get("format", {}).get("files", {}).get(os.path.basename(file_path))
                      or meta.get("format", {}).get("datatype"))
                norm = normalize_datatype(dt) if dt else None
            except Exception:
                norm = None
            if norm:
                return result(norm[0], norm[1], 100.0, f"Metadata ({os.path.basename(m_path)}: {dt})",
                              {LEGACY_NAMES[norm[0]]: 1.0}, [])

        # 2. Filename hints
        fname = os.path.basename(file_path).lower()
        hint_map = [(r"[_.\-](cu8|uint8|u8)(?=[_.\-]|$)", "cu8"),
                    (r"[_.\-](cs8|ci8|sc8|int8|s8|i8)(?=[_.\-]|$)", "cs8"),
                    (r"[_.\-](cs16|ci16|sc16|int16|s16|i16)(?=[_.\-]|$)", "cs16"),
                    (r"[_.\-](cf32|fc32|float32|complex64|f32)(?=[_.\-]|$)", "cf32"),
                    (r"[_.\-](cf64|fc64|float64|complex128|f64)(?=[_.\-]|$)", "cf64")]
        for pattern, key in hint_map:
            if re.search(pattern, fname):
                endian = "big" if re.search(r"[_.\-]be(?=[_.\-]|$)", fname) else "little"
                return result(key, endian, 95.0, f"Filename Hint ('{fname}')", {LEGACY_NAMES[key]: 0.95}, [])

        # Disguised WAV
        try:
            with open(file_path, "rb") as f:
                hdr = f.read(12)
            if len(hdr) >= 12 and hdr[:4] == b"RIFF" and hdr[8:12] == b"WAVE":
                return result("cs16", "little", 100.0, "RIFF/WAVE Header", {"int16": 1.0},
                              ["File is a WAV recording disguised as IQ."])
        except Exception:
            pass

        # 3. Header skip + blind byte-structure detection
        header_offset, _ = cls.detect_header_offset(file_path)
        if header_offset > 0:
            warnings.append(f"Skipped {header_offset} bytes of header at file start.")
        data_bytes = file_size - header_offset
        if data_bytes <= 0:
            return result("cs16", "little", 50.0, "Fallback Default", {}, ["File contains only header data."],
                          header_offset)

        region = 256 * 1024
        if data_bytes <= 3 * region:
            regions = [("whole file", header_offset, data_bytes)]
        else:
            mid = header_offset + (data_bytes // 2 // 16) * 16 - region // 2
            regions = [("beginning", header_offset, region), ("middle", mid, region),
                       ("end", header_offset + ((data_bytes - region) // 16) * 16, region)]
        votes = []
        for i, (name, off, nbytes) in enumerate(regions):
            if progress_callback:
                progress_callback(int(100 * (i + 1) / len(regions)), f"Analyzing {name}...")
            with open(file_path, "rb") as f:
                f.seek(off)
                d = detect_iq_datatype(f.read(nbytes))
            if d["scores"]:                                    # region had enough non-zero data
                votes.append(d)
        if not votes:
            return result("cs16", "little", 50.0, "Fallback Default", {},
                          warnings + ["Auto-detection found no non-zero data to analyse."], header_offset)
        tally: Dict[Tuple[str, str], List[float]] = {}
        for d in votes:
            tally.setdefault((d["key"], d["endian"]), []).append(d["confidence"])
        (key, endian), confs = max(tally.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))
        conf = float(np.mean(confs)) * len(confs) / len(votes)
        if len(tally) > 1:
            warnings.append("Regions disagree: " + ", ".join(f"{k.upper()}x{len(v)}" for (k, _), v in tally.items()))
        best = max(votes, key=lambda d: d["confidence"] if (d["key"], d["endian"]) == (key, endian) else -1)
        logger.info(f"[FORMAT DETECTOR] {os.path.basename(file_path)}: {key} {endian} ({conf:.1f}%, "
                    f"{len(regions)} region(s), {time.time() - start_time:.3f}s) - {best['reason']}")
        return result(key, endian, conf, "Byte-structure analysis: " + best["reason"],
                      {k: float(v) for k, v in best["scores"].items()}, warnings, header_offset)

    @classmethod
    def load_signal(
        cls,
        file_path: str,
        force_format: Optional[str] = None,
        sample_rate: Optional[float] = None,
        center_frequency: float = 0.0,
        dtype: Optional[str] = None,
        endianness: str = "little",
        iq_order: str = "IQ",
        byte_offset: int = 0,
        wav_mode: str = "auto",
        max_samples: Optional[int] = None
    ) -> SignalBuffer:
        """
        Auto-detects format (or uses force_format) and delegates signal reading.
        """
        fmt = force_format if force_format else cls.detect_format(file_path)
        fmt_upper = fmt.upper()

        if fmt_upper == "WAV":
            buffer = WAVReader.read(
                file_path=file_path,
                mode=wav_mode,
                center_frequency=center_frequency,
                max_samples=max_samples
            )
            if (buffer.sample_rate is None or buffer.sample_rate <= 0) and sample_rate is not None:
                buffer.sample_rate = sample_rate
            return buffer

        elif fmt_upper == "SIGMF":
            if file_path.endswith(".sigmf-data"):
                meta_path = file_path.replace(".sigmf-data", ".sigmf-meta")
            else:
                meta_path = file_path
            buffer = SigMFReader.read(meta_file_path=meta_path, max_samples=max_samples)
            if (buffer.sample_rate is None or buffer.sample_rate <= 0) and sample_rate is not None:
                buffer.sample_rate = sample_rate
            return buffer

        else:  # 'IQ' or default fallback
            # Run 5-step format detection if dtype is None or 'auto'
            if dtype is None or dtype == "auto":
                det = cls.auto_detect_iq_format(file_path)
                dtype = det.format_name
                byte_offset = byte_offset or det.header_offset
                endianness = det.endianness if dtype in ("int16", "float32", "complex128") else endianness

            # Auto-check for companion metadata files (.json, .meta, .txt, .sigmf-meta)
            if sample_rate is None or sample_rate <= 0:
                base_path = os.path.splitext(file_path)[0]
                for meta_ext in [".json", ".meta", ".txt", ".sigmf-meta"]:
                    meta_candidate = base_path + meta_ext
                    if os.path.exists(meta_candidate):
                        try:
                            with open(meta_candidate, "r") as f:
                                meta_data = json.load(f)
                                if "sample_rate" in meta_data:
                                    sample_rate = float(meta_data["sample_rate"])
                                elif "global" in meta_data and "core:sample_rate" in meta_data["global"]:
                                    sample_rate = float(meta_data["global"]["core:sample_rate"])
                                break
                        except Exception:
                            pass

            return IQReader.read(
                file_path=file_path,
                dtype=dtype,
                endianness=endianness,
                iq_order=iq_order,
                byte_offset=byte_offset,
                sample_rate=sample_rate,
                center_frequency=center_frequency,
                max_samples=max_samples
            )
