import os
import json
from typing import Optional, Dict, Any
import numpy as np
from core.signal_buffer import SignalBuffer
from ingestion.iq_reader import IQReader


class SigMFReader:
    """
    Reader for SigMF metadata (.sigmf-meta) and binary data (.sigmf-data).
    """

    SIGMF_DATATYPE_MAP = {
        "cf32_le": ("complex64", "little"),
        "cf32_be": ("complex64", "big"),
        "ci16_le": ("int16", "little"),
        "ci16_be": ("int16", "big"),
        "ci8": ("int8", "little"),
        "cu8": ("uint8", "little"),
        "f32_le": ("float32", "little"),
        "f32_be": ("float32", "big"),
    }

    @classmethod
    def read(
        cls,
        meta_file_path: str,
        data_file_path: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> SignalBuffer:
        """
        Reads a SigMF dataset starting from the metadata file path.
        """
        if not os.path.exists(meta_file_path):
            raise FileNotFoundError(f"SigMF metadata file not found: {meta_file_path}")

        if data_file_path is None:
            # Infer .sigmf-data path
            base, ext = os.path.splitext(meta_file_path)
            if ext == ".sigmf-meta":
                data_file_path = base + ".sigmf-data"
            else:
                data_file_path = meta_file_path + ".sigmf-data"

        if not os.path.exists(data_file_path):
            # Try same directory with .sigmf-data extension
            dir_name = os.path.dirname(meta_file_path)
            base_name = os.path.basename(meta_file_path).replace(".sigmf-meta", "")
            alt_data = os.path.join(dir_name, base_name + ".sigmf-data")
            if os.path.exists(alt_data):
                data_file_path = alt_data
            else:
                raise FileNotFoundError(f"SigMF data file not found at: {data_file_path}")

        with open(meta_file_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        global_meta = meta_json.get("global", {})
        captures = meta_json.get("captures", [{}])

        sigmf_dtype = global_meta.get("core:datatype", "cf32_le")
        sample_rate = global_meta.get("core:sample_rate", None)
        if sample_rate is not None:
            sample_rate = float(sample_rate)

        center_freq = 0.0
        if captures and isinstance(captures, list) and len(captures) > 0:
            center_freq = float(captures[0].get("core:frequency", 0.0))

        dtype_str, endianness = cls.SIGMF_DATATYPE_MAP.get(sigmf_dtype, ("complex64", "little"))

        buf = IQReader.read(
            file_path=data_file_path,
            dtype=dtype_str,
            endianness=endianness,
            iq_order="IQ",
            byte_offset=0,
            sample_rate=sample_rate,
            center_frequency=center_freq,
            max_samples=max_samples
        )

        buf.source_format = "SigMF"
        buf.metadata.update({"sigmf_global": global_meta, "sigmf_captures": captures})
        return buf
