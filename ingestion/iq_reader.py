import os
from typing import Optional, Dict, Any, Union
import numpy as np
from core.signal_buffer import SignalBuffer


class IQReader:
    """
    Reader for raw binary .IQ files supporting various dtypes, endianness,
    I/Q byte ordering, and offset configurations.
    """

    SUPPORTED_DTYPES = {
        "int16": np.int16,
        "int8": np.int8,
        "uint8": np.uint8,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }

    @staticmethod
    def auto_detect_dtype(file_path: str) -> str:
        """
        Estimate dtype based on file size heuristics and sample magnitude analysis.
        """
        size = os.path.getsize(file_path)

        # Inspect initial samples to verify if complex64 produces subnormal values (~1e-40)
        try:
            test_c64 = np.fromfile(file_path, dtype=np.complex64, count=64)
            if len(test_c64) > 0:
                mean_mag = float(np.mean(np.abs(test_c64)))
                # Subnormal magnitude indicates integer PCM / int16 encoding
                if mean_mag < 1e-10 or mean_mag > 1e10 or np.isnan(mean_mag):
                    return "int16"
        except Exception:
            pass

        if size % 8 == 0:
            return "complex64"
        elif size % 4 == 0:
            return "int16"
        else:
            return "uint8"

    @classmethod
    def read(
        cls,
        file_path: str,
        dtype: Optional[str] = None,
        endianness: str = "little",
        iq_order: str = "IQ",
        byte_offset: int = 0,
        sample_rate: Optional[float] = None,
        center_frequency: float = 0.0,
        max_samples: Optional[int] = None
    ) -> SignalBuffer:
        """
        Reads binary IQ file and produces a normalized SignalBuffer (np.complex64).

        Args:
            file_path: Path to the binary .iq / raw file.
            dtype: Format string ('int16', 'int8', 'uint8', 'float32', 'complex64', etc.).
            endianness: 'little' (<) or 'big' (>).
            iq_order: 'IQ' or 'QI'.
            byte_offset: Header offset in bytes to skip before reading samples.
            sample_rate: Optional sampling frequency in Hz.
            center_frequency: Center frequency in Hz.
            max_samples: Maximum number of complex samples to read (None for all).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"IQ file not found: {file_path}")

        if dtype is None or dtype == "auto":
            dtype = cls.auto_detect_dtype(file_path)

        dtype_clean = dtype.lower()
        if dtype_clean not in cls.SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(cls.SUPPORTED_DTYPES.keys())}")

        np_dtype = cls.SUPPORTED_DTYPES[dtype_clean]

        endian_prefix = "<" if endianness.lower() == "little" else ">"

        with open(file_path, "rb") as f:
            f.seek(byte_offset)

            if dtype_clean in ("complex64", "complex128"):
                dt = np.dtype(f"{endian_prefix}c8" if dtype_clean == "complex64" else f"{endian_prefix}c16")
                count = max_samples if max_samples is not None else -1
                raw_data = np.fromfile(f, dtype=dt, count=count)
                samples = raw_data.astype(np.complex64)
            else:
                # Interleaved I and Q samples
                dt = np.dtype(f"{endian_prefix}{np_dtype().dtype.char}")
                count = (max_samples * 2) if max_samples is not None else -1
                raw_data = np.fromfile(f, dtype=dt, count=count)

                if len(raw_data) % 2 != 0:
                    raw_data = raw_data[:-1]  # Ensure even pairs

                i_samples = raw_data[0::2].astype(np.float32)
                q_samples = raw_data[1::2].astype(np.float32)

                # Normalize integer types to [-1.0, 1.0]
                if dtype_clean == "int16":
                    i_samples /= 32768.0
                    q_samples /= 32768.0
                elif dtype_clean == "int8":
                    i_samples /= 128.0
                    q_samples /= 128.0
                elif dtype_clean == "uint8":
                    i_samples = (i_samples - 128.0) / 128.0
                    q_samples = (q_samples - 128.0) / 128.0

                if iq_order.upper() == "QI":
                    samples = (q_samples + 1j * i_samples).astype(np.complex64)
                else:
                    samples = (i_samples + 1j * q_samples).astype(np.complex64)

        metadata: Dict[str, Any] = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "dtype_used": dtype_clean,
            "endianness": endianness,
            "iq_order": iq_order,
            "byte_offset": byte_offset,
        }

        return SignalBuffer(
            samples=samples,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            source_format="IQ",
            metadata=metadata
        )
