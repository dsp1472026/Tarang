import os
import struct
import numpy as np
from scipy.signal import hilbert
from typing import Optional, Dict, Any, Tuple
from core.signal_buffer import SignalBuffer


class WAVReader:
    """
    Parser for RIFF/WAVE files. Parses header metadata, format chunks,
    and converts stereo or mono audio data into a SignalBuffer (complex IQ or real).
    """

    @classmethod
    def parse_header(cls, file_path: str) -> Dict[str, Any]:
        """
        Parses RIFF WAVE header chunks.
        """
        metadata: Dict[str, Any] = {}
        with open(file_path, "rb") as f:
            riff_header = f.read(12)
            if len(riff_header) < 12 or riff_header[:4] != b"RIFF" or riff_header[8:12] != b"WAVE":
                raise ValueError(f"File is not a valid RIFF/WAVE file: {file_path}")

            metadata["file_size"] = os.path.getsize(file_path)

            # Read chunks
            data_offset = None
            data_size = 0
            fmt_parsed = False

            while True:
                chunk_hdr = f.read(8)
                if len(chunk_hdr) < 8:
                    break
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_hdr)
                chunk_id_str = chunk_id.decode("latin1", errors="ignore")

                if chunk_id_str == "fmt ":
                    fmt_data = f.read(chunk_size)
                    if len(fmt_data) >= 16:
                        audio_fmt, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack(
                            "<HHIIHH", fmt_data[:16]
                        )
                        metadata["audio_format"] = audio_fmt  # 1 = PCM, 3 = IEEE Float
                        metadata["channels"] = channels
                        metadata["sample_rate"] = sample_rate
                        metadata["byte_rate"] = byte_rate
                        metadata["block_align"] = block_align
                        metadata["bits_per_sample"] = bits_per_sample
                        fmt_parsed = True
                    if chunk_size > 16:
                        # Skip extra params if any
                        pass
                elif chunk_id_str == "data":
                    data_offset = f.tell()
                    data_size = chunk_size
                    f.seek(chunk_size, 1)  # Skip data chunk for now
                else:
                    # Skip unknown or metadata chunk (e.g. LIST, info)
                    f.seek(chunk_size, 1)

                # Alignment pad byte if chunk_size is odd
                if chunk_size % 2 != 0:
                    f.seek(1, 1)

            if not fmt_parsed or data_offset is None:
                raise ValueError("Corrupt WAV file: missing fmt or data chunk.")

            metadata["data_offset"] = data_offset
            metadata["data_size"] = data_size

        return metadata

    @classmethod
    def read(
        cls,
        file_path: str,
        mode: str = "auto",  # 'auto', 'STEREO_IQ', 'STEREO_AUDIO', 'MONO_REAL', 'MONO_ANALYTIC'
        center_frequency: float = 0.0,
        max_samples: Optional[int] = None
    ) -> SignalBuffer:
        """
        Reads WAV file into a SignalBuffer.

        Modes:
            - 'STEREO_IQ': Channel 0 = I, Channel 1 = Q -> complex64
            - 'STEREO_AUDIO': Channel 0 & 1 averaged or treated as complex analytic
            - 'MONO_REAL': Single channel real samples -> complex64 (imag=0)
            - 'MONO_ANALYTIC': Single channel transformed via Hilbert transform to analytic complex IQ
            - 'auto': Stereo -> STEREO_IQ, Mono -> MONO_ANALYTIC
        """
        meta = cls.parse_header(file_path)
        channels = meta["channels"]
        sample_rate = float(meta["sample_rate"])
        bits_per_sample = meta["bits_per_sample"]
        audio_format = meta["audio_format"]
        data_offset = meta["data_offset"]
        data_size = meta["data_size"]

        if mode == "auto":
            mode = "STEREO_IQ" if channels == 2 else "MONO_ANALYTIC"

        # Determine dtype for numpy reading
        if audio_format == 3:  # IEEE Float
            dtype = np.float32 if bits_per_sample == 32 else np.float64
        elif bits_per_sample == 16:
            dtype = np.int16
        elif bits_per_sample == 8:
            dtype = np.uint8
        elif bits_per_sample == 32:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported bits per sample in WAV: {bits_per_sample}")

        bytes_per_sample = bits_per_sample // 8
        frame_bytes = bytes_per_sample * channels
        total_frames = data_size // frame_bytes

        if max_samples is not None:
            frames_to_read = min(max_samples, total_frames)
        else:
            frames_to_read = total_frames

        with open(file_path, "rb") as f:
            f.seek(data_offset)
            raw_bytes = f.read(frames_to_read * frame_bytes)

        raw_arr = np.frombuffer(raw_bytes, dtype=dtype)
        if channels > 1:
            raw_arr = raw_arr.reshape(-1, channels)

        # Normalize raw data to float32 range [-1.0, 1.0]
        if dtype == np.int16:
            norm_arr = raw_arr.astype(np.float32) / 32768.0
        elif dtype == np.int32:
            norm_arr = raw_arr.astype(np.float32) / 2147483648.0
        elif dtype == np.uint8:
            norm_arr = (raw_arr.astype(np.float32) - 128.0) / 128.0
        else:
            norm_arr = raw_arr.astype(np.float32)

        # Process according to selected mode
        if mode == "STEREO_IQ" and channels >= 2:
            i_ch = norm_arr[:, 0]
            q_ch = norm_arr[:, 1]
            samples = (i_ch + 1j * q_ch).astype(np.complex64)
        elif mode == "MONO_ANALYTIC" or (mode == "auto" and channels == 1):
            if channels > 1:
                real_ch = norm_arr[:, 0]
            else:
                real_ch = norm_arr
            # Hilbert transform to create analytic complex signal
            samples = hilbert(real_ch).astype(np.complex64)
        else:  # MONO_REAL or default fallback
            if channels > 1:
                real_ch = norm_arr[:, 0]
            else:
                real_ch = norm_arr
            samples = (real_ch + 0j).astype(np.complex64)

        meta["read_mode"] = mode
        return SignalBuffer(
            samples=samples,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            source_format="WAV",
            metadata=meta
        )
