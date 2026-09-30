from typing import List
import numpy as np


class DifferentialDecoder:
    """
    Differential bit encoding and decoding for phase-ambiguous or differentially encoded signals
    (e.g., DBPSK, DQPSK bitstreams).
    """

    SUPPORTED_MODES = ["DISABLED", "DIFF_DECODE", "DIFF_ENCODE"]

    @classmethod
    def candidate_generator(cls) -> List[str]:
        """
        Yields candidate modes for hypothesis testing downstream.
        """
        return ["DISABLED", "DIFF_DECODE"]

    @classmethod
    def process(
        cls,
        bits: np.ndarray,
        mode: str = "DIFF_DECODE",
        init_bit: int = 0
    ) -> np.ndarray:
        """
        Processes 1D bit array with differential decoding or encoding.

        Args:
            bits: 1D array of bits (0 or 1)
            mode: 'DISABLED' (pass-through), 'DIFF_DECODE', or 'DIFF_ENCODE'
            init_bit: Reference initial bit state (0 or 1)

        Returns:
            Processed 1D bit array
        """
        if len(bits) == 0:
            return np.array([], dtype=np.uint8)

        bits = np.asarray(bits, dtype=np.uint8) & 1

        if mode.upper() in ("DISABLED", "OFF", "NONE"):
            return bits.copy()

        elif mode.upper() in ("DIFF_DECODE", "DECODE"):
            # b[n] = d[n] XOR d[n-1]
            prev_bits = np.empty_like(bits)
            prev_bits[0] = init_bit & 1
            prev_bits[1:] = bits[:-1]
            decoded = bits ^ prev_bits
            return decoded.astype(np.uint8)

        elif mode.upper() in ("DIFF_ENCODE", "ENCODE"):
            # d[n] = d[n-1] XOR b[n]
            encoded = np.empty_like(bits)
            state = init_bit & 1
            for i, b in enumerate(bits):
                state = state ^ int(b)
                encoded[i] = state
            return encoded.astype(np.uint8)

        else:
            raise ValueError(f"Unsupported differential mode: {mode}. Must be one of {cls.SUPPORTED_MODES}")
