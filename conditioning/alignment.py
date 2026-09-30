from typing import List, Tuple
import numpy as np


class BitAlignment:
    """
    Bit alignment and bit-shifting module to test byte boundary offsets 0..7
    and align bitstreams to sync words or preamble patterns.
    """

    @classmethod
    def candidate_generator(cls) -> List[int]:
        """
        Yields all candidate bit offset shifts (0 through 7).
        """
        return list(range(8))

    @classmethod
    def shift_bits(cls, bits: np.ndarray, shift: int = 0) -> np.ndarray:
        """
        Shifts bit array by dropping the first 'shift' bits.

        Args:
            bits: 1D array of bits (0 or 1)
            shift: Bit offset to drop from start (0..7)

        Returns:
            Shifted bit array starting from index 'shift'
        """
        if len(bits) == 0 or shift <= 0:
            return np.asarray(bits, dtype=np.uint8).copy()

        shift = shift % len(bits) if len(bits) > 0 else 0
        return np.asarray(bits[shift:], dtype=np.uint8).copy()

    @classmethod
    def align_to_pattern(cls, bits: np.ndarray, pattern_bits: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Searches for exact or best match of pattern_bits in the first 256 bits of the stream.

        Returns:
            Tuple of (aligned_bits, best_offset) where best_offset is in [0..7]
            or start index of pattern.
        """
        if len(bits) == 0 or len(pattern_bits) == 0:
            return np.asarray(bits, dtype=np.uint8).copy(), 0

        bits = np.asarray(bits, dtype=np.uint8) & 1
        pat = np.asarray(pattern_bits, dtype=np.uint8) & 1
        pat_len = len(pat)

        best_offset = 0
        max_matches = -1

        search_range = min(len(bits) - pat_len + 1, 256)
        if search_range <= 0:
            return bits.copy(), 0

        for offset in range(search_range):
            chunk = bits[offset : offset + pat_len]
            matches = int(np.sum(chunk == pat))
            if matches > max_matches:
                max_matches = matches
                best_offset = offset
                if matches == pat_len:
                    break  # Exact match found

        aligned_bits = bits[best_offset:].copy()
        return aligned_bits, best_offset
