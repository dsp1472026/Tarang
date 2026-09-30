from typing import List
import numpy as np


class BitOrderConverter:
    """
    Bit ordering conversion tool supporting MSB-first, LSB-first,
    and byte-wise bit-reversal transformations for downstream FEC and framing.
    """

    SUPPORTED_MODES = ["MSB", "LSB", "SWAP_BITS_PER_BYTE"]

    @classmethod
    def candidate_generator(cls) -> List[str]:
        """
        Yields all supported bit ordering candidates for hypothesis testing.
        """
        return list(cls.SUPPORTED_MODES)

    @classmethod
    def convert_bit_order(cls, bits: np.ndarray, mode: str = "MSB") -> np.ndarray:
        """
        Converts 1D array of bits (0 or 1) based on the specified bit ordering mode.

        Modes:
            - 'MSB': Keeps standard MSB-first ordering (bit 7 down to bit 0 per byte).
            - 'LSB': Reverses the full bit vector.
            - 'SWAP_BITS_PER_BYTE': Reverses bits within each 8-bit chunk.
        """
        if len(bits) == 0:
            return np.array([], dtype=np.uint8)

        bits = np.asarray(bits, dtype=np.uint8) & 1

        if mode.upper() == "MSB":
            return bits.copy()

        elif mode.upper() == "LSB":
            return bits[::-1].copy()

        elif mode.upper() == "SWAP_BITS_PER_BYTE":
            out_bits = bits.copy()
            n_bytes = len(bits) // 8
            for i in range(n_bytes):
                chunk = out_bits[i * 8 : (i + 1) * 8]
                out_bits[i * 8 : (i + 1) * 8] = chunk[::-1]
            return out_bits

        else:
            raise ValueError(f"Unsupported bit order mode: {mode}. Must be one of {cls.SUPPORTED_MODES}")

    @classmethod
    def pack_bits(cls, bits: np.ndarray, mode: str = "MSB") -> bytes:
        """
        Packs a 1D array of bits (0 or 1) into bytes according to specified bit ordering mode.
        """
        conv_bits = cls.convert_bit_order(bits, mode=mode)
        n_bytes = len(conv_bits) // 8
        if n_bytes == 0:
            return b""

        trimmed_bits = conv_bits[: n_bytes * 8].reshape(-1, 8)
        powers = 2 ** np.arange(7, -1, -1, dtype=np.uint8)
        byte_vals = np.sum(trimmed_bits * powers, axis=1, dtype=np.uint8)
        return byte_vals.tobytes()

    @classmethod
    def unpack_bytes(cls, data: bytes, mode: str = "MSB") -> np.ndarray:
        """
        Unpacks bytes into a 1D array of bits (0 or 1) according to specified bit ordering mode.
        """
        if not data:
            return np.array([], dtype=np.uint8)

        byte_arr = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(byte_arr)
        return cls.convert_bit_order(bits, mode=mode)
