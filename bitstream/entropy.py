from typing import Dict, Any
import numpy as np


class BitstreamEntropy:
    """
    Bitstream entropy and randomness analysis tool.
    Computes binary entropy, byte-level Shannon entropy, and block-wise entropy profiles
    to distinguish raw data, headers, scrambled/encrypted payloads, and sparse bitstreams.
    """

    @classmethod
    def shannon_entropy(cls, bits: np.ndarray, block_size: int = 8) -> float:
        """
        Computes Shannon entropy for a bitstream aggregated into block_size chunks.

        Args:
            bits: 1D array of bits (0 or 1)
            block_size: Size of symbol chunks in bits (default 8 for byte-level entropy)

        Returns:
            Shannon entropy in bits per block (0.0 to block_size, max 8.0 for bytes)
        """
        if len(bits) < block_size:
            return 0.0

        bits = np.asarray(bits, dtype=np.uint8) & 1
        n_blocks = len(bits) // block_size
        trimmed = bits[: n_blocks * block_size].reshape(-1, block_size)

        # Convert bit blocks to integer values
        powers = 2 ** np.arange(block_size - 1, -1, -1, dtype=np.uint32)
        symbols = np.sum(trimmed * powers, axis=1)

        # Compute symbol probability distribution
        _, counts = np.unique(symbols, return_counts=True)
        probs = counts / float(n_blocks)

        entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
        return min(float(block_size), max(0.0, entropy))

    @classmethod
    def binary_entropy(cls, bits: np.ndarray) -> float:
        """
        Computes binary Shannon entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).

        Returns:
            Entropy value in range [0.0, 1.0]
        """
        if len(bits) == 0:
            return 0.0

        bits = np.asarray(bits, dtype=np.uint8) & 1
        p1 = float(np.mean(bits))
        p0 = 1.0 - p1

        if p0 <= 0 or p1 <= 0:
            return 0.0

        h = -(p1 * np.log2(p1) + p0 * np.log2(p0))
        return float(np.clip(h, 0.0, 1.0))

    @classmethod
    def block_entropy(cls, bits: np.ndarray, window_bits: int = 256, step_bits: int = 64) -> np.ndarray:
        """
        Computes sliding block entropy profile over the bitstream.

        Returns:
            1D array of byte-level entropy values across sliding windows
        """
        if len(bits) < window_bits:
            return np.array([cls.shannon_entropy(bits, block_size=8)], dtype=np.float32)

        n = len(bits)
        profile = []
        for start in range(0, n - window_bits + 1, step_bits):
            chunk = bits[start : start + window_bits]
            ent = cls.shannon_entropy(chunk, block_size=8)
            profile.append(ent)

        return np.array(profile, dtype=np.float32)

    @classmethod
    def estimate_randomness(cls, bits: np.ndarray) -> Dict[str, float]:
        """
        Provides comprehensive bitstream randomness and statistics overview.
        """
        if len(bits) == 0:
            return {"bit_balance": 0.5, "binary_entropy": 0.0, "byte_entropy": 0.0, "randomness_score": 0.0}

        bits = np.asarray(bits, dtype=np.uint8) & 1
        p1 = float(np.mean(bits))
        bin_ent = cls.binary_entropy(bits)
        byte_ent = cls.shannon_entropy(bits, block_size=8)

        # Randomness score: normalized product of binary & byte entropy (1.0 = ideal white noise/encrypted)
        rand_score = float(bin_ent * (byte_ent / 8.0))

        return {
            "bit_balance": p1,
            "binary_entropy": bin_ent,
            "byte_entropy": byte_ent,
            "randomness_score": rand_score
        }
