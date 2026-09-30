from typing import List, Tuple
import numpy as np


class BitstreamCorrelation:
    """
    Bitstream correlation and periodicity detection tool.
    Computes bitwise autocorrelation and frame-to-frame distance matrices
    to detect frame lengths, repeating headers, and interleaver periods.
    """

    @classmethod
    def autocorrelation(cls, bits: np.ndarray, max_lag: int = 512) -> np.ndarray:
        """
        Computes normalized bitwise autocorrelation R[k] for k = 0..max_lag.

        Args:
            bits: 1D array of bits (0 or 1)
            max_lag: Maximum lag in bits

        Returns:
            1D array of correlation values in range [-1.0, 1.0] for lags 0..max_lag
        """
        if len(bits) < 2:
            return np.array([1.0], dtype=np.float32)

        bits = np.asarray(bits, dtype=np.uint8) & 1
        n = len(bits)
        max_lag = min(max_lag, n - 1)

        # Convert bits to bipolar {-1, +1}
        bipolar = (2 * bits.astype(np.float32)) - 1.0

        corrs = np.zeros(max_lag + 1, dtype=np.float32)
        corrs[0] = 1.0  # Zero lag correlation

        for k in range(1, max_lag + 1):
            prod = bipolar[: n - k] * bipolar[k:]
            corrs[k] = float(np.mean(prod))

        return corrs

    @classmethod
    def find_periodicity(
        cls,
        bits: np.ndarray,
        max_lag: int = 512,
        min_peak: float = 0.15
    ) -> List[Tuple[int, float]]:
        """
        Finds candidate periodicities (frame lengths or interleaver block sizes)
        by detecting local maxima in the autocorrelation sequence.

        Returns:
            List of (lag_in_bits, peak_correlation) sorted by peak correlation descending
        """
        r = cls.autocorrelation(bits, max_lag=max_lag)
        if len(r) < 3:
            return []

        peaks = []
        for k in range(2, len(r) - 1):
            if r[k] > r[k - 1] and r[k] > r[k + 1] and r[k] >= min_peak:
                peaks.append((k, float(r[k])))

        # Sort by correlation peak strength descending
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks

    @classmethod
    def distance_matrix(cls, bits: np.ndarray, frame_len: int) -> np.ndarray:
        """
        Computes normalized Hamming distance matrix between consecutive frames of size frame_len.

        Returns:
            2D symmetric numpy matrix of distances in range [0.0, 1.0]
        """
        if len(bits) < frame_len or frame_len <= 0:
            return np.zeros((0, 0), dtype=np.float32)

        bits = np.asarray(bits, dtype=np.uint8) & 1
        n_frames = len(bits) // frame_len
        frames = bits[: n_frames * frame_len].reshape(n_frames, frame_len)

        dist_mat = np.zeros((n_frames, n_frames), dtype=np.float32)
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                dist = float(np.mean(frames[i] != frames[j]))
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist

        return dist_mat
