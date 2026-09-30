from typing import List, Dict, Any
from collections import defaultdict
import numpy as np


class PatternDetector:
    """
    Bitstream pattern and n-gram frequency analysis tool.
    Detects repeated bit patterns, preamble/sync candidate words, and periodic bit motifs.
    """

    @classmethod
    def find_repeated_patterns(
        cls,
        bits: np.ndarray,
        min_len: int = 8,
        max_len: int = 32,
        min_count: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Scans bitstream for repeated n-gram bit sequences.

        Args:
            bits: 1D array of bits (0 or 1)
            min_len: Minimum pattern length in bits
            max_len: Maximum pattern length in bits
            min_count: Minimum occurrence count

        Returns:
            List of dictionaries sorted by occurrence count descending:
            [{"pattern_bits": np.ndarray, "hex": str, "length": int, "count": int, "positions": List[int]}]
        """
        if len(bits) < min_len:
            return []

        bits = np.asarray(bits, dtype=np.uint8) & 1
        n = len(bits)

        results = []
        for length in range(min_len, min(max_len + 1, n + 1), 8):
            occurrences = defaultdict(list)
            for i in range(0, n - length + 1, 8):
                chunk_tuple = tuple(bits[i : i + length])
                occurrences[chunk_tuple].append(i)

            for pat_tuple, pos_list in occurrences.items():
                if len(pos_list) >= min_count:
                    pat_arr = np.array(pat_tuple, dtype=np.uint8)
                    # Convert to hex representation
                    n_bytes = len(pat_arr) // 8
                    if n_bytes > 0:
                        powers = 2 ** np.arange(7, -1, -1, dtype=np.uint8)
                        bytes_val = np.sum(pat_arr[: n_bytes * 8].reshape(-1, 8) * powers, axis=1)
                        hex_str = bytes_val.tobytes().hex().upper()
                    else:
                        hex_str = "".join(map(str, pat_arr))

                    results.append({
                        "pattern_bits": pat_arr,
                        "hex": hex_str,
                        "length": length,
                        "count": len(pos_list),
                        "positions": pos_list
                    })

        results.sort(key=lambda x: (x["count"], x["length"]), reverse=True)
        return results

    @classmethod
    def find_sync_candidates(
        cls,
        bits: np.ndarray,
        word_len: int = 16,
        min_period: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Detects sync word candidates by finding patterns that repeat at constant bit intervals.

        Returns:
            List of sync candidates: [{"sync_hex": str, "period_bits": int, "count": int, "positions": List[int]}]
        """
        if len(bits) < word_len * 2:
            return []

        bits = np.asarray(bits, dtype=np.uint8) & 1
        patterns = cls.find_repeated_patterns(bits, min_len=word_len, max_len=word_len, min_count=2)

        candidates = []
        for p in patterns:
            pos = p["positions"]
            if len(pos) >= 2:
                diffs = np.diff(pos)
                unique_diffs, counts = np.unique(diffs, return_counts=True)
                most_common_period = int(unique_diffs[np.argmax(counts)])
                period_count = int(np.max(counts))

                if most_common_period >= min_period and period_count >= 1:
                    candidates.append({
                        "sync_hex": p["hex"],
                        "sync_bits": p["pattern_bits"],
                        "period_bits": most_common_period,
                        "count": p["count"],
                        "period_count": period_count,
                        "positions": pos
                    })

        candidates.sort(key=lambda x: (x["period_count"], x["count"]), reverse=True)
        return candidates
