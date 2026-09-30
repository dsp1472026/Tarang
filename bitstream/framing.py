from typing import List, Dict, Any
import numpy as np
from bitstream.correlation import BitstreamCorrelation
from bitstream.patterns import PatternDetector


class FramingForensics:
    """
    Burst structure and frame length estimation forensics tool.
    Analyzes continuous or packetized bitstreams to isolate burst transmission boundaries,
    inter-burst gaps, and candidate frame length parameters.
    """

    @classmethod
    def detect_bursts(
        cls,
        bits: np.ndarray,
        window_size: int = 64,
        activity_threshold: float = 0.15
    ) -> List[Dict[str, Any]]:
        """
        Detects active data transmission bursts in a bitstream based on density/activity windowing.

        Args:
            bits: 1D array of bits (0 or 1)
            window_size: Moving window size in bits
            activity_threshold: Minimum transition/density threshold to mark active transmission

        Returns:
            List of burst entries: [{"start_bit": int, "end_bit": int, "length_bits": int}]
        """
        if len(bits) < window_size:
            return [{"start_bit": 0, "end_bit": len(bits), "length_bits": len(bits)}] if len(bits) > 0 else []

        bits = np.asarray(bits, dtype=np.uint8) & 1

        # Activity metric: bit transition count within window
        diffs = np.abs(np.diff(bits))
        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        activity = np.convolve(diffs, kernel, mode="same")

        is_active = activity >= activity_threshold

        bursts = []
        in_burst = False
        start_idx = 0

        for idx, active in enumerate(is_active):
            if active and not in_burst:
                in_burst = True
                start_idx = idx
            elif not active and in_burst:
                in_burst = False
                end_idx = idx
                if end_idx - start_idx >= window_size:
                    bursts.append({
                        "start_bit": start_idx,
                        "end_bit": end_idx,
                        "length_bits": end_idx - start_idx
                    })

        if in_burst:
            bursts.append({
                "start_bit": start_idx,
                "end_bit": len(bits),
                "length_bits": len(bits) - start_idx
            })

        if not bursts and len(bits) > 0:
            bursts.append({"start_bit": 0, "end_bit": len(bits), "length_bits": len(bits)})

        return bursts

    @classmethod
    def estimate_frame_length(
        cls,
        bits: np.ndarray,
        min_len: int = 32,
        max_len: int = 1024
    ) -> Dict[str, Any]:
        """
        Estimates candidate frame length using autocorrelation peaks and sync pattern periodicity.

        Returns:
            Dict containing best_frame_length, confidence, and candidate_lengths
        """
        if len(bits) < min_len * 2:
            return {"best_frame_length": None, "confidence": 0.0, "candidates": []}

        # 1. Autocorrelation periodicity candidates
        corr_peaks = BitstreamCorrelation.find_periodicity(bits, max_lag=max_len, min_peak=0.1)

        # 2. Sync pattern periodicity candidates
        sync_candidates = PatternDetector.find_sync_candidates(bits, word_len=16, min_period=min_len)

        candidates_map = {}
        for lag, score in corr_peaks:
            if min_len <= lag <= max_len:
                candidates_map[lag] = candidates_map.get(lag, 0.0) + float(score) * 0.5

        for sc in sync_candidates:
            period = sc["period_bits"]
            if min_len <= period <= max_len:
                candidates_map[period] = candidates_map.get(period, 0.0) + float(sc["period_count"]) * 0.3

        if not candidates_map:
            return {"best_frame_length": None, "confidence": 0.0, "candidates": []}

        sorted_cands = sorted(candidates_map.items(), key=lambda x: x[1], reverse=True)
        best_len, best_score = sorted_cands[0]
        confidence = float(np.clip(best_score, 0.0, 1.0))

        return {
            "best_frame_length": best_len,
            "confidence": confidence,
            "candidates": [{"length_bits": k, "score": float(v)} for k, v in sorted_cands[:5]]
        }
