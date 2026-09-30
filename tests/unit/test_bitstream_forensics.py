import numpy as np
import pytest
from bitstream import BitstreamEntropy, BitstreamCorrelation, PatternDetector, FramingForensics


def test_bitstream_entropy():
    # Constant 0 bits -> zero entropy
    zeros = np.zeros(256, dtype=np.uint8)
    assert BitstreamEntropy.binary_entropy(zeros) == 0.0
    assert BitstreamEntropy.shannon_entropy(zeros, block_size=8) == 0.0

    # Random / alternating bits -> high entropy
    rnd = np.random.randint(0, 2, size=512, dtype=np.uint8)
    bin_ent = BitstreamEntropy.binary_entropy(rnd)
    assert bin_ent > 0.9

    stats = BitstreamEntropy.estimate_randomness(rnd)
    assert "randomness_score" in stats


def test_bitstream_correlation_and_periodicity():
    # Repeated 32-bit pattern
    pattern = np.random.randint(0, 2, size=32, dtype=np.uint8)
    stream = np.tile(pattern, 16)

    corrs = BitstreamCorrelation.autocorrelation(stream, max_lag=128)
    assert corrs[32] > 0.8  # Strong correlation peak at lag 32

    periodicities = BitstreamCorrelation.find_periodicity(stream, max_lag=128)
    assert len(periodicities) > 0
    lags = [p[0] for p in periodicities]
    assert 32 in lags or 64 in lags


def test_pattern_detector():
    sync = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8)
    payload = np.random.randint(0, 2, size=64, dtype=np.uint8)
    frame = np.concatenate([sync, payload])
    stream = np.tile(frame, 4)

    patterns = PatternDetector.find_repeated_patterns(stream, min_len=16, max_len=16, min_count=2)
    assert len(patterns) > 0

    sync_cands = PatternDetector.find_sync_candidates(stream, word_len=16, min_period=32)
    assert len(sync_cands) > 0


def test_framing_forensics():
    # Periodic burst signal with quiet regions
    active_burst = np.tile(np.array([1, 0], dtype=np.uint8), 64)
    quiet_gap = np.zeros(128, dtype=np.uint8)
    signal = np.concatenate([quiet_gap, active_burst, quiet_gap, active_burst])

    bursts = FramingForensics.detect_bursts(signal, window_size=32, activity_threshold=0.2)
    assert len(bursts) >= 1

    # Frame length estimation
    frame_info = FramingForensics.estimate_frame_length(np.tile(active_burst, 4), min_len=32, max_len=256)
    assert "best_frame_length" in frame_info
