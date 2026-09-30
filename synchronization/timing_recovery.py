"""
Symbol Timing Recovery using Gardner / Mueller-Müller Timing Error Detector.
Downsamples IQ signal to 1 Sample Per Symbol (SPS) for symbol decision & constellation plotting.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


@dataclass
class TimingRecoveryResult:
    buffer: SignalBuffer             # 1 SPS synchronized symbol buffer
    sps_est: float
    timing_error_std: float
    parameter: Parameter


class SymbolTimingRecovery:
    """
    Gardner Timing Recovery with linear fractional interpolation.
    Outputs clean 1 SPS symbol stream.
    """

    @classmethod
    def process(
        cls,
        buffer: SignalBuffer,
        samples_per_symbol: float = 8.0,
        loop_gain: float = 0.01
    ) -> TimingRecoveryResult:
        if not buffer.is_valid() or samples_per_symbol <= 1.0:
            ev = Evidence(description="Buffer invalid or SPS <= 1.0 for timing recovery.", source="TIMING_RECOVERY", score=0.0)
            param = Parameter(name="timing_recovery", value=1.0, unit="SPS", source="UNKNOWN", method=None, confidence=0.0, evidence=[ev])
            return TimingRecoveryResult(buffer=buffer, sps_est=samples_per_symbol, timing_error_std=0.0, parameter=param)

        samples = buffer.samples
        n = len(samples)

        # Output 1 SPS symbol array
        symbols = []
        errors = []

        # Select initial sampling phase at eye diagram center (halfway from symbol transition peak)
        sps = float(samples_per_symbol)
        int_sps = max(1, int(round(sps)))
        if int_sps > 1 and n >= int_sps * 4:
            diffs = np.abs(samples[1:] - samples[:-1])
            eval_len = min(len(diffs), int_sps * 100)
            ph_trans = max(range(int_sps), key=lambda ph: float(np.mean(diffs[ph:eval_len:int_sps])))
            idx = float((ph_trans + int_sps // 2) % int_sps)
        else:
            idx = 0.0

        prev_st = 0.0j
        mid_st = 0.0j

        while idx < n - 2:
            i_int = int(np.floor(idx))
            f = idx - i_int

            # Fractional linear interpolation
            curr_st = (1.0 - f) * samples[i_int] + f * samples[i_int + 1]

            # Gardner TED: Error = Re{(curr - prev) * conj(mid)}
            mid_idx = idx - sps / 2.0
            if mid_idx >= 0 and int(mid_idx) + 1 < n:
                m_int = int(np.floor(mid_idx))
                m_f = mid_idx - m_int
                mid_st = (1.0 - m_f) * samples[m_int] + m_f * samples[m_int + 1]
            else:
                mid_st = 0.0j

            ted_error = float(np.real((curr_st - prev_st) * np.conj(mid_st)))
            errors.append(ted_error)

            symbols.append(curr_st)
            prev_st = curr_st

            # Update step index with loop filter correction
            idx += sps + loop_gain * ted_error

        sym_array = np.array(symbols, dtype=np.complex64)
        err_std = float(np.std(errors)) if errors else 0.0

        # Symbol buffer at 1 SPS (symbol rate = sample_rate / sps)
        sym_fs = (buffer.sample_rate / sps) if buffer.sample_rate else 1.0
        sym_buffer = SignalBuffer(
            samples=sym_array,
            sample_rate=sym_fs,
            center_frequency=buffer.center_frequency,
            start_time=buffer.start_time,
            source_format=buffer.source_format,
            metadata=buffer.metadata
        )

        ev = Evidence(
            description=f"Gardner timing recovery produced {len(sym_array)} symbols at 1 SPS (error std: {err_std:.4f})",
            source="GARDNER_TIMING_RECOVERY",
            score=0.92,
            metadata={"output_symbols": len(sym_array), "sps_in": sps, "error_std": err_std}
        )

        param = Parameter(
            name="timing_recovery",
            value=1.0,
            unit="SPS",
            source="ESTIMATED",
            method="GARDNER_TED",
            confidence=0.92,
            evidence=[ev]
        )

        return TimingRecoveryResult(buffer=sym_buffer, sps_est=sps, timing_error_std=err_std, parameter=param)
