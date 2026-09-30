"""
Automatic Gain Control (AGC) Module.
Normalizes complex baseband IQ samples to target RMS / average envelope power.
"""

from dataclasses import dataclass
import numpy as np

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


@dataclass
class AGCResult:
    buffer: SignalBuffer
    gain_applied: float
    parameter: Parameter


class AutomaticGainControl:
    """
    Normalizes input signal amplitude to target power.
    """

    @classmethod
    def process(cls, buffer: SignalBuffer, target_power: float = 1.0) -> AGCResult:
        if not buffer.is_valid():
            ev = Evidence(description="Buffer is invalid for AGC.", source="AGC", score=0.0)
            param = Parameter(name="agc_gain", value=1.0, unit="linear", source="UNKNOWN", method=None, confidence=0.0, evidence=[ev])
            return AGCResult(buffer=buffer, gain_applied=1.0, parameter=param)

        samples = buffer.samples
        p_current = np.mean(np.abs(samples)**2)

        if p_current <= 0:
            ev = Evidence(description="Zero power signal in AGC.", source="AGC", score=0.0)
            param = Parameter(name="agc_gain", value=1.0, unit="linear", source="UNKNOWN", method="NORMALIZATION", confidence=0.0, evidence=[ev])
            return AGCResult(buffer=buffer, gain_applied=1.0, parameter=param)

        gain = float(np.sqrt(target_power / p_current))
        norm_samples = (samples * gain).astype(np.complex64)

        norm_buffer = SignalBuffer(
            samples=norm_samples,
            sample_rate=buffer.sample_rate,
            center_frequency=buffer.center_frequency,
            start_time=buffer.start_time,
            source_format=buffer.source_format,
            metadata=buffer.metadata
        )

        ev = Evidence(
            description=f"AGC applied normalized gain of {gain:.4f} linear ({20.0 * np.log10(gain):.2f} dB)",
            source="AGC_MODULE",
            score=0.95,
            metadata={"gain_linear": gain, "gain_db": 20.0 * np.log10(gain)}
        )

        param = Parameter(
            name="agc_gain",
            value=gain,
            unit="linear",
            source="ESTIMATED",
            method="POWER_NORMALIZATION",
            confidence=0.95,
            evidence=[ev]
        )

        return AGCResult(buffer=norm_buffer, gain_applied=gain, parameter=param)
