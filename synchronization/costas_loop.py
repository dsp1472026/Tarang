"""
Carrier Frequency and Phase Synchronization using Costas Loop / M-th Power Loop.
Locks phase and removes residual CFO from PSK / QAM baseband signals.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


@dataclass
class CostasResult:
    buffer: SignalBuffer
    cfo_residual_hz: float
    phase_error_rad: float
    parameter: Parameter


class CostasLoop:
    """
    Costas Loop / Phase Locked Loop for carrier recovery.
    """

    @classmethod
    def process(
        cls,
        buffer: SignalBuffer,
        order: int = 4,              # 2 for BPSK, 4 for QPSK/QAM, 8 for 8PSK
        loop_bandwidth: float = 0.01
    ) -> CostasResult:
        if not buffer.is_valid():
            ev = Evidence(description="Buffer is invalid for Costas Loop.", source="COSTAS_LOOP", score=0.0)
            param = Parameter(name="carrier_sync", value=0.0, unit="Hz", source="UNKNOWN", method=None, confidence=0.0, evidence=[ev])
            return CostasResult(buffer=buffer, cfo_residual_hz=0.0, phase_error_rad=0.0, parameter=param)

        fs = float(buffer.sample_rate) if buffer.sample_rate and buffer.sample_rate > 0 else 1.0
        samples = buffer.samples
        n = len(samples)

        # Loop filter parameters (2nd order DPLL)
        damping = 0.707
        wn = loop_bandwidth
        alpha = (4.0 * damping * wn) / (1.0 + 2.0 * damping * wn + wn**2)
        beta = (4.0 * wn**2) / (1.0 + 2.0 * damping * wn + wn**2)

        phase = 0.0
        freq = 0.0
        out_samples = np.zeros(n, dtype=np.complex64)

        for i in range(n):
            # Derotate sample by current estimated phase
            out = samples[i] * np.exp(-1j * phase)
            out_samples[i] = out

            # Phase Error Detector (PED) based on constellation order
            if order == 2:
                # BPSK PED: Re(out) * Im(out)
                error = float(out.real * out.imag)
            elif order == 8:
                # 8PSK PED
                error = float(np.sin(8.0 * np.angle(out)))
            else:
                # QPSK / QAM PED: sign(Re(out)) * Im(out) - sign(Im(out)) * Re(out)
                error = float(np.sign(out.real) * out.imag - np.sign(out.imag) * out.real)

            # Loop filter update
            freq += beta * error
            phase += freq + alpha * error

        # Compute residual CFO in Hz
        residual_cfo_hz = float((freq * fs) / (2.0 * np.pi))

        sync_buffer = SignalBuffer(
            samples=out_samples,
            sample_rate=buffer.sample_rate,
            center_frequency=buffer.center_frequency,
            start_time=buffer.start_time,
            source_format=buffer.source_format,
            metadata=buffer.metadata
        )

        ev = Evidence(
            description=f"Costas Loop carrier recovery locked (Order: {order}, Residual CFO: {residual_cfo_hz:.2f} Hz)",
            source="COSTAS_LOOP",
            score=0.90,
            metadata={"order": order, "residual_cfo_hz": residual_cfo_hz, "loop_bandwidth": loop_bandwidth}
        )

        param = Parameter(
            name="carrier_sync",
            value=residual_cfo_hz,
            unit="Hz",
            source="ESTIMATED",
            method=f"COSTAS_LOOP_ORDER_{order}",
            confidence=0.90,
            evidence=[ev]
        )

        return CostasResult(buffer=sync_buffer, cfo_residual_hz=residual_cfo_hz, phase_error_rad=float(phase), parameter=param)
