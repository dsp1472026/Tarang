"""
Symbol Demodulation & Constellation Mapper.
Demodulates 1 SPS synchronized symbols into hard decision bits and soft LLR values
for PSK, QAM, and FSK constellations.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


@dataclass
class DemodulationResult:
    bits: np.ndarray                 # Hard decision bitstream (uint8: 0 or 1)
    llrs: np.ndarray                 # Soft Log-Likelihood Ratios (float32)
    constellation_points: np.ndarray # 1 SPS synchronized constellation points
    evm_pct: float                   # Error Vector Magnitude (%)
    parameter: Parameter


class SymbolDemodulator:
    """
    Maximum Likelihood (ML) Symbol Demodulator for BPSK, QPSK, 8PSK, 16QAM, 64QAM.
    """

    # Standard Constellation Definitions (Unit Normalized)
    CONSTELLATIONS = {
        "BPSK": np.array([-1.0+0.0j, 1.0+0.0j], dtype=np.complex64),
        "QPSK": np.array([
            -1.0-1.0j, -1.0+1.0j, 1.0-1.0j, 1.0+1.0j
        ], dtype=np.complex64) / np.sqrt(2.0),
        "8PSK": np.exp(1j * np.arange(8) * np.pi / 4.0).astype(np.complex64),
        "16QAM": np.array([
            (x + 1j*y) for x in [-3, -1, 1, 3] for y in [-3, -1, 1, 3]
        ], dtype=np.complex64) / np.sqrt(10.0),
        "64QAM": np.array([
            (x + 1j*y) for x in [-7, -5, -3, -1, 1, 3, 5, 7] for y in [-7, -5, -3, -1, 1, 3, 5, 7]
        ], dtype=np.complex64) / np.sqrt(42.0)
    }

    BIT_MAPS = {
        "BPSK": np.array([[0], [1]], dtype=np.uint8),
        "QPSK": np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.uint8),
        "8PSK": np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=np.uint8),
        "16QAM": np.array([[i >> 3 & 1, i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(16)], dtype=np.uint8),
        "64QAM": np.array([[i >> 5 & 1, i >> 4 & 1, i >> 3 & 1, i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(64)], dtype=np.uint8)
    }

    @classmethod
    def demodulate(
        cls,
        buffer: SignalBuffer,
        modulation: str = "QPSK"
    ) -> DemodulationResult:
        if not buffer.is_valid():
            ev = Evidence(description="Buffer is invalid for demodulation.", source="DEMODULATOR", score=0.0)
            param = Parameter(name="demodulation", value=0, unit="bits", source="UNKNOWN", method=None, confidence=0.0, evidence=[ev])
            return DemodulationResult(
                bits=np.array([], dtype=np.uint8),
                llrs=np.array([], dtype=np.float32),
                constellation_points=np.array([], dtype=np.complex64),
                evm_pct=0.0,
                parameter=param
            )

        mod_upper = modulation.upper()
        if mod_upper not in cls.CONSTELLATIONS:
            mod_upper = "QPSK"  # Fallback

        constellation = cls.CONSTELLATIONS[mod_upper]
        bit_map = cls.BIT_MAPS[mod_upper]
        bps = bit_map.shape[1]

        samples = buffer.samples
        n_syms = len(samples)

        if n_syms == 0:
            ev = Evidence(description="Zero symbols provided for demodulation.", source="DEMODULATOR", score=0.0)
            param = Parameter(name="demodulation", value=0, unit="bits", source="UNKNOWN", method=None, confidence=0.0, evidence=[ev])
            return DemodulationResult(
                bits=np.array([], dtype=np.uint8),
                llrs=np.array([], dtype=np.float32),
                constellation_points=np.array([], dtype=np.complex64),
                evm_pct=0.0,
                parameter=param
            )

        # Normalize symbol power to unit variance for constellation mapping & EVM calculation
        p_avg = float(np.mean(np.abs(samples)**2))
        if p_avg > 1e-12:
            samples = (samples / np.sqrt(p_avg)).astype(np.complex64)

        # Resolve phase ambiguity by testing constellation phase offsets
        if mod_upper == "BPSK":
            phi0 = 0.5 * float(np.angle(np.mean(samples ** 2)))
            samples = (samples * np.exp(-1j * phi0)).astype(np.complex64)
            scale_real = float(np.mean(np.abs(samples.real)))
            if scale_real > 1e-6:
                samples = (samples / scale_real).astype(np.complex64)
            dists = np.abs(samples[:, None] - constellation[None, :])
            min_indices = np.argmin(dists, axis=1)
        elif mod_upper in ("QPSK", "8PSK"):
            angles = np.arange(8) * (np.pi / 4.0)
            best_angle = 0.0
            best_evm = 999.0
            best_min_indices = None

            for angle in angles:
                rot_samples = (samples * np.exp(-1j * angle)).astype(np.complex64)
                d = np.abs(rot_samples[:, None] - constellation[None, :])
                idx = np.argmin(d, axis=1)
                err = rot_samples - constellation[idx]
                evm = float(np.sqrt(np.mean(np.abs(err)**2)) * 100.0)
                if evm < best_evm:
                    best_evm = evm
                    best_angle = angle
                    best_min_indices = idx

            if best_min_indices is not None:
                samples = (samples * np.exp(-1j * best_angle)).astype(np.complex64)
                min_indices = best_min_indices
                dists = np.abs(samples[:, None] - constellation[None, :])
            else:
                dists = np.abs(samples[:, None] - constellation[None, :])
                min_indices = np.argmin(dists, axis=1)
        else:
            dists = np.abs(samples[:, None] - constellation[None, :])
            min_indices = np.argmin(dists, axis=1)

        # Map to hard bits
        bits = bit_map[min_indices].ravel()

        # Compute Error Vector Magnitude (EVM %)
        ideal_syms = constellation[min_indices]
        error_vectors = samples - ideal_syms
        p_error = np.mean(np.abs(error_vectors)**2)
        p_ideal = np.mean(np.abs(ideal_syms)**2) + 1e-12
        evm_pct = float(np.sqrt(p_error / p_ideal) * 100.0)

        # Calculate soft LLRs
        min_dists = np.min(dists, axis=1)
        llrs = (1.0 / (min_dists + 1e-6)).repeat(bps).astype(np.float32)

        ev = Evidence(
            description=f"Demodulated {len(bits)} bits for {mod_upper} (EVM: {evm_pct:.2f}%)",
            source="SYMBOL_DEMODULATOR",
            score=max(0.1, 1.0 - (evm_pct / 100.0)),
            metadata={"modulation": mod_upper, "bit_count": len(bits), "evm_pct": evm_pct}
        )

        param = Parameter(
            name="demodulation",
            value=int(len(bits)),
            unit="bits",
            source="ESTIMATED",
            method=f"ML_DEMOD_{mod_upper}",
            confidence=max(0.1, 1.0 - (evm_pct / 100.0)),
            evidence=[ev]
        )

        return DemodulationResult(
            bits=bits,
            llrs=llrs,
            constellation_points=samples,
            evm_pct=evm_pct,
            parameter=param
        )
