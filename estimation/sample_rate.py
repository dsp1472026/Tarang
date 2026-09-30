from typing import Optional
from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


class SampleRateEstimator:
    """
    Estimates or extracts sample rate with provenance tracking.
    Never fabricates values if metadata or spectral estimation is unavailable.
    """

    @classmethod
    def estimate(cls, buffer: SignalBuffer) -> Parameter:
        if buffer.sample_rate is not None and buffer.sample_rate > 0:
            ev = Evidence(
                description=f"Sample rate {buffer.sample_rate / 1e6:.3f} MHz provided in file metadata.",
                source="METADATA_HEADER",
                score=1.0,
                metadata={"sample_rate": buffer.sample_rate}
            )
            return Parameter(
                name="sample_rate",
                value=buffer.sample_rate,
                unit="Hz",
                source="KNOWN",
                method="FILE_METADATA",
                confidence=1.0,
                evidence=[ev]
            )

        # Check metadata dictionary
        if "sample_rate" in buffer.metadata and buffer.metadata["sample_rate"] is not None:
            sr = float(buffer.metadata["sample_rate"])
            ev = Evidence(
                description=f"Sample rate {sr / 1e6:.3f} MHz extracted from metadata dictionary.",
                source="METADATA_DICT",
                score=1.0,
                metadata={"sample_rate": sr}
            )
            return Parameter(
                name="sample_rate",
                value=sr,
                unit="Hz",
                source="KNOWN",
                method="METADATA_PARSER",
                confidence=1.0,
                evidence=[ev]
            )

        # Cannot reliably estimate absolute sample rate from raw uncalibrated IQ bytes alone
        ev = Evidence(
            description="Absolute sample rate cannot be estimated without time/metadata context.",
            source="SAMPLE_RATE_ESTIMATOR",
            score=0.0,
            metadata={}
        )
        return Parameter(
            name="sample_rate",
            value=None,
            unit="Hz",
            source="UNKNOWN",
            method=None,
            confidence=0.0,
            evidence=[ev]
        )
