from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class SignalConfig:
    """
    Central configuration object for sample rate, frequency, dtype, and pipeline settings.
    Ensures a single source of truth for sample rate (fs) across all processing layers.
    """
    sample_rate: float
    center_frequency: float = 0.0
    dtype: str = "complex64"
    format_name: str = "IQ"
    threshold_db: float = 6.0
    byte_offset: int = 0
    endianness: str = "little"
    iq_order: str = "IQ"

    def __post_init__(self):
        if self.sample_rate is None or self.sample_rate <= 0:
            raise ValueError(f"Invalid SignalConfig sample_rate: {self.sample_rate}. Must be a positive float in Hz.")


@dataclass
class SignalBuffer:
    """
    Core data structure representing complex I/Q or real signal samples
    along with associated metadata and provenance.
    """
    samples: np.ndarray
    sample_rate: Optional[float] = None
    center_frequency: float = 0.0
    start_time: float = 0.0
    source_format: str = "UNKNOWN"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.samples is not None and not isinstance(self.samples, np.ndarray):
            self.samples = np.array(self.samples)

    def is_valid(self) -> bool:
        """
        Check whether the buffer contains non-empty, valid numerical samples.
        """
        if self.samples is None:
            return False
        if not isinstance(self.samples, np.ndarray):
            return False
        if self.samples.size == 0:
            return False
        if np.isnan(self.samples).any() or np.isinf(self.samples).any():
            return False
        return True

    def __len__(self) -> int:
        """
        Return the number of samples in the buffer.
        """
        if self.samples is None:
            return 0
        return len(self.samples)

    def slice(self, start: int, stop: Optional[int] = None) -> "SignalBuffer":
        """
        Return a sliced sub-buffer maintaining metadata and sample rate settings.
        """
        if self.samples is None:
            sliced_samples = np.array([], dtype=np.complex64)
        else:
            sliced_samples = self.samples[start:stop]

        return SignalBuffer(
            samples=sliced_samples,
            sample_rate=self.sample_rate,
            center_frequency=self.center_frequency,
            start_time=self.start_time + (start / self.sample_rate) if self.sample_rate and self.sample_rate > 0 else self.start_time,
            source_format=self.source_format,
            metadata=self.metadata.copy()
        )
