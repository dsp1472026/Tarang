import numpy as np
from scipy import signal as dsp_signal

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


class BandwidthEstimator:
    """
    Estimates occupied bandwidth (OBW 99% power container or X-dB power drop).
    """

    @classmethod
    def estimate(
        cls,
        buffer: SignalBuffer,
        nperseg: int = 1024,
        power_percentage: float = 0.999
    ) -> Parameter:
        import logging
        logger = logging.getLogger(__name__)

        if not buffer.is_valid():
            ev = Evidence(description="Buffer is invalid.", source="BANDWIDTH_ESTIMATOR", score=0.0)
            return Parameter(
                name="bandwidth",
                value=0.0,
                unit="Hz",
                source="UNKNOWN",
                method=None,
                confidence=0.0,
                evidence=[ev]
            )

        fs = buffer.sample_rate if buffer.sample_rate and buffer.sample_rate > 0 else 1.0
        nperseg = max(16, min(nperseg, len(buffer.samples)))
        freqs, psd = dsp_signal.welch(
            buffer.samples,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            return_onesided=False,
            scaling="density"
        )
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)

        # 1. 99.9% Cumulative power distribution
        cum_power = np.cumsum(psd)
        total_power = cum_power[-1]

        if total_power <= 0:
            return Parameter(
                name="bandwidth",
                value=0.0,
                unit="Hz",
                source="UNKNOWN",
                method="OBW",
                confidence=0.0,
                evidence=[Evidence(description="Zero total spectral power", source="OBW", score=0.0)]
            )

        cum_power_norm = cum_power / total_power
        lower_tail = (1.0 - power_percentage) / 2.0
        upper_tail = 1.0 - lower_tail

        idx_lower = np.searchsorted(cum_power_norm, lower_tail)
        idx_upper = np.searchsorted(cum_power_norm, upper_tail)

        idx_lower = min(idx_lower, len(freqs) - 1)
        idx_upper = min(idx_upper, len(freqs) - 1)
        bw_999 = float(abs(freqs[idx_upper] - freqs[idx_lower]))

        # 2. 20 dB-down threshold method (PSD >= 0.01 * PSD_max)
        psd_peak = np.max(psd)
        threshold_20db = 0.01 * psd_peak
        above_thresh = np.where(psd >= threshold_20db)[0]
        if len(above_thresh) > 0:
            bw_20db = float(abs(freqs[above_thresh[-1]] - freqs[above_thresh[0]]))
        else:
            bw_20db = bw_999

        reported_bw = max(bw_999, bw_20db)

        logger.info(
            f"[DIAGNOSTIC] Bandwidth methods: 99.9% = {bw_999:.2f} Hz, "
            f"20dB-down = {bw_20db:.2f} Hz, reported = {reported_bw:.2f} Hz."
        )

        ev = Evidence(
            description=f"Estimated Bandwidth (max of 99.9% OBW and 20dB-down): {reported_bw / 1e3:.2f} kHz.",
            source="BANDWIDTH_ESTIMATOR_HYBRID",
            score=0.90,
            metadata={"bw_999_hz": bw_999, "bw_20db_hz": bw_20db, "reported_bw_hz": reported_bw}
        )

        return Parameter(
            name="bandwidth",
            value=reported_bw,
            unit="Hz",
            source="ESTIMATED",
            method="BANDWIDTH_HYBRID_999_20DB",
            confidence=0.90,
            evidence=[ev]
        )
