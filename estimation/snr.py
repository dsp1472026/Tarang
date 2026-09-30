import numpy as np

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


class SNREstimator:
    """
    Estimates Signal-to-Noise Ratio (SNR) in dB using M2M4 moment estimator
    and spectral noise floor estimation.
    """

    @classmethod
    def estimate_m2m4(cls, samples: np.ndarray) -> float:
        """
        M2M4 moment estimator for SNR estimation of complex signals.
        """
        if len(samples) < 10:
            return 0.0

        r = np.abs(samples)
        m2 = float(np.mean(r**2))
        m4 = float(np.mean(r**4))

        if m2 <= 0:
            return 0.0

        diff = 2 * (m2**2) - m4
        if diff > 0:
            p_signal = np.sqrt(diff)
            p_noise = m2 - p_signal
            if p_noise > 1e-12 and p_signal > 0:
                snr_lin = p_signal / p_noise
                return float(10.0 * np.log10(max(snr_lin, 1e-3)))

        # Fallback for constant-modulus / low-noise / M2M4 failure
        mean_r = float(np.mean(r))
        var_r = float(np.var(r))

        if var_r < 1e-10:
            return 50.0  # High SNR clean signal

        if mean_r > 0:
            snr_lin = (mean_r**2) / var_r
            return float(10.0 * np.log10(max(snr_lin, 1e-3)))

        return 0.0

    @classmethod
    def estimate(cls, buffer: SignalBuffer, bandwidth_hz: float = None) -> Parameter:
        import logging
        logger = logging.getLogger(__name__)

        if not buffer.is_valid():
            ev = Evidence(description="Buffer is invalid.", source="SNR_ESTIMATOR", score=0.0)
            return Parameter(
                name="snr",
                value=0.0,
                unit="dB",
                source="UNKNOWN",
                method=None,
                confidence=0.0,
                evidence=[ev]
            )

        fs = buffer.sample_rate if buffer.sample_rate and buffer.sample_rate > 0 else 1.0
        samples = buffer.samples
        N = len(samples)

        if bandwidth_hz is None:
            bandwidth_hz = buffer.metadata.get("bandwidth_hz", fs * 0.1) if buffer.metadata else fs * 0.1

        bw_padded = min(fs * 0.95, float(bandwidth_hz * 1.2))
        
        from scipy import signal as dsp_signal
        
        nperseg = min(1024, N)
        freqs, psd = dsp_signal.welch(
            samples,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            return_onesided=False,
            scaling="density"
        )
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)

        # Center in-band mask at signal peak frequency
        peak_idx = int(np.argmax(psd))
        fc_peak = float(freqs[peak_idx])

        inband_mask = np.abs(freqs - fc_peak) <= (bw_padded / 2.0)
        outband_mask = ~inband_mask

        sig_power = float(np.mean(psd[inband_mask])) if np.any(inband_mask) else float(psd[peak_idx])
        noise_floor = float(np.median(psd[outband_mask])) if np.any(outband_mask) else 1e-12
        noise_floor = max(noise_floor, 1e-12)

        if sig_power > noise_floor:
            snr_calc = float(10.0 * np.log10((sig_power - noise_floor) / noise_floor))
        else:
            snr_calc = cls.estimate_m2m4(samples)

        # Clamp SNR to 40 dB max for clean synthetic signals
        snr_db = float(np.clip(snr_calc, -10.0, 40.0))

        logger.info(
            f"[DIAGNOSTIC] SNR estimation: signal_power={sig_power:.6e}, noise_power={noise_floor:.6e}, "
            f"bandwidth_used={bw_padded:.2f} Hz, snr_db={snr_db:.2f} dB."
        )

        ev = Evidence(
            description=f"Band-limited estimated SNR: {snr_db:.2f} dB.",
            source="BANDLIMITED_SNR_ESTIMATOR",
            score=min(1.0, max(0.2, (snr_db + 10) / 50.0)),
            metadata={"snr_db": snr_db, "bandwidth_used_hz": bw_padded, "signal_power": sig_power, "noise_power": noise_floor}
        )

        return Parameter(
            name="snr",
            value=snr_db,
            unit="dB",
            source="ESTIMATED",
            method="BANDLIMITED_SPECTRAL_ESTIMATOR",
            confidence=0.85,
            evidence=[ev]
        )
