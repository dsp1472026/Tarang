import numpy as np
from scipy import signal as dsp_signal

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


class CenterFrequencyEstimator:
    """
    Estimates coarse center frequency offset from power spectral density centroid.
    """

    @classmethod
    def estimate(cls, buffer: SignalBuffer, nperseg: int = 2048) -> Parameter:
        if not buffer.is_valid():
            ev = Evidence(description="Buffer is invalid or empty.", source="CENTER_FREQ_ESTIMATOR", score=0.0)
            return Parameter(
                name="center_frequency",
                value=0.0,
                unit="Hz",
                source="UNKNOWN",
                method=None,
                confidence=0.0,
                evidence=[ev]
            )

        import logging
        logger = logging.getLogger(__name__)

        fs = buffer.sample_rate if buffer.sample_rate and buffer.sample_rate > 0 else 1.0
        is_real_signal = np.all(buffer.samples.imag == 0)
        win_seg = min(2048, len(buffer.samples))
        if is_real_signal:
            freqs, psd = dsp_signal.welch(
                buffer.samples.real,
                fs=fs,
                window="hann",
                nperseg=win_seg,
                scaling="density"
            )
            pos_mask = freqs > 10.0
            freqs = freqs[pos_mask]
            psd = psd[pos_mask]
        else:
            freqs, psd = dsp_signal.welch(
                buffer.samples,
                fs=fs,
                window="hann",
                nperseg=win_seg,
                noverlap=win_seg // 2,
                return_onesided=False,
                scaling="density"
            )
            freqs = np.fft.fftshift(freqs)
            psd = np.fft.fftshift(psd)

        # Band midpoint estimation for suppressed-carrier signals (-7dB / 20% peak threshold)
        peak_val = float(np.max(psd)) if len(psd) > 0 else 0.0
        mask = psd > 0.20 * peak_val if peak_val > 0 else (psd > 0)
        freqs_pass = freqs[mask] if np.any(mask) else freqs

        if len(freqs_pass) > 0:
            coarse_offset = float((freqs_pass[0] + freqs_pass[-1]) / 2.0)
        else:
            coarse_offset = 0.0

        peak_bin = int(np.argmax(psd))
        psd_peak_freq = float(freqs[peak_bin])

        est_offset = coarse_offset

        # 4th-power carrier line refinement for suppressed-carrier signals (e.g. QPSK / 4QAM)
        if len(buffer.samples) >= 256:
            try:
                t_arr = np.arange(len(buffer.samples)) / fs
                if is_real_signal:
                    dm_sig = 2.0 * buffer.samples.real * np.exp(-2j * np.pi * coarse_offset * t_arr)
                else:
                    dm_sig = buffer.samples * np.exp(-2j * np.pi * coarse_offset * t_arr)
                
                n_fft = min(len(dm_sig), 16384)
                sig_4th = (dm_sig[:n_fft]) ** 4
                fft_pwr = np.abs(np.fft.fftshift(np.fft.fft(sig_4th, n=n_fft)))
                fftfreqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / fs))
                
                mask_window = np.abs(fftfreqs) <= 200.0
                if np.any(mask_window):
                    pwr_win = fft_pwr.copy()
                    pwr_win[~mask_window] = 0.0
                    peak_idx = int(np.argmax(pwr_win))
                    mean_pwr = float(np.mean(fft_pwr)) + 1e-12
                    if pwr_win[peak_idx] > 5.0 * mean_pwr:
                        f_residual = float(fftfreqs[peak_idx] / 4.0)
                        est_offset += f_residual
            except Exception:
                pass

        logger.info(
            f"[DIAGNOSTIC] FFT frequency axis range: [{freqs[0]:.2f}, {freqs[-1]:.2f}] Hz, "
            f"PSD peak bin: {peak_bin}, Peak frequency: {psd_peak_freq:.2f} Hz, Centroid: {coarse_offset:.2f} Hz, Final: {est_offset:.2f} Hz."
        )

        total_center_freq = buffer.center_frequency + est_offset

        ev = Evidence(
            description=f"Spectral centroid center frequency estimated at {total_center_freq / 1e6:.4f} MHz (offset: {est_offset / 1e3:.2f} kHz).",
            source="SPECTRAL_CENTROID",
            score=0.85,
            metadata={"estimated_offset_hz": est_offset, "base_center_freq_hz": buffer.center_frequency}
        )

        return Parameter(
            name="center_frequency",
            value=total_center_freq,
            unit="Hz",
            source="ESTIMATED",
            method="SPECTRAL_CENTROID",
            confidence=0.85,
            evidence=[ev]
        )
