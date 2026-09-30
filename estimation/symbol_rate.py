"""
Symbol Rate Estimation via Conjugate Differential Product, Phase Derivative Feature,
FFT-accelerated Unbiased Autocorrelation, Parabolic Sub-sample Refinement,
and Composite Evidence Confidence Scoring.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.signal import fftconvolve

from core.signal_buffer import SignalBuffer
from core.evidence import Parameter, Evidence


@dataclass
class SymbolRateConfig:
    max_sps: int = 64                 # Search up to this many samples/symbol
    min_sps: int = 2
    peak_threshold_ratio: float = 0.35
    prominence_window: int = 3        # Local prominence window (samples)
    min_prominence_ratio: float = 0.15
    clip_phase: float = np.pi / 2.0    # Robustify phase derivative feature
    sub_block_count: int = 8          # Sub-block count for consistency verification
    fft_correlation: bool = True      # FFT-based correlation for large buffers
    fft_correlation_min_len: int = 32768


class SymbolRateEstimator:
    """
    Production-grade symbol rate estimator utilizing conjugate phase transitions,
    parabolic peak interpolation, sub-block consistency, and spectral line validation.
    """

    @classmethod
    def estimate(
        cls,
        buffer: SignalBuffer,
        config: Optional[SymbolRateConfig] = None
    ) -> Parameter:
        if config is None:
            config = SymbolRateConfig()

        if not buffer.is_valid() or buffer.sample_rate is None or buffer.sample_rate <= 0:
            ev = Evidence(
                description="Cannot estimate symbol rate without a valid sample rate.",
                source="SYMBOL_RATE_ESTIMATOR",
                score=0.0
            )
            return Parameter(
                name="symbol_rate",
                value=0.0,
                unit="Baud",
                source="UNKNOWN",
                method=None,
                confidence=0.0,
                evidence=[ev]
            )

        fs = float(buffer.sample_rate)
        samples = np.asarray(buffer.samples, dtype=np.complex64)
        n = samples.size

        if n < 32:
            ev = Evidence(
                description="Signal buffer too short for symbol rate estimation.",
                source="SYMBOL_RATE_ESTIMATOR",
                score=0.0,
                metadata={"n_samples": n}
            )
            return Parameter(
                name="symbol_rate",
                value=0.0,
                unit="Baud",
                source="UNKNOWN",
                method=None,
                confidence=0.0,
                evidence=[ev]
            )

        # ------------------------------------------------------------------
        # 1. Conjugate differential product with amplitude weighting
        # ------------------------------------------------------------------
        z = samples[1:] * np.conj(samples[:-1])
        a = np.abs(z)
        a_mean = float(np.mean(a)) + 1e-12
        z_weighted = z * (a / a_mean)

        # ------------------------------------------------------------------
        # 2. Carrier frequency offset removal via circular mean phase
        # ------------------------------------------------------------------
        phase_inc = np.angle(z_weighted)
        psi = float(np.angle(np.mean(np.exp(1j * phase_inc))))
        z_demod = z_weighted * np.exp(-1j * psi)

        # ------------------------------------------------------------------
        # 3. Compute Phase Derivative Feature & Magnitude Transition Feature
        # ------------------------------------------------------------------
        dphi = np.angle(z_demod)
        dphi = np.clip(dphi, -config.clip_phase, config.clip_phase)
        env_phase = dphi ** 2.0
        env_phase = env_phase - np.mean(env_phase)

        # Magnitude derivative feature for complex modulations (QPSK, 8PSK, QAM)
        mag_sq = np.abs(samples)**2
        dmag = np.diff(mag_sq)
        env_mag = dmag ** 2.0
        env_mag = env_mag - np.mean(env_mag) if env_mag.size > 0 else env_phase

        # ------------------------------------------------------------------
        # 4. Unbiased Autocorrelation for Phase & Magnitude Features
        # ------------------------------------------------------------------
        tau_max = min(config.max_sps, env_phase.size // 2)
        if tau_max < config.min_sps:
            ev = Evidence(
                description="Buffer too short for configured max SPS search range.",
                source="SYMBOL_RATE_ESTIMATOR",
                score=0.0
            )
            return Parameter(
                name="symbol_rate",
                value=0.0,
                unit="Baud",
                source="UNKNOWN",
                method=None,
                confidence=0.0,
                evidence=[ev]
            )

        r_phase = cls._unbiased_autocorrelation(
            env_phase,
            tau_max=tau_max,
            use_fft=config.fft_correlation and env_phase.size >= config.fft_correlation_min_len
        )
        r_mag = cls._unbiased_autocorrelation(
            env_mag,
            tau_max=tau_max,
            use_fft=config.fft_correlation and env_mag.size >= config.fft_correlation_min_len
        )

        # Find peaks for both phase & magnitude features
        peak_phase = cls._find_first_significant_peak(r_phase, config.peak_threshold_ratio, config.prominence_window, config.min_prominence_ratio)
        peak_mag = cls._find_first_significant_peak(r_mag, config.peak_threshold_ratio, config.prominence_window, config.min_prominence_ratio)

        # Select feature envelope that gives higher autocorrelation peak prominence
        prom_phase = float(r_phase[peak_phase]) if peak_phase else 0.0
        prom_mag = float(r_mag[peak_mag]) if peak_mag else 0.0

        if prom_mag > 1.2 * prom_phase and peak_mag is not None:
            env = env_mag
            r_full = r_mag
            peak_lag = peak_mag
        else:
            env = env_phase
            r_full = r_phase
            peak_lag = peak_phase if peak_phase is not None else peak_mag

        if peak_lag is None:
            search = r_full[config.min_sps:]
            if search.size == 0:
                ev = Evidence(
                    description="No significant peak found in autocorrelation.",
                    source="SYMBOL_RATE_ESTIMATOR",
                    score=0.10
                )
                return Parameter(
                    name="symbol_rate",
                    value=0.0,
                    unit="Baud",
                    source="UNKNOWN",
                    method=None,
                    confidence=0.10,
                    evidence=[ev]
                )
            peak_lag = int(np.argmax(search)) + config.min_sps

        # ------------------------------------------------------------------
        # 6. Sub-sample SPS refinement via parabolic interpolation & FFT line validation
        # ------------------------------------------------------------------
        sps_refined = cls._parabolic_peak_refine(r_full, peak_lag)
        est_sym_rate = float(fs / sps_refined) if sps_refined > 0 else 0.0

        # FFT spectral line cross-validation
        if env.size >= 64:
            fft_win = np.hanning(env.size)
            spec = np.abs(np.fft.rfft(env * fft_win))
            fft_freqs = np.fft.rfftfreq(env.size, d=1.0 / fs)
            min_freq = fs / float(config.max_sps)
            max_freq = fs / float(config.min_sps)
            freq_mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
            if np.any(freq_mask):
                spec_sel = spec[freq_mask]
                freqs_sel = fft_freqs[freq_mask]
                best_idx = np.argmax(spec_sel)
                fft_peak_freq = freqs_sel[best_idx]
                fft_peak_val = spec_sel[best_idx]
                median_spec = np.median(spec) + 1e-12
                if fft_peak_val / median_spec > 3.5:
                    fft_sps = fs / fft_peak_freq
                    if abs(fft_sps - sps_refined) <= 1.5 or abs(round(fft_sps) - round(sps_refined)) == 0:
                        sps_refined = fft_sps
                        est_sym_rate = float(fft_peak_freq)

        # ------------------------------------------------------------------
        # 7. Composite Evidence-Driven Confidence
        # ------------------------------------------------------------------
        confidence, evidence_list = cls._compute_confidence(
            env=env,
            sps_refined=sps_refined,
            r_full=r_full,
            config=config,
            sample_rate=fs
        )

        return Parameter(
            name="symbol_rate",
            value=est_sym_rate,
            unit="Baud",
            source="ESTIMATED",
            method="CONJUGATE_DIFFERENTIAL_AUTOCORRELATION",
            confidence=confidence,
            evidence=evidence_list
        )

    @staticmethod
    def _unbiased_autocorrelation(
        env: np.ndarray,
        tau_max: int,
        use_fft: bool
    ) -> np.ndarray:
        n = env.size
        if use_fft:
            r = fftconvolve(env, env[::-1], mode="full")[n - 1: 2 * n - 1]
        else:
            r = np.correlate(env, env, mode="full")[n - 1: 2 * n - 1]

        r = r[: tau_max + 1]
        denom = (n - np.arange(tau_max + 1)).astype(np.float64)
        denom[denom <= 0] = 1.0
        return r / denom

    @staticmethod
    def _find_first_significant_peak(
        r: np.ndarray,
        threshold_ratio: float,
        prominence_window: int,
        min_prominence_ratio: float
    ) -> Optional[int]:
        if r.size < 3:
            return None

        # Exclude tau=0 (zero-lag total variance) to evaluate threshold on non-zero lag peaks
        min_lag = 2
        max_lag = r.size - 1
        if max_lag <= min_lag:
            return None

        # Find all local peaks for tau in [min_lag, max_lag - 1]
        local_peaks = []
        for tau in range(min_lag, max_lag):
            if r[tau] > r[tau - 1] and r[tau] > r[tau + 1]:
                local_peaks.append(tau)

        if not local_peaks:
            return None

        peak_max = max(float(r[tau]) for tau in local_peaks)
        if peak_max <= 0:
            return None

        # Select first fundamental peak meeting threshold relative to peak_max
        abs_thresh = max(0.25 * peak_max, threshold_ratio * peak_max)
        for tau in local_peaks:
            if r[tau] >= abs_thresh:
                return tau

        return local_peaks[0]

    @staticmethod
    def _parabolic_peak_refine(r: np.ndarray, tau: int) -> float:
        if tau <= 0 or tau >= r.size - 1:
            return float(tau)

        y0, y1, y2 = float(r[tau - 1]), float(r[tau]), float(r[tau + 1])
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) < 1e-12:
            return float(tau)
        delta = 0.5 * (y0 - y2) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
        return float(tau + delta)

    @classmethod
    def _compute_confidence(
        cls,
        env: np.ndarray,
        sps_refined: float,
        r_full: np.ndarray,
        config: SymbolRateConfig,
        sample_rate: float
    ) -> Tuple[float, List[Evidence]]:
        evidence: List[Evidence] = []

        # 1. Peak prominence score
        r_max = float(np.max(r_full)) + 1e-12
        peak_int = int(round(sps_refined))
        if 1 <= peak_int < r_full.size:
            local_lo = max(0, peak_int - 2)
            local_hi = min(r_full.size, peak_int + 3)
            local_min = float(np.min(r_full[local_lo:local_hi]))
            prominence = float(np.clip((r_full[peak_int] - local_min) / r_max, 0.0, 1.0))
        else:
            prominence = 0.0

        evidence.append(Evidence(
            description=f"Autocorrelation peak prominence ratio: {prominence:.2f}",
            source="SYMBOL_RATE_PROMINENCE",
            score=prominence,
            metadata={"sps_lag": peak_int, "prominence_ratio": prominence}
        ))

        # 2. Sub-block consistency score
        consistency = cls._sub_block_consistency(env, sps_refined, config)
        evidence.append(Evidence(
            description=f"Sub-block SPS consistency score: {consistency:.2f}",
            source="SYMBOL_RATE_CONSISTENCY",
            score=consistency,
            metadata={"sub_blocks": config.sub_block_count, "consistency_score": consistency}
        ))

        # 3. Independent FFT spectral line check
        fft_score = cls._fft_line_score(env, sps_refined)
        evidence.append(Evidence(
            description=f"FFT spectral line validation score: {fft_score:.2f}",
            source="SYMBOL_RATE_FFT_LINE",
            score=fft_score,
            metadata={"fft_score": fft_score}
        ))

        # Composite score weighting (0.45 prominence + 0.35 consistency + 0.20 FFT line)
        composite = float(np.clip(0.45 * prominence + 0.35 * consistency + 0.20 * fft_score, 0.0, 1.0))

        evidence.append(Evidence(
            description=f"Estimated Symbol Rate: {sample_rate / sps_refined / 1e3:.2f} kBaud (SPS: {sps_refined:.2f}, Confidence: {composite * 100:.1f}%)",
            source="SYMBOL_RATE_ESTIMATOR",
            score=composite,
            metadata={
                "symbol_rate_baud": sample_rate / sps_refined,
                "samples_per_symbol": sps_refined,
                "confidence": composite
            }
        ))

        return composite, evidence

    @classmethod
    def _sub_block_consistency(
        cls,
        env: np.ndarray,
        sps_refined: float,
        config: SymbolRateConfig
    ) -> float:
        k = max(2, config.sub_block_count)
        n = env.size
        if n < k * 64:
            return 0.0

        block = n // k
        estimates: List[float] = []
        for i in range(k):
            seg = env[i * block: (i + 1) * block]
            if seg.size < 32:
                continue
            r = cls._unbiased_autocorrelation(
                seg,
                tau_max=min(config.max_sps, seg.size // 2),
                use_fft=False
            )
            p = cls._find_first_significant_peak(
                r,
                config.peak_threshold_ratio,
                config.prominence_window,
                config.min_prominence_ratio
            )
            if p is None:
                continue
            estimates.append(cls._parabolic_peak_refine(r, p))

        if len(estimates) < 3:
            return 0.0

        est = np.asarray(estimates, dtype=np.float64)
        rel_dev = np.abs(est - sps_refined) / max(sps_refined, 1e-6)
        within = float(np.mean(rel_dev <= 0.10))
        spread_penalty = float(np.clip(np.std(rel_dev) / 0.10, 0.0, 1.0))
        return float(np.clip(within * (1.0 - 0.5 * spread_penalty), 0.0, 1.0))

    @staticmethod
    def _fft_line_score(env: np.ndarray, sps_refined: float) -> float:
        if env.size < 256 or sps_refined <= 0:
            return 0.0

        e = env - np.mean(env)
        win = np.hanning(e.size)
        spec = np.abs(np.fft.rfft(e * win))
        freqs = np.fft.rfftfreq(e.size, d=1.0)

        target = 1.0 / sps_refined
        if target <= 0 or target >= 0.5:
            return 0.0

        bw = 0.02
        mask = np.abs(freqs - target) <= bw
        if not np.any(mask):
            return 0.0

        line_power = float(np.max(spec[mask]))
        median_power = float(np.median(spec)) + 1e-12
        ratio = line_power / median_power
        return float(np.clip((ratio - 1.0) / 49.0, 0.0, 1.0))
