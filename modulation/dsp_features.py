"""
DSP Feature Extractor for Modulation Classification.
Extracts high-order cumulants, instantaneous phase/amplitude statistics, PAPR,
and constellation geometry (radial rings and phase symmetry).
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np


@dataclass
class DSPFeatures:
    """
    Extracted DSP & Constellation Geometry Features.
    """
    c20: complex
    c21: float
    c40: complex
    c41: complex
    c42: float
    c60: complex
    c40_c42_ratio: float
    gamma_max: float             # Normalized amplitude variance
    phase_var: float             # Unwrapped phase variance
    inst_freq_var: float         # Instantaneous frequency variance
    inst_freq_kurtosis: float    # Instantaneous frequency kurtosis
    papr_db: float               # Peak-to-Average Power Ratio (dB)
    spectral_flatness: float
    ring_count: int              # Number of detected radial constellation rings
    phase_symmetry_fold: int     # 2, 4, or 8 fold phase symmetry
    radius_ratio: float          # Rotation-invariant max(|x|) / median(|x|)
    phase4_var: float            # Variance of 4th-power phase arg(x^4)
    cluster_count: int           # K-means estimated cluster count (2, 4, 8, 16, 64)
    amplitude_peaks: int         # 1D amplitude histogram peak count
    freq_peaks: int              # 1D instantaneous frequency histogram peak count
    metadata: Dict[str, Any]


class DSPFeatureExtractor:
    """
    Extracts statistical, cumulant, and constellation geometry features from IQ samples.
    """

    @classmethod
    def extract_cumulants(cls, samples: np.ndarray) -> Tuple[complex, float, complex, complex, float, complex, float]:
        """
        Calculates high-order cumulants: C20, C21, C40, C41, C42, C60, and |C40|/|C42| ratio after DC removal and unit power normalization.
        """
        import logging
        logger = logging.getLogger(__name__)

        if len(samples) < 10:
            return 0j, 0.0, 0j, 0j, 0.0, 0j, 0.0

        if len(samples) > 4000:
            samples = samples[:4000]

        # 1. Remove DC offset
        x_dc = samples - np.mean(samples)

        # 2. Normalize to unit average power
        p_avg = float(np.mean(np.abs(x_dc)**2))
        if p_avg <= 0:
            return 0j, 0.0, 0j, 0j, 0.0, 0j, 0.0

        x = x_dc / np.sqrt(p_avg)

        # 3. Compute normalized moments & cumulants
        m10 = np.mean(x)
        m20 = np.mean(x**2)
        m21 = np.mean(np.abs(x)**2)
        m40 = np.mean(x**4)
        m41 = np.mean((x**3) * np.conj(x))
        m42 = np.mean(np.abs(x)**4)
        m60 = np.mean(x**6)

        c20 = complex(m20 - (m10**2))
        c21 = float(np.real(m21 - np.abs(m10)**2))
        c40 = complex(m40 - 3.0 * (m20**2))
        c41 = complex(m41 - 3.0 * m20 * m21)
        c42 = float(np.real(m42 - np.abs(m20)**2 - 2.0 * (m21**2)))
        c60 = complex(m60 - 15.0 * m40 * m20 + 30.0 * (m20**3))

        ratio = float(np.abs(c40) / (abs(c42) + 1e-12))

        logger.info(
            f"[DIAGNOSTIC] Cumulants: |C20|={abs(c20):.4f}, |C40|={abs(c40):.4f}, |C41|={abs(c41):.4f}, |C42|={abs(c42):.4f}."
        )

        return c20, c21, c40, c41, c42, c60, ratio

    @classmethod
    def extract_constellation_geometry(cls, samples: np.ndarray) -> Tuple[int, int, float, float, int, int, int]:
        """
        Extracts rotation-invariant constellation geometry features:
        1. Ring count via radial histogram peak detection.
        2. Phase symmetry fold (2, 4, 8) via M-fold phase histogram analysis.
        3. Radius ratio: max(|x|) / median(|x|).
        4. 4th-power phase variance: var(arg(x^4)).
        5. Cluster count: K-means inertia drop estimation for k in {2, 4, 8, 16}.
        6. Amplitude histogram peak count & Instantaneous Frequency peak count.
        """
        if len(samples) < 32:
            return 1, 4, 1.0, 0.0, 2, 1, 1

        mag = np.abs(samples)
        mag_norm = mag / (np.mean(mag) + 1e-12)

        # 1. Radius ratio (rotation-invariant max / median amplitude)
        med_mag = float(np.median(mag))
        radius_ratio = float(np.max(mag) / (med_mag + 1e-12)) if med_mag > 0 else 1.0

        # 2. 4th-power phase variance (rotation-invariant phase dispersion)
        phase4 = np.angle(samples**4)
        phase4_var = float(np.var(phase4))

        # 3. Radial ring count & amplitude histogram peaks
        counts, _ = np.histogram(mag_norm, bins=30, range=(0.1, 2.5))
        peaks = 0
        for i in range(1, len(counts) - 1):
            if counts[i] > counts[i - 1] and counts[i] > counts[i + 1] and counts[i] > 0.04 * len(samples):
                peaks += 1
        ring_count = max(1, min(peaks, 9))
        amplitude_peaks = max(1, peaks)

        # 4. Phase symmetry fold (M-fold rotation) after 4th-power phase alignment
        angles = np.angle(samples)
        fold_scores = {}
        for m in [2, 4, 8]:
            m_angles = np.mod(m * angles, 2.0 * np.pi)
            hist, _ = np.histogram(m_angles, bins=20, range=(0.0, 2.0 * np.pi))
            score = float(np.max(hist) - np.mean(hist))
            fold_scores[m] = score
        best_fold = max(fold_scores.keys(), key=lambda k: fold_scores[k])

        # 5. K-Means Constellation Cluster Count Estimation
        cluster_count = 2
        try:
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.cluster import KMeans
            # Take representative sub-sample of up to 400 points
            sub_n = min(len(samples), 400)
            sub_samples = samples[:sub_n]
            sub_data = np.column_stack([sub_samples.real, sub_samples.imag])
            n_unique = len(np.unique(sub_samples))
            inertias = {}
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                for k in [2, 4, 8, 16]:
                    if n_unique >= k:
                        km = KMeans(n_clusters=k, max_iter=10, n_init=1, random_state=42).fit(sub_data)
                        inertias[k] = float(km.inertia_)
                    else:
                        inertias[k] = 0.0

            # Select k where inertia drops near baseline noise level (<= 4.5)
            best_k = 2
            for k in [2, 4, 8, 16]:
                if k in inertias and inertias[k] <= 4.5:
                    best_k = k
                    break
            if best_k == 2 and inertias.get(2, 0.0) > 20.0:
                if inertias.get(4, 99.0) <= 20.0:
                    best_k = 4
                elif inertias.get(8, 99.0) <= 20.0:
                    best_k = 8
                else:
                    best_k = 16
            cluster_count = best_k
        except Exception:
            cluster_count = 2

        # 6. Instantaneous Frequency Peak Count for FSK
        inst_freq = np.angle(samples[1:] * np.conj(samples[:-1]))
        f_counts, _ = np.histogram(inst_freq, bins=25)
        freq_peaks = 0
        for i in range(1, len(f_counts) - 1):
            if f_counts[i] > f_counts[i - 1] and f_counts[i] > f_counts[i + 1] and f_counts[i] > 0.05 * len(inst_freq):
                freq_peaks += 1
        freq_peaks = max(1, freq_peaks)

        return ring_count, best_fold, radius_ratio, phase4_var, cluster_count, amplitude_peaks, freq_peaks

    @classmethod
    def extract(cls, samples: np.ndarray) -> DSPFeatures:
        """
        Extracts full set of DSP and constellation geometry features.
        """
        if len(samples) == 0:
            return DSPFeatures(
                c20=0j, c21=0.0, c40=0j, c41=0j, c42=0.0, c60=0j, c40_c42_ratio=0.0,
                gamma_max=0.0, phase_var=0.0, inst_freq_var=0.0, inst_freq_kurtosis=0.0,
                papr_db=0.0, spectral_flatness=0.0, ring_count=1, phase_symmetry_fold=4,
                radius_ratio=1.0, phase4_var=0.0, cluster_count=2, amplitude_peaks=1, freq_peaks=1,
                metadata={}
            )

        c20, c21, c40, c41, c42, c60, ratio = cls.extract_cumulants(samples)

        mag = np.abs(samples)
        p_avg = np.mean(mag**2) + 1e-12
        p_peak = np.max(mag**2)
        papr_db = float(10.0 * np.log10(p_peak / p_avg))

        # Normalized amplitude variance
        mag_norm = mag / np.sqrt(p_avg)
        gamma_max = float(np.var(mag_norm))

        # Instantaneous phase variance
        phase = np.unwrap(np.angle(samples))
        phase_var = float(np.var(phase - np.mean(phase)))

        # Instantaneous frequency
        inst_freq = np.angle(samples[1:] * np.conj(samples[:-1]))
        if len(inst_freq) > 5:
            inst_freq_var = float(np.var(inst_freq))
            mean_f = np.mean(inst_freq)
            std_f = np.std(inst_freq) + 1e-12
            inst_freq_kurtosis = float(np.mean(((inst_freq - mean_f) / std_f)**4) - 3.0)
        else:
            inst_freq_var = 0.0
            inst_freq_kurtosis = 0.0

        # Spectral flatness
        fft_mag = np.abs(np.fft.fft(samples))
        fft_mag = np.maximum(fft_mag, 1e-12)
        g_mean = float(np.exp(np.mean(np.log(fft_mag))))
        a_mean = float(np.mean(fft_mag))
        spectral_flatness = float(g_mean / a_mean) if a_mean > 0 else 0.0

        ring_count, fold, radius_ratio, phase4_var, cluster_count, amp_peaks, freq_peaks = cls.extract_constellation_geometry(samples)

        metadata = {
            "c20_mag": float(np.abs(c20)),
            "c40_mag": float(np.abs(c40)),
            "c41_mag": float(np.abs(c41)),
            "c42_mag": float(abs(c42)),
            "c40_c42_ratio": ratio,
            "gamma_max": gamma_max,
            "papr_db": papr_db,
            "ring_count": ring_count,
            "phase_symmetry_fold": fold,
            "radius_ratio": radius_ratio,
            "phase4_var": phase4_var,
            "cluster_count": cluster_count,
            "amplitude_peaks": amp_peaks,
            "freq_peaks": freq_peaks
        }

        return DSPFeatures(
            c20=c20,
            c21=c21,
            c40=c40,
            c41=c41,
            c42=c42,
            c60=c60,
            c40_c42_ratio=ratio,
            gamma_max=gamma_max,
            phase_var=phase_var,
            inst_freq_var=inst_freq_var,
            inst_freq_kurtosis=inst_freq_kurtosis,
            papr_db=papr_db,
            spectral_flatness=spectral_flatness,
            ring_count=ring_count,
            phase_symmetry_fold=fold,
            radius_ratio=radius_ratio,
            phase4_var=phase4_var,
            cluster_count=cluster_count,
            amplitude_peaks=amp_peaks,
            freq_peaks=freq_peaks,
            metadata=metadata
        )
