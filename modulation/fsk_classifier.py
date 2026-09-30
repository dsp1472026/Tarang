"""
L4 modulation analysis for the FSK family.

1. Family discrimination (FSK/CPM vs linear PSK/QAM):
   CPFSK/MSK/GMSK are constant-envelope at every sample; pulse-shaped PSK/QAM
   are not (RRC shaping gives an envelope CV of 0.3-0.5 even noiseless). The
   envelope CV, measured after band isolation, is compared with what the
   in-band SNR alone would produce:
       excess_cv = sqrt(max(cv^2 - cv_noise^2, 0)),  cv_noise ~ sqrt(1 / (2 SNR_inband))
   IQ imbalance adds envelope ripple, so the smaller of the raw and IQ-balanced
   CV is used. PAPR is not used.
2. Within-FSK naming from the estimated (M, h, pulse, BT, continuity) via
   dsp.cpm.fsk_variant_name (MSK / GMSK / GFSK / Sunde 2FSK / M-FSK).
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from core.evidence import Evidence
from estimation.fsk_estimator import Band, iq_balance

FSK_EXCESS_CV_MAX = 0.12      # below: constant envelope (FSK / CPM)
LINEAR_EXCESS_CV_MIN = 0.20   # above: envelope-modulated (pulse-shaped PSK / QAM)


@dataclass
class FamilyDecision:
    family: str               # "FSK" | "LINEAR" | "UNCERTAIN"
    p_fsk: float
    env_cv: float
    cv_noise: float
    excess_cv: float
    evidence: List[Evidence]


def classify_family(xb: np.ndarray, band: Band, noise_per_sample: float = None) -> FamilyDecision:
    """
    xb: the burst (DC removed). band: occupied_band() of the burst.
    noise_per_sample: mean |x|^2 of noise-only samples (from the guard), if any;
    otherwise the noise is taken from the PSD floor.
    The envelope is measured after band isolation (channel filter over the
    occupied band): this removes most of the noise but leaves the envelope of a
    pulse-shaped linear modulation intact.
    """
    from dsp.conditioning import channel_filter

    def cv(z):
        a = np.abs(z)
        e = max(1, len(a) // 20)                     # ignore filter transients at the edges
        a = a[e:-e] if len(a) > 4 * e else a
        return float(np.std(a) / (np.mean(a) + 1e-20))

    n = np.arange(len(xb))
    lo = np.exp(-2j * np.pi * band.center * n)
    xbal, _ = iq_balance(xb, threshold=0.0)
    # Wide isolation filter (+-1.0 x the 97 % band): narrower filters clip the spectral tails of
    # rectangular CPFSK and create envelope ripple at tone transitions.
    filtered = [channel_filter(z * lo, 0.0, band.bw99, 1.0) for z in (xb, xbal)]
    env_cv = min(cv(z) for z in filtered)
    if noise_per_sample is not None and np.isfinite(noise_per_sample):
        # the isolation filter passes a two-sided band of 2 x bw99 (cycles/sample)
        n_in = noise_per_sample * min(1.0, 2.0 * band.bw99)
        p_tot = float(np.mean(np.abs(filtered[0]) ** 2))
        snr_inband = max(p_tot - n_in, 1e-9) / max(n_in, 1e-12)
    else:
        snr_inband = 10 ** (band.snr_db / 10)
    cv_noise = float(np.sqrt(1.0 / (2.0 * max(snr_inband, 1e-3))))
    excess = float(np.sqrt(max(env_cv ** 2 - cv_noise ** 2, 0.0)))
    p_fsk = float(1.0 / (1.0 + np.exp((excess - 0.16) / 0.02)))
    if excess <= FSK_EXCESS_CV_MAX:
        fam = "FSK"
    elif excess >= LINEAR_EXCESS_CV_MIN:
        fam = "LINEAR"
    else:
        fam = "UNCERTAIN"
    ev = [Evidence(f"Envelope CV {env_cv:.3f} vs {cv_noise:.3f} expected from noise alone "
                   f"(in-band SNR {10 * np.log10(snr_inband):.1f} dB, noise from "
                   f"{'guard samples' if noise_per_sample is not None else 'PSD floor'}) -> excess {excess:.3f}: "
                   + {"FSK": "constant envelope (FSK/CPM)", "LINEAR": "envelope-modulated (PSK/QAM-like)",
                      "UNCERTAIN": "inconclusive"}[fam], "FAMILY_ENVELOPE", p_fsk if fam != "LINEAR" else 1 - p_fsk,
                   {"env_cv": env_cv, "cv_noise": cv_noise, "excess_cv": excess})]
    return FamilyDecision(fam, p_fsk, env_cv, cv_noise, excess, ev)
