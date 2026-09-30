"""
Blind sample-rate hypotheses for raw .iq files (no metadata, no recording duration).

What the samples determine: the SYMBOL RATE RELATIVE TO THE SAMPLE RATE (samples/symbol),
from cyclostationary features:
    PSK / QAM   |x|^2 cyclic line (symbol-rate hypotheses scored by strobe compactness) refined by
                Oerder-Meyr on the matched-filter output (the PSK front end, psk_front_end)
    FSK / CPM   spectral lines of IF-transition transforms (estimate_rate)
The family is chosen by the classifier's FSK-vs-linear log-odds (unified_classifier.fsk_vs_linear).

What they cannot determine: the absolute sample rate. A signal at 8 samples/symbol looks the
same at 9.6 kHz / 1200 Bd and at 1.92 MHz / 240 kBd. So absolute rates are HYPOTHESES:
    fs = (samples/symbol) x (a standard symbol rate)
ranked by (1) whether fs lands on a standard SDR / audio sample rate (within 1 %; the standard
value is then proposed) and (2) a
fixed preference order of common symbol rates. The result is marked ASSUMED in the provenance;
the recording-duration entry remains the exact method.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Common symbol rates (Bd), in preference order (earlier = preferred when several fit).
STANDARD_BAUDS = [1200, 2400, 4800, 9600, 300, 600, 19200, 38400, 10000, 50, 75, 100, 110, 150, 3600, 7200,
                  14400, 16000, 28800, 57600, 64000, 100000, 115200, 125000, 250000, 270833.33, 500000,
                  1000000, 2000000]
# Common sample rates (Hz): audio / modem rates and SDR rates.
STANDARD_RATES = [8000, 9600, 10000, 11025, 16000, 19200, 20000, 22050, 32000, 38400, 44100, 48000, 50000, 76800,
                  96000, 100000, 153600, 192000, 200000, 240000, 250000, 256000, 288000, 300000, 384000, 500000,
                  768000, 960000, 1000000, 1024000, 1200000, 1536000, 1800000, 1920000, 2000000, 2048000,
                  2400000, 2500000, 2560000, 3000000, 3200000, 4000000, 5000000, 6000000, 8000000, 10000000,
                  12500000, 20000000, 25000000, 30720000, 40000000, 50000000, 61440000]
MAX_SAMPLES = 1 << 18


MATCH_TOL = 0.01          # a measured sps on a few dozen symbols is typically within ~1 %


@dataclass
class RateHypothesis:
    sample_rate: float                   # proposed sample rate (the standard rate when one matches)
    symbol_rate: float
    standard_fs: Optional[float]         # the standard sample rate it matches (None = no match)
    measured_fs: float = 0.0             # sps x symbol rate before snapping

    def label(self) -> str:
        if self.standard_fs:
            return (f"{self.sample_rate:,.0f} Hz  (standard rate; symbol rate {self.symbol_rate:,.2f} Bd; "
                    f"measured {self.measured_fs:,.1f} Hz)")
        return f"{self.sample_rate:,.1f} Hz  (symbol rate {self.symbol_rate:,.2f} Bd; non-standard sample rate)"


@dataclass
class BlindRateEstimate:
    sps: Optional[float]
    family: str                           # "FSK" | "linear" | "unknown"
    method: str
    confidence: float
    hypotheses: List[RateHypothesis] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _hypotheses(sps: float, fs_min: float = 1e3, fs_max: float = 1e8) -> List[RateHypothesis]:
    out = []
    for rb in STANDARD_BAUDS:
        fs = sps * rb
        if not (fs_min <= fs <= fs_max):
            continue
        err = [abs(fs / f - 1) for f in STANDARD_RATES]
        k = int(np.argmin(err))
        std = float(STANDARD_RATES[k]) if err[k] < MATCH_TOL else None
        out.append(RateHypothesis(std if std else float(fs), float(rb), std, float(fs)))
    # standard sample rate first (stable: keeps the baud preference order inside each group)
    out.sort(key=lambda h: h.standard_fs is None)
    return out


def estimate(samples: np.ndarray) -> BlindRateEstimate:
    from dsp.conditioning import detect_burst, normalize_power
    from estimation.fsk_estimator import estimate_rate, inst_freq, occupied_band

    x = np.asarray(samples[:MAX_SAMPLES], dtype=np.complex128)
    if len(x) < 64:
        return BlindRateEstimate(None, "unknown", "too few samples", 0.0, notes=["fewer than 64 samples"])
    from modulation.unified_classifier import fsk_vs_linear
    logit, _, fev, _ = fsk_vs_linear(x)                     # the classifier's FSK-vs-linear log-odds
    x, _ = normalize_power(x)
    band = occupied_band(x)
    burst = detect_burst(x, band.center, band.bw99)
    notes = [f"FSK-vs-linear log-odds {logit:+.1f} (envelope, cumulants, IF histogram/kurtosis, cyclic and "
             f"M-th power lines)"]
    if logit > 0:
        from dsp.conditioning import channel_filter
        n = np.arange(len(x))
        y = channel_filter(x * np.exp(-2j * np.pi * band.center * n), 0.0, band.bw99, 0.6)
        f = inst_freq(y)
        r = estimate_rate(y[burst.start:burst.stop], f[burst.start:burst.stop - 1], band.bw99)
        sps, method, conf = float(r.sps), "FSK IF-transition spectral lines + decision-directed refinement", \
            float(np.clip(r.match, 0, 1))
        family = "FSK"
    else:
        from demodulation.psk_receiver import psk_front_end
        fe = psk_front_end(samples[:MAX_SAMPLES], None)
        sps = float(fe.sps)
        p = fe.params.get("sps")
        conf = float(p.confidence) if p is not None and p.confidence is not None else 0.5
        method = "|x|^2 cyclostationary line (-3 dB band prior, compactness-scored hypotheses) + Oerder-Meyr"
        family = "linear"
    hyps = _hypotheses(sps)
    notes.append(f"{len(hyps)} (sample rate, symbol rate) pairs consistent with {sps:.4f} samples/symbol; "
                 "the absolute rate cannot be measured from the samples alone")
    return BlindRateEstimate(sps, family, method, conf, hyps, notes)
