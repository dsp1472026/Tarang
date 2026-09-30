"""
Phase 2 Multi-Stage Asynchronous Pipeline Worker.
Executes Layers 1 through 6:
Layer 1: File Ingestion
Layer 2: FFT Energy Detection & SOI Isolation
Layer 3: Parameter Estimation (Fc, BW, Symbol Rate, SNR)
Layer 4: Hybrid Modulation Classification (RadioML ONNX + Cumulants + Constellation Geometry)
Layer 5: Synchronization (AGC, Costas Loop Carrier Recovery, Gardner Timing Recovery)
Layer 6: Symbol Demodulation & EVM Calculation
"""

import logging
from typing import Dict, Any
import numpy as np
import scipy.signal as dsp_signal

from core.signal_buffer import SignalBuffer, SignalConfig
from gui.qt_compat import QThread, pyqtSignal
from gui.workers.base_worker import BaseWorker

from ingestion.format_detector import FormatDetector
from detection.fft_detector import FFTDetector
from estimation.sample_rate import SampleRateEstimator
from estimation.center_frequency import CenterFrequencyEstimator
from estimation.bandwidth import BandwidthEstimator
from estimation.symbol_rate import SymbolRateEstimator
from estimation.snr import SNREstimator

from modulation.ml_classifier import HybridModulationClassifier
from synchronization.agc import AutomaticGainControl
from synchronization.costas_loop import CostasLoop
from synchronization.timing_recovery import SymbolTimingRecovery
from demodulation.demodulator import SymbolDemodulator

logger = logging.getLogger(__name__)


class Phase2PipelineWorker(BaseWorker):
    """
    Asynchronous QThread executing full Phase 1 + Phase 2 pipeline with
    centralized sample rate management and pre-demodulation sanity validation.
    """

    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent=parent)
        self.config = config
        self.classifier = HybridModulationClassifier()

    def work(self) -> Dict[str, Any]:
        self.progress.emit(5, "Layer 1: Reading signal file...")

        file_path = self.config.get("file_path", "")
        dtype = self.config.get("dtype", "complex64")
        sample_rate = self.config.get("sample_rate", None)
        center_frequency = float(self.config.get("center_frequency", 0.0))
        threshold_db = float(self.config.get("threshold_db", 6.0))

        # Layer 1: Ingestion
        buffer = FormatDetector.load_signal(
            file_path=file_path,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            dtype=dtype
        )

        if not buffer or not buffer.is_valid():
            raise ValueError(f"Failed to load valid signal buffer from file: {file_path}")

        # Centralize resolved sample rate in SignalConfig
        resolved_fs = buffer.sample_rate if (buffer.sample_rate and buffer.sample_rate > 0) else 1.0e6
        signal_cfg = SignalConfig(
            sample_rate=resolved_fs,
            center_frequency=buffer.center_frequency,
            dtype=dtype,
            format_name=buffer.source_format,
            threshold_db=threshold_db
        )

        logger.info(f"[DIAGNOSTIC] File Name          : {file_path}")
        logger.info(f"[DIAGNOSTIC] Resolved fs (Hz)    : {signal_cfg.sample_rate:,.2f}")
        logger.info(f"[DIAGNOSTIC] Sample Format       : {dtype}")
        logger.info(f"[DIAGNOSTIC] Total Complex Count : {len(buffer):,}")

        self.progress.emit(20, "Layer 2: Detecting Signals of Interest (SOIs)...")
        sois = FFTDetector.detect_sois(buffer, threshold_db_above_noise=threshold_db)

        self.progress.emit(35, "Layer 3: Estimating Parameters (Fc, BW, Baud, SNR)...")
        params = {}
        params["sample_rate"] = SampleRateEstimator.estimate(buffer)
        params["center_frequency"] = CenterFrequencyEstimator.estimate(buffer)
        params["bandwidth"] = BandwidthEstimator.estimate(buffer)
        params["snr"] = SNREstimator.estimate(buffer, bandwidth_hz=params["bandwidth"].value)

        # Downmix coarse center frequency offset to baseband using centralized fs
        total_fc = float(params["center_frequency"].value) if (params["center_frequency"] and params["center_frequency"].value is not None) else buffer.center_frequency
        fc_offset = total_fc - buffer.center_frequency
        proc_samples = buffer.samples
        proc_fs = signal_cfg.sample_rate

        # For 1D real passband signals (e.g. WAV files), perform quadrature downmixing + LPF
        is_real = np.all(buffer.samples.imag == 0)
        if is_real and abs(fc_offset) > 10.0:
            from scipy.signal import butter, filtfilt
            t = np.arange(len(buffer.samples)) / proc_fs
            raw_dm = 2.0 * buffer.samples.real * np.exp(-2j * np.pi * fc_offset * t)
            cutoff = min(proc_fs / 4.0, max(2000.0, abs(fc_offset) * 0.8))
            b, a = butter(4, cutoff / (proc_fs / 2.0), btype='low')
            i_filt = filtfilt(b, a, raw_dm.real)
            q_filt = filtfilt(b, a, raw_dm.imag)
            proc_samples = (i_filt + 1j * q_filt).astype(np.complex64)
        elif abs(fc_offset) > 10.0:
            t = np.arange(len(proc_samples)) / proc_fs
            proc_samples = (proc_samples * np.exp(-2j * np.pi * fc_offset * t)).astype(np.complex64)

        unmixed_buffer = SignalBuffer(
            samples=proc_samples,
            sample_rate=proc_fs,
            center_frequency=0.0,
            start_time=buffer.start_time,
            source_format=buffer.source_format,
            metadata=buffer.metadata
        )

        # Estimate symbol rate on downmixed baseband signal for maximum precision
        params["symbol_rate"] = SymbolRateEstimator.estimate(unmixed_buffer)
        symbol_rate_val = float(params["symbol_rate"].value) if (params["symbol_rate"] and params["symbol_rate"].value) else (proc_fs / 8.0)

        # Compute Samples Per Symbol (SPS)
        sps = proc_fs / symbol_rate_val if symbol_rate_val > 0 else 8.0

        # Downsample high-SPS signals (e.g. 48 kHz WAV files) to target SPS ~ 8 for neural network & DSP rules
        classifier_samples = proc_samples
        classifier_fs = proc_fs
        if sps >= 24.0 and len(classifier_samples) > 1024:
            factor = int(round(sps / 8.0))
            if factor > 1:
                classifier_samples = classifier_samples[::factor]
                classifier_fs = classifier_fs / factor

        classifier_buffer = SignalBuffer(
            samples=classifier_samples,
            sample_rate=classifier_fs,
            center_frequency=0.0,
            start_time=buffer.start_time,
            source_format=buffer.source_format,
            metadata=buffer.metadata
        )

        mod_param = self.classifier.classify(classifier_buffer)
        params["modulation"] = mod_param
        mod_type = str(mod_param.value)

        # Pre-Demodulation Diagnostic Logging & Sanity Checks
        logger.info(f"[DIAGNOSTIC] Estimated Symbol Rate: {symbol_rate_val:,.2f} Hz")
        logger.info(f"[DIAGNOSTIC] Computed SPS         : {sps:.4f}")
        logger.info(f"[DIAGNOSTIC] Classified Modulation: {mod_type}")

        sps_raw = proc_fs / symbol_rate_val
        if abs(sps_raw - round(sps_raw)) / sps_raw < 0.005:
            sps = float(round(sps_raw))
        else:
            sps = float(sps_raw)

        logger.info(f"[DIAGNOSTIC] Runtime SPS = fs / symbol_rate = {proc_fs:,.2f} / {symbol_rate_val:,.2f} = {sps_raw:.4f} (Used: {sps:.2f})")
        
        if sps < 1.5 or sps > 1000.0:
            raise ValueError(
                f"Demodulation Sanity Check Failed: Computed samples-per-symbol (SPS={sps:.2f}) "
                f"is out of acceptable range [1.5, 1000.0] for fs={proc_fs:,.0f} Hz and Baud={symbol_rate_val:,.0f} Hz."
            )

        span_symbols = 10
        rrc_taps_count = int(span_symbols * sps) + 1
        sampling_instants_count = int(len(unmixed_buffer.samples) / sps)
        logger.info(
            f"[DIAGNOSTIC] SPS = {sps:.2f}, Timing recovery step = 1/{sps:.2f}, "
            f"RRC filter taps = {rrc_taps_count}, Sampling instants count = {sampling_instants_count}"
        )

        self.progress.emit(70, "Layer 5: Synchronizing Signal (AGC, RRC Matched Filter, Gardner TED, Costas Loop)...")
        # 1. RRC Matched Filtering
        N = int(span_symbols * sps)
        tt = np.arange(-N / 2.0, N / 2.0 + 1.0) / float(sps)
        beta = 0.35
        num = np.sin(np.pi * tt * (1.0 - beta)) + 4.0 * beta * tt * np.cos(np.pi * tt * (1.0 + beta))
        den = np.pi * tt * (1.0 - (4.0 * beta * tt)**2)
        h = np.where(np.abs(tt) < 1e-8, 1.0 - beta + 4.0 * beta / np.pi, num / (den + 1e-12))
        h = h / np.sqrt(np.sum(h**2))
        mf_samples = dsp_signal.convolve(unmixed_buffer.samples, h, mode="same")
        unmixed_buffer = SignalBuffer(
            samples=mf_samples.astype(np.complex64),
            sample_rate=unmixed_buffer.sample_rate,
            center_frequency=unmixed_buffer.center_frequency,
            start_time=unmixed_buffer.start_time,
            source_format=unmixed_buffer.source_format,
            metadata=unmixed_buffer.metadata
        )

        # 2. AGC
        agc_res = AutomaticGainControl.process(unmixed_buffer)
        # 2. Gardner Timing Recovery to 1 SPS
        timing_res = SymbolTimingRecovery.process(agc_res.buffer, samples_per_symbol=float(sps))
        # 3. Costas Loop at 1 SPS with modulation-aware loop order
        mod_order = 4
        if "BPSK" in mod_type or "ASK" in mod_type:
            mod_order = 2
        elif "8PSK" in mod_type:
            mod_order = 8
        elif "QPSK" in mod_type or "QAM" in mod_type:
            mod_order = 4
        costas_res = CostasLoop.process(timing_res.buffer, order=mod_order)

        self.progress.emit(85, "Layer 6: Demodulating Symbols & Calculating EVM...")
        demod_res1 = SymbolDemodulator.demodulate(costas_res.buffer, modulation=mod_type)
        demod_res2 = SymbolDemodulator.demodulate(timing_res.buffer, modulation=mod_type)

        # 4. Energy-Peak Strobe Synchronization Fallback
        demod_res_strobe = None
        strobe_samples = unmixed_buffer.samples
        if sps >= 1.5 and len(strobe_samples) >= int(sps):
            int_sps = int(round(sps))
            energies = [np.mean(np.abs(strobe_samples[m::int_sps])**2) for m in range(min(int_sps, len(strobe_samples)))]
            best_m = int(np.argmax(energies)) if len(energies) > 0 else 0
            syms_strobe = strobe_samples[best_m::int_sps]
            strobe_buffer = SignalBuffer(
                samples=syms_strobe,
                sample_rate=symbol_rate_val,
                center_frequency=0.0,
                start_time=buffer.start_time,
                source_format=buffer.source_format,
                metadata=buffer.metadata
            )
            demod_res_strobe = SymbolDemodulator.demodulate(strobe_buffer, modulation=mod_type)

        candidates = [r for r in [demod_res1, demod_res2, demod_res_strobe] if r is not None and len(r.bits) > 0]
        demod_res = min(candidates, key=lambda r: r.evm_pct) if candidates else demod_res1

        self.progress.emit(100, "Phase 2 Pipeline Complete.")

        return {
            "buffer": buffer,
            "sois": sois,
            "parameters": params,
            "agc_result": agc_res,
            "timing_result": timing_res,
            "costas_result": costas_res,
            "demod_result": demod_res
        }
