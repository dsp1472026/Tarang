from typing import Dict, Any, Optional
from gui.workers.base_worker import BaseWorker
from ingestion.format_detector import FormatDetector
from detection.fft_detector import FFTDetector
from estimation.sample_rate import SampleRateEstimator
from estimation.center_frequency import CenterFrequencyEstimator
from estimation.bandwidth import BandwidthEstimator
from estimation.snr import SNREstimator
from estimation.symbol_rate import SymbolRateEstimator


class Phase1PipelineWorker(BaseWorker):
    """
    Worker executing Phase 1 pipeline tasks (Ingestion -> SOI Detection -> Parameter Estimation)
    in a background thread.
    """

    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = config

    def work(self) -> Dict[str, Any]:
        file_path = self.config.get("file_path")
        if not file_path:
            raise ValueError("No file path provided to worker configuration.")

        # Step 1: Ingestion
        self.progress.emit(10, f"Loading signal file: {file_path}...")
        if self.is_cancelled:
            return {}

        buffer = FormatDetector.load_signal(
            file_path=file_path,
            force_format=self.config.get("force_format"),
            sample_rate=self.config.get("sample_rate"),
            center_frequency=self.config.get("center_frequency", 0.0),
            dtype=self.config.get("dtype"),
            endianness=self.config.get("endianness", "little"),
            iq_order=self.config.get("iq_order", "IQ"),
            byte_offset=self.config.get("byte_offset", 0),
            max_samples=self.config.get("max_samples")
        )

        if self.is_cancelled:
            return {}

        # Step 2: Signal Detection (Welch FFT)
        self.progress.emit(40, "Running FFT Signal of Interest (SOI) detection...")
        thresh_db = self.config.get("threshold_db", 6.0)
        sois = FFTDetector.detect_sois(buffer, threshold_db_above_noise=thresh_db)

        if self.is_cancelled:
            return {}

        # Step 3: Parameter Estimation
        self.progress.emit(70, "Extracting signal parameters (sample rate, Fc, BW, SNR, Symbol Rate)...")
        param_sample_rate = SampleRateEstimator.estimate(buffer)
        param_center_freq = CenterFrequencyEstimator.estimate(buffer)
        param_bw = BandwidthEstimator.estimate(buffer)
        param_snr = SNREstimator.estimate(buffer)
        param_sym_rate = SymbolRateEstimator.estimate(buffer)

        parameters = {
            "sample_rate": param_sample_rate,
            "center_frequency": param_center_freq,
            "bandwidth": param_bw,
            "snr": param_snr,
            "symbol_rate": param_sym_rate
        }

        self.progress.emit(100, "Phase 1 pipeline completed successfully.")

        return {
            "buffer": buffer,
            "sois": sois,
            "parameters": parameters
        }
