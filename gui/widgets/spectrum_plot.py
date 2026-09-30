from gui.qt_compat import QWidget, QVBoxLayout
import numpy as np
import pyqtgraph as pg
from scipy import signal as dsp_signal

from core.signal_buffer import SignalBuffer


class SpectrumPlotWidget(QWidget):
    """
    PyQtGraph-based Power Spectral Density (PSD) Spectrum plot widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title="Power Spectral Density (PSD)")
        self.plot_widget.setLabel("bottom", "Frequency", units="Hz")
        self.plot_widget.setLabel("left", "Power", units="dB")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plot_widget)

        self.curve = self.plot_widget.plot(pen=pg.mkPen(color="#2563EB", width=1.5))
        self.threshold_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(color="#DC2626", style=pg.QtCore.Qt.PenStyle.DashLine))
        self.plot_widget.addItem(self.threshold_line)
        self.threshold_line.hide()

    def update_plot(self, buffer: SignalBuffer, nperseg: int = 1024, threshold_db: float = None):
        self.apply(self.compute(buffer, nperseg), threshold_db)

    @staticmethod
    def compute(buffer: SignalBuffer, nperseg: int = 1024):
        """(frequencies, PSD dB) or None - pure numpy, safe to run off the GUI thread."""
        if buffer is None or not buffer.is_valid():
            return None
        fs = buffer.sample_rate if buffer.sample_rate and buffer.sample_rate > 0 else 1.0
        nperseg = max(16, min(nperseg, len(buffer)))
        noverlap = nperseg // 2
        freqs, psd = dsp_signal.welch(
            buffer.samples,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False,
            scaling="density"
        )
        freqs = np.fft.fftshift(freqs) + buffer.center_frequency
        psd_db = 10.0 * np.log10(np.maximum(np.fft.fftshift(psd), 1e-18))
        return freqs, psd_db

    def apply(self, data, threshold_db: float = None):
        """Show a compute() result (GUI thread; fast)."""
        if data is None:
            self.curve.clear()
            return
        freqs, psd_db = data
        self.curve.setData(freqs, psd_db)

        if threshold_db is not None:
            self.threshold_line.setValue(threshold_db)
            self.threshold_line.show()
        else:
            self.threshold_line.hide()
