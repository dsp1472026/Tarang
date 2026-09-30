from gui.qt_compat import QWidget, QVBoxLayout
import numpy as np
import pyqtgraph as pg
from scipy import signal as dsp_signal

from core.signal_buffer import SignalBuffer


class WaterfallPlotWidget(QWidget):
    """
    PyQtGraph-based Waterfall / Spectrogram display widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.graphics_view = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics_view)

        self.plot = self.graphics_view.addPlot(title="Spectrogram / Waterfall")
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Time", units="s")

        self.img = pg.ImageItem()
        self.plot.addItem(self.img)

        # Apply viridis/plasma color map
        colormap = pg.colormap.get("viridis")
        self.img.setLookupTable(colormap.getLookupTable(0.0, 1.0, 256))

    def update_plot(self, buffer: SignalBuffer, nperseg: int = 512, noverlap: int = 256):
        self.apply(self.compute(buffer, nperseg, noverlap))

    @staticmethod
    def compute(buffer: SignalBuffer, nperseg: int = 512, noverlap: int = 256):
        """(spectrogram dB, levels, rect) or None - pure numpy, safe to run off the GUI thread."""
        if buffer is None or not buffer.is_valid():
            return None

        fs = buffer.sample_rate if buffer.sample_rate and buffer.sample_rate > 0 else 1.0
        nperseg = max(16, min(nperseg, len(buffer)))
        if noverlap is None or noverlap >= nperseg:
            noverlap = nperseg // 2
        else:
            noverlap = min(noverlap, max(0, nperseg - 1))

        freqs, times, Sxx = dsp_signal.spectrogram(
            buffer.samples,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False,
            mode="psd"
        )

        freqs = np.fft.fftshift(freqs) + buffer.center_frequency
        Sxx = np.fft.fftshift(Sxx, axes=0)
        Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-18))

        # ImageItem is column-major (first axis = x): Sxx_db is (frequency, time), matching the
        # rectangle below (x = frequency, y = time). Passing Sxx_db.T drew the axes swapped.
        # Colour levels from percentiles: empty, filtered bands (-180 dB) otherwise flatten the contrast.
        lo, hi = np.percentile(Sxx_db, [20, 99.8])
        rect = ((float(freqs[0]), float(times[0]), float(freqs[-1] - freqs[0]), float(times[-1] - times[0]))
                if len(freqs) > 1 and len(times) > 1 else None)
        return Sxx_db.astype(np.float32), (float(lo), float(max(hi, lo + 1.0))), rect

    def apply(self, data):
        """Show a compute() result (GUI thread; fast)."""
        if data is None:
            self.img.clear()
            return
        Sxx_db, levels, rect = data
        self.img.setImage(Sxx_db, autoLevels=False, levels=levels)
        if rect is not None:
            self.img.setRect(pg.QtCore.QRectF(*rect))
