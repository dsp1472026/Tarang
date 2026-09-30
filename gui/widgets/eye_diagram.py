from gui.qt_compat import QWidget, QVBoxLayout
import numpy as np
import pyqtgraph as pg


class EyeDiagramWidget(QWidget):
    """
    Eye Diagram visualization plot widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title="Eye Diagram")
        self.plot_widget.setLabel("bottom", "Time (symbol periods)")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.layout.addWidget(self.plot_widget)

    def update_plot(self, samples: np.ndarray, samples_per_symbol: int = 8, num_traces: int = 200):
        self.plot_widget.clear()
        if samples is None or len(samples) == 0 or samples_per_symbol <= 1:
            return

        trace_len = 2 * samples_per_symbol
        total_traces = len(samples) // trace_len
        if total_traces <= 0:
            return

        display_traces = min(total_traces, num_traces)
        x = np.linspace(0, 2, trace_len)

        for i in range(display_traces):
            segment = samples[i * trace_len : (i + 1) * trace_len]
            # Plot real part (In-phase)
            self.plot_widget.plot(x, segment.real, pen=pg.mkPen(color=(37, 99, 235, 60), width=1.0))
