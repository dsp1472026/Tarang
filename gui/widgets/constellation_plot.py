from gui.qt_compat import QWidget, QVBoxLayout
import numpy as np
import pyqtgraph as pg

from core.signal_buffer import SignalBuffer


class ConstellationPlotWidget(QWidget):
    """
    Scatter plot widget displaying IQ complex constellation diagram.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget(title="Constellation Diagram")
        self.plot_widget.setLabel("bottom", "In-Phase (I)")
        self.plot_widget.setLabel("left", "Quadrature (Q)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setAspectLocked(True)
        self.layout.addWidget(self.plot_widget)

        self.scatter = pg.ScatterPlotItem(size=4, pen=pg.mkPen(None), brush=pg.mkBrush(37, 99, 235, 140))
        self.plot_widget.addItem(self.scatter)

    def update_plot(self, samples: np.ndarray, max_points: int = 5000):
        if samples is None or len(samples) == 0:
            self.scatter.clear()
            return

        if len(samples) > max_points:
            # Decimate or random sample for display speed
            idx = np.random.choice(len(samples), max_points, replace=False)
            sub_samples = samples[idx]
        else:
            sub_samples = samples

        i_vals = sub_samples.real
        q_vals = sub_samples.imag

        self.scatter.setData(x=i_vals, y=q_vals)
