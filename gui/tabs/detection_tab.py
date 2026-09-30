from typing import List
from gui.qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QLabel, Qt
)

from gui.widgets.background import BackgroundRunner
from gui.widgets.spectrum_plot import SpectrumPlotWidget
from gui.widgets.waterfall_plot import WaterfallPlotWidget
from core.signal_buffer import SignalBuffer
from detection.fft_detector import DetectedSOI


class DetectionTab(QWidget):
    """
    Detection Tab: Displays Spectrum Plot, Waterfall Plot, and SOI Candidate Table.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner = BackgroundRunner(self)          # spectrum + spectrogram off the GUI thread
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Upper visualization layout: Spectrum & Waterfall
        viz_container = QWidget()
        viz_layout = QHBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)

        self.spectrum_widget = SpectrumPlotWidget()
        self.waterfall_widget = WaterfallPlotWidget()

        viz_layout.addWidget(self.spectrum_widget, 1)
        viz_layout.addWidget(self.waterfall_widget, 1)

        # Lower table layout: Detected Signals of Interest (SOIs)
        table_group = QGroupBox("Detected Signals of Interest (SOIs)")
        table_layout = QVBoxLayout(table_group)

        self.soi_table = QTableWidget(0, 6)
        self.soi_table.setHorizontalHeaderLabels([
            "SOI ID", "Center Freq (MHz)", "Bandwidth (kHz)", "Power (dB)", "SNR (dB)", "Evidence Confidence"
        ])
        self.soi_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        table_layout.addWidget(self.soi_table)

        splitter.addWidget(viz_container)
        splitter.addWidget(table_group)

        main_layout.addWidget(splitter)

    def update_data(self, buffer: SignalBuffer, sois: List[DetectedSOI], threshold_db: float = None):
        # A 2-4 M-sample capture took ~1 s of Welch + spectrogram on the GUI thread: computed in the
        # background now, drawn when ready (the SOI table below is filled at once)
        spec, fall = self.spectrum_widget, self.waterfall_widget
        self._runner.submit(lambda: (spec.compute(buffer), fall.compute(buffer)),
                            lambda res: (spec.apply(res[0], threshold_db), fall.apply(res[1])))

        self.soi_table.setRowCount(0)
        for soi in sois:
            row = self.soi_table.rowCount()
            self.soi_table.insertRow(row)

            score_str = f"{soi.evidence[0].score:.2f}" if soi.evidence else "N/A"

            self.soi_table.setItem(row, 0, QTableWidgetItem(str(soi.soi_id)))
            self.soi_table.setItem(row, 1, QTableWidgetItem(f"{soi.center_freq_hz / 1e6:.4f}"))
            self.soi_table.setItem(row, 2, QTableWidgetItem(f"{soi.bandwidth_hz / 1e3:.2f}"))
            self.soi_table.setItem(row, 3, QTableWidgetItem(f"{soi.power_db:.2f}"))
            self.soi_table.setItem(row, 4, QTableWidgetItem(f"{soi.snr_db:.2f}"))
            self.soi_table.setItem(row, 5, QTableWidgetItem(score_str))
