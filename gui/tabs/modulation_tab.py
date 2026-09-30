from typing import Dict, Any, Optional
import numpy as np
import pyqtgraph as pg
from gui.qt_compat import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QTextEdit, QPushButton, QSplitter
from gui.widgets.tables import wrap_label
from gui.widgets.constellation_plot import ConstellationPlotWidget
from core.evidence import Parameter
from dsp.histogram_features import amplitude_histogram, count_amplitude_levels, count_phase_clusters, phase_histogram


class ModulationTab(QWidget):
    """
    GUI Tab displaying Layer 4 Modulation Classification results & Diagnostic Panel.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Summary Header
        self.header_label = wrap_label(QLabel("Modulation Classification: Awaiting Analysis..."))
        self.header_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #0F766E;")
        layout.addWidget(self.header_label)

        # 2. Confidence Bar
        conf_group = QGroupBox("Classification Confidence")
        conf_layout = QHBoxLayout(conf_group)
        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setTextVisible(True)
        conf_layout.addWidget(self.conf_bar)
        layout.addWidget(conf_group)

        # 3. Diagnostic Panel (Collapsible Box)
        self.diag_group = QGroupBox("Diagnostic Panel: Classifier Internal State && DSP Features")
        self.diag_group.setCheckable(True)
        self.diag_group.setChecked(True)
        diag_layout = QHBoxLayout(self.diag_group)

        # Left Column: Feature Metrics & ONNX Vector
        diag_text_layout = QVBoxLayout()
        self.diag_text = QTextEdit()
        self.diag_text.setReadOnly(True)
        self.diag_text.setStyleSheet("font-family: monospace; font-size: 11px; background-color: #FFFFFF; color: #1E293B;")
        diag_text_layout.addWidget(self.diag_text)
        diag_layout.addLayout(diag_text_layout, stretch=1)

        # Right Column: Constellation Plot
        self.const_plot = ConstellationPlotWidget()
        diag_layout.addWidget(self.const_plot, stretch=1)

        # Third Column: phase / amplitude histograms of the displayed symbols
        hist_layout = QVBoxLayout()
        self.phase_hist_plot = pg.PlotWidget(title="Phase histogram")
        self.phase_hist_plot.setLabel("bottom", "Phase (rad)")
        self.phase_hist_plot.setLabel("left", "Normalized density")
        self.phase_hist_plot.setXRange(-np.pi, np.pi)
        self.phase_hist_curve = self.phase_hist_plot.plot(pen=pg.mkPen(15, 118, 110), fillLevel=0, brush=(20, 184, 166, 50))
        self.amp_hist_plot = pg.PlotWidget(title="Amplitude histogram")
        self.amp_hist_plot.setLabel("bottom", "Amplitude")
        self.amp_hist_plot.setLabel("left", "Normalized density")
        self.amp_hist_curve = self.amp_hist_plot.plot(pen=pg.mkPen(37, 99, 235), fillLevel=0, brush=(37, 99, 235, 45))
        hist_layout.addWidget(self.phase_hist_plot)
        hist_layout.addWidget(self.amp_hist_plot)
        diag_layout.addLayout(hist_layout, stretch=1)

        layout.addWidget(self.diag_group)

        # 4. Evidence Table
        table_group = QGroupBox("Multi-Pillar Evidence Trail")
        table_layout = QVBoxLayout(table_group)
        self.evidence_table = QTableWidget()
        self.evidence_table.setColumnCount(4)
        self.evidence_table.setHorizontalHeaderLabels(["Pillar / Source", "Description", "Score", "Details"])
        self.evidence_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.evidence_table)
        layout.addWidget(table_group)

    def update_modulation(self, mod_param: Parameter, samples: Optional[np.ndarray] = None):
        if not mod_param:
            return

        mod_name = str(mod_param.value)
        conf_pct = int((mod_param.confidence or 0.0) * 100)

        self.header_label.setText(f"Detected Modulation: {mod_name}")
        self.conf_bar.setValue(conf_pct)

        # Update Constellation Plot if samples provided
        if samples is not None and len(samples) > 0:
            self.const_plot.update_plot(samples)
            self._update_histograms(samples)

        # Build Diagnostic Report
        diag_lines = [
            f"=== MODULATION CLASSIFIER DIAGNOSTIC REPORT ===",
            f"Detected Modulation (Top-1): {mod_name}",
            f"Composite Confidence: {conf_pct}%",
            f"Classification Source: {mod_param.source}",
            f"Method: {mod_param.method or 'N/A'}",
            "",
            "--- CUMULANT FEATURES vs THEORETICAL REFERENCES ---"
        ]

        ev_list = mod_param.evidence or []
        dsp_meta = {}
        for ev in ev_list:
            if ev.source == "DSP_CUMULANTS_GEOMETRY":
                dsp_meta = ev.metadata or {}
                break

        if dsp_meta:
            c20 = dsp_meta.get("c20_mag", 0.0)
            c40 = dsp_meta.get("c40_mag", 0.0)
            c41 = dsp_meta.get("c41_mag", 0.0)
            c42 = dsp_meta.get("c42_mag", 0.0)
            diag_lines.extend([
                f"|C20| = {c20:.4f}  (BPSK: 1.000, QPSK: 0.000)",
                f"|C40| = {c40:.4f}  (BPSK: 2.000, QPSK: 1.000, 8PSK: 0.000, 16QAM: 0.680, 64QAM: 0.620)",
                f"|C41| = {c41:.4f}  (BPSK: 2.000, QPSK: 0.000, 8PSK: 0.000)",
                f"|C42| = {c42:.4f}  (BPSK: 2.000, QPSK: 1.000, 8PSK: 1.000, 16QAM: 0.680, 64QAM: 0.620)",
                "",
                "--- ROTATION-INVARIANT CONSTELLATION GEOMETRY ---",
                f"Radius Ratio max(|x|)/med(|x|): {dsp_meta.get('radius_ratio', 1.0):.3f} (PSK: 1.00, 16QAM: 1.34, 64QAM: 1.45)",
                f"K-Means Cluster Count (k):     {dsp_meta.get('cluster_count', 2)}",
                f"Radial Ring Count:             {dsp_meta.get('ring_count', 1)}",
                f"Phase Symmetry Fold:           {dsp_meta.get('phase_symmetry_fold', 4)}-fold",
                f"Amplitude Histogram Peaks:     {dsp_meta.get('amplitude_peaks', 1)}",
                f"Instantaneous Freq Peaks:     {dsp_meta.get('freq_peaks', 1)}"
            ])
        else:
            diag_lines.append("No DSP feature metadata extracted.")

        diag_lines.extend([
            "",
            "--- ONNX MODEL & EVIDENCE SUMMARY ---"
        ])
        for ev in ev_list:
            diag_lines.append(f"[{ev.source}] {ev.description}")

        self.diag_text.setText("\n".join(diag_lines))

        # Update Evidence Table
        self.evidence_table.setRowCount(len(ev_list))
        for row, ev in enumerate(ev_list):
            self.evidence_table.setItem(row, 0, QTableWidgetItem(str(ev.source)))
            self.evidence_table.setItem(row, 1, QTableWidgetItem(str(ev.description)))
            self.evidence_table.setItem(row, 2, QTableWidgetItem(f"{ev.score * 100:.1f}%"))
            self.evidence_table.setItem(row, 3, QTableWidgetItem(str(ev.metadata)))

    def _update_histograms(self, samples: np.ndarray):
        try:
            ph, pb = phase_histogram(samples)
            ah, ab = amplitude_histogram(samples)
            nc, na = count_phase_clusters(samples), count_amplitude_levels(samples)
        except ValueError:          # empty or non-finite samples
            self.phase_hist_curve.clear()
            self.amp_hist_curve.clear()
            return
        self.phase_hist_curve.setData(pb, ph)
        self.amp_hist_curve.setData(ab, ah)
        self.phase_hist_plot.setTitle(f"Phase histogram ({nc} clusters)")
        self.amp_hist_plot.setTitle(f"Amplitude histogram ({na} levels)")
