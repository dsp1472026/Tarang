"""
Main GUI Window for SIH 26147 Signal Analysis Application.
Fully integrated with Phase 1 & Phase 2 Pipeline Processing.
"""

import sys
from typing import Dict, Any

from gui.qt_compat import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QMessageBox, QApplication, Qt, QLabel, QComboBox
)
from gui import theme
from gui.qt_compat import QtWidgets

from gui.tabs.input_tab import InputTab
from gui.tabs.detection_tab import DetectionTab
from gui.tabs.parameters_tab import ParametersTab
from gui.tabs.modulation_tab import ModulationTab
from gui.tabs.sync_tab import SyncTab
from gui.tabs.demodulation_tab import DemodulationTab
from gui.tabs.conditioning_tab import ConditioningTab
from gui.tabs.bitstream_tab import BitstreamTab
from gui.tabs.deinterleave_tab import DeinterleaveTab
from gui.tabs.fec_tab import FECTab
from gui.tabs.framing_tab import FramingTab
from gui.tabs.report_tab import ReportTab
from gui.tabs.image_tab import ImageTab
from gui.tabs.signals_tab import SignalsTab
from gui.widgets.progress_panel import ProgressPanel
from gui.workers.unified_worker import UnifiedPipelineWorker


class MainWindow(QMainWindow):
    """
    Main GUI Window for SIH 26147 Signal Analysis System (.IQ / .wav).
    Supports multi-stage processing via UnifiedPipelineWorker (blind PSK/FSK receivers + unified classifier).
    """

    def __init__(self):
        theme.apply_theme(QApplication.instance())      # before any widget / plot is built
        super().__init__()
        self.setWindowTitle("SIH 26147 — Automated Signal Analysis System (.IQ / .wav)")
        self.resize(1360, 860)

        self.current_worker: UnifiedPipelineWorker = None
        self._results = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # Header: title + signal selector (shown when a capture carries several signals)
        head = QHBoxLayout()
        title = QLabel("SIH 26147  \u00b7  Automated Signal Analysis")
        title.setStyleSheet(f"font-size: 15pt; font-weight: 700; color: {theme.ACCENT};")
        subtitle = QLabel(".IQ / .wav  \u2192  detection \u2192 classification \u2192 demodulation \u2192 decoding")
        subtitle.setStyleSheet(f"color: {theme.MUTED}; padding-left: 10px;")
        subtitle.setMinimumWidth(10)          # natural width when there is room; may clip, never widens the window
        head.addWidget(title)
        head.addWidget(subtitle)
        head.addStretch(1)
        self.signal_label = QLabel("Signal:")
        self.signal_label.setStyleSheet(f"color: {theme.WARN}; font-weight: 600;")
        self.signal_combo = QComboBox()
        self.signal_combo.setMinimumWidth(380)
        self.signal_combo.currentIndexChanged.connect(self._on_signal_combo)
        head.addWidget(self.signal_label)
        head.addWidget(self.signal_combo)
        self.signal_label.hide()
        self.signal_combo.hide()
        main_layout.addLayout(head)

        # Tab Widget
        self.tabs = QTabWidget()

        # Instantiate Tabs
        self.input_tab = InputTab()
        self.detection_tab = DetectionTab()
        self.parameters_tab = ParametersTab()
        self.modulation_tab = ModulationTab()
        self.sync_tab = SyncTab()
        self.demodulation_tab = DemodulationTab()
        self.conditioning_tab = ConditioningTab()
        self.bitstream_tab = BitstreamTab()
        self.deinterleave_tab = DeinterleaveTab()
        self.fec_tab = FECTab()
        self.framing_tab = FramingTab()
        self.report_tab = ReportTab()
        self.image_tab = ImageTab()
        self.signals_tab = SignalsTab()

        # Add Active Phase 1, Phase 2, and Phase 3 Tabs
        self._add_tab(self.input_tab, "1. Input && Setup")
        self._add_tab(self.detection_tab, "2. Signal Detection")
        self._add_tab(self.parameters_tab, "3. Parameters")
        self._add_tab(self.modulation_tab, "4. Modulation")
        self._add_tab(self.sync_tab, "5. Sync")
        self._add_tab(self.demodulation_tab, "6. Demodulation")
        self._add_tab(self.conditioning_tab, "7. Conditioning")
        self._add_tab(self.bitstream_tab, "8. Bitstream")

        # Phase 4: layers 8-11 + report
        self._add_tab(self.deinterleave_tab, "9. De-interleaving")
        self._add_tab(self.fec_tab, "10. FEC")
        self._add_tab(self.framing_tab, "11. Framing")
        self._add_tab(self.report_tab, "12. Report")
        self._add_tab(self.image_tab, "13. Image")
        self._add_tab(self.signals_tab, "14. Signals")
        self.signals_tab.signal_selected.connect(self._select_signal)

        main_layout.addWidget(self.tabs)

        # Progress Panel
        self.progress_panel = ProgressPanel()
        self.progress_panel.cancel_requested.connect(self.cancel_worker)
        main_layout.addWidget(self.progress_panel)

        # Connect Signals
        self.input_tab.ingest_requested.connect(self.run_pipeline)

    def _add_tab(self, widget, title: str):
        """Add a tab inside a scroll area (see above); self._pages maps the tab to its page."""
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setWidget(widget)
        if not hasattr(self, "_pages"):
            self._pages = {}
        self._pages[widget] = area
        self.tabs.addTab(area, title)

    def run_pipeline(self, config: Dict[str, Any]):
        if self.current_worker and self.current_worker.isRunning():
            QMessageBox.warning(self, "Worker Running", "A pipeline analysis task is already in progress.")
            return

        self.progress_panel.start_task("Starting Multi-Stage Analysis Pipeline...")

        self.current_worker = UnifiedPipelineWorker(config)
        self.current_worker.progress.connect(self.progress_panel.update_progress)
        self.current_worker.finished.connect(self.on_pipeline_finished)
        self.current_worker.error.connect(self.on_pipeline_error)
        self.current_worker.cancelled.connect(self.on_pipeline_cancelled)

        self.current_worker.start()

    def cancel_worker(self):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            self.progress_panel.status_label.setText("Cancelling worker...")

    def on_pipeline_finished(self, results: Dict[str, Any]):
        self.progress_panel.finish_task("Analysis Pipeline Complete.")
        if not results:
            return
        self._results = results
        signals = results.get("signals") or [results]
        multi = len(signals) > 1
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        for i, r in enumerate(signals):
            mod = r["parameters"]["Modulation"].value if "Modulation" in (r.get("parameters") or {}) else "?"
            self.signal_combo.addItem(f"{r.get('label') or f'Signal {i + 1}'}  \u2014  {mod}")
        self.signal_combo.blockSignals(False)
        self.signal_label.setVisible(multi)
        self.signal_combo.setVisible(multi)
        self.input_tab.display_buffer_summary(results.get("capture_buffer") or results.get("buffer"))
        self.signals_tab.update_signals(signals, results.get("bands") or [],
                                        results.get("capture_buffer") or results.get("buffer"))
        self._show_result(signals[0])
        # Land on the most informative tab
        if multi:
            self.tabs.setCurrentWidget(self._pages[self.signals_tab])
        elif signals[0].get("image") is not None:
            self.tabs.setCurrentWidget(self._pages[self.image_tab])
        else:
            self.tabs.setCurrentWidget(self._pages[self.detection_tab])

    def _on_signal_combo(self, i: int):
        self._select_signal(i)

    def _select_signal(self, i: int):
        signals = (self._results or {}).get("signals") or []
        if not 0 <= i < len(signals):
            return
        if self.signal_combo.currentIndex() != i:
            self.signal_combo.blockSignals(True)
            self.signal_combo.setCurrentIndex(i)
            self.signal_combo.blockSignals(False)
        self.signals_tab.select(i)
        self._show_result(signals[i])

    def _show_result(self, results: Dict[str, Any]):
        """Fill every tab from one signal's result."""
        if results.get("error"):
            for t in (self.deinterleave_tab, self.fec_tab, self.framing_tab):
                t.clear(f"analysis failed: {results['error']}")
            self.demodulation_tab.clear(f"Analysis of this signal failed: {results['error']}")
            self.image_tab.update_image(None)
            return
        buffer = results.get("buffer")
        sois = results.get("sois", [])
        parameters = results.get("parameters", {})
        agc_res = results.get("agc_result")
        costas_res = results.get("costas_result")
        timing_res = results.get("timing_result")
        demod_res = results.get("demod_result")

        # Update GUI tabs across all layers (detection: always the whole capture and all its signals)
        whole = (self._results or {}).get("capture_buffer") or buffer
        self.detection_tab.update_data(whole, (self._results or {}).get("capture_sois", sois))
        self.parameters_tab.update_parameters(parameters)

        mod_param = parameters.get("Modulation") or parameters.get("modulation")
        if mod_param:
            self.modulation_tab.update_modulation(mod_param, demod_res.constellation_points if demod_res else None)

        if agc_res or costas_res or timing_res:
            self.sync_tab.update_sync(agc_res, costas_res, timing_res)
        else:                                  # FSK / QAM: no PSK carrier & timing stage this run
            fam = results.get("classification").family if results.get("classification") else ""
            self.sync_tab.clear_sync("FSK: rate, timing and tone model are in the Parameters tab" if fam == "FSK"
                                     else "No carrier/timing synchronisation for this signal"
                                     )

        cls = results.get("classification")
        self.demodulation_tab.update_payload(results.get("payload_text"), bool(results.get("crc_valid")),
                                             str(parameters["Modulation"].value) if "Modulation" in parameters
                                             else (cls.modulation if cls else ""))
        post = results.get("postdemod")
        if demod_res:
            self.demodulation_tab.update_demodulation(demod_res)
            self.conditioning_tab.set_bitstream(demod_res.bits)
            if post is not None:
                self.bitstream_tab.update_postdemod(post)
            else:
                self.bitstream_tab.set_bitstream(demod_res.bits, demod_res.llrs)
        else:                                  # clear the previous run's bits
            self.demodulation_tab.clear("No demodulator for this modulation yet.")
            self.conditioning_tab.set_bitstream(None)
            self.bitstream_tab.set_bitstream(None)
        if post is not None:
            self.deinterleave_tab.update_postdemod(post)
            self.fec_tab.update_postdemod(post)
            self.framing_tab.update_postdemod(post, results.get("payload_text"))
        else:
            image = results.get("image")
            for t in (self.deinterleave_tab, self.fec_tab, self.framing_tab):
                t.clear(f"file frame recovered directly ({image.describe()}) - see the Image tab" if image is not None
                        else "not run for this signal")
        self.image_tab.update_image(results.get("image"), (results.get("config") or {}).get("file_path", ""))
        rep = dict(results)
        if self._results is not None:
            rep["signals"] = self._results.get("signals")
        self.report_tab.update_results(rep)

    def on_pipeline_error(self, err_msg: str):
        self.progress_panel.finish_task("Error during processing.")
        QMessageBox.critical(self, "Pipeline Error", f"An error occurred during signal analysis:\n{err_msg}")

    def on_pipeline_cancelled(self):
        self.progress_panel.finish_task("Operation cancelled by user.")
        QMessageBox.information(self, "Cancelled", "Analysis operation was cancelled.")


def main():
    app = QApplication(sys.argv)
    theme.apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
