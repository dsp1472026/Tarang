"""
Tab 14 - Signals: every signal found in the capture (a capture can carry several transmissions on
different carriers), what each one was identified as and what it decoded to. Selecting a row shows
that signal in all the other tabs.
"""

import numpy as np
import pyqtgraph as pg
from scipy.signal import welch

from gui import theme
from gui.qt_compat import QGroupBox, QLabel, QSplitter, Qt, QVBoxLayout, QWidget, pyqtSignal
from gui.widgets.background import BackgroundRunner
from gui.widgets.tables import fill_table, make_table, wrap_label


def _fmt_hz(v):
    if v is None:
        return "-"
    return f"{v / 1e6:+.4f} MHz" if abs(v) >= 1e6 else f"{v / 1e3:+.3f} kHz" if abs(v) >= 1e3 else f"{v:+.1f} Hz"


def _bw_hz(v):
    return "-" if v is None else f"{v / 1e3:.2f} kHz" if v >= 1e3 else f"{v:.0f} Hz"


class SignalsTab(QWidget):
    signal_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results, self._bands = [], []
        self._current = 0
        self._runner = BackgroundRunner(self)          # capture spectrum off the GUI thread
        main = QVBoxLayout(self)
        self.header = wrap_label(QLabel("Signals: awaiting analysis"))
        self.header.setStyleSheet(f"font-weight: 600; font-size: 12pt; color: {theme.ACCENT};")
        main.addWidget(self.header)

        split = QSplitter(Qt.Orientation.Vertical)
        sb = QGroupBox("Capture spectrum (each detected signal shaded)")
        sl = QVBoxLayout(sb)
        self.spec = pg.PlotWidget()
        self.spec.setLabel("bottom", "Frequency", units="Hz")
        self.spec.setLabel("left", "PSD", units="dB")
        self.spec.showGrid(x=True, y=True, alpha=0.25)
        self.spec_curve = self.spec.plot(pen=pg.mkPen(theme.ACCENT_2, width=1.4))
        sl.addWidget(self.spec)
        split.addWidget(sb)
        tb = QGroupBox("Signals found")
        tl = QVBoxLayout(tb)
        self.table = make_table(["#", "Centre", "Bandwidth", "SNR", "Modulation", "Result"])
        self.table.itemSelectionChanged.connect(self._on_row)
        tl.addWidget(self.table)
        hint = wrap_label(QLabel("Select a signal to show it in every other tab."))
        hint.setStyleSheet(f"color: {theme.MUTED};")
        tl.addWidget(hint)
        split.addWidget(tb)
        split.setSizes([420, 300])
        main.addWidget(split)

    def update_signals(self, results, bands, capture_buffer=None):
        self._results, self._bands = list(results or []), list(bands or [])
        n = len(self._results)
        self.header.setText(f"Signals: {n} signal{'s' if n != 1 else ''} found in the capture" if n > 1 else
                            "Signals: one signal in the capture (analysed whole)")
        rows = []
        for i, r in enumerate(self._results):
            b = r.get("band")
            mod = r["parameters"]["Modulation"].value if "Modulation" in (r.get("parameters") or {}) else "-"
            if r.get("error"):
                result = f"error: {r['error']}"
            elif r.get("image") is not None:
                result = r["image"].describe()
            else:
                t = r.get("payload_text")
                result = (("CRC VALID: " if r.get("crc_valid") else "unverified: ") + repr(t[:60])) if t else "nothing decoded"
            rows.append([i + 1, _fmt_hz(b.center_hz) if b else "0 (whole)", _bw_hz(b.bandwidth_hz) if b else "-",
                         f"{b.snr_db:.1f} dB" if b else "-", str(mod), result])
        fill_table(self.table, rows, [bool(r.get("crc_valid") or r.get("image") is not None) for r in self._results])
        self._plot_spectrum(capture_buffer)
        if n:
            # the main window shows signal 1 itself; selecting it here must not trigger a second full
            # render of every tab (it doubled the time the window was busy)
            self.table.blockSignals(True)
            self.table.selectRow(0)
            self.table.blockSignals(False)
            self._current = 0

    def _plot_spectrum(self, buf):
        for it in list(self.spec.items()):
            if isinstance(it, (pg.LinearRegionItem, pg.TextItem)):
                self.spec.removeItem(it)
        if buf is None or buf.samples is None or not len(buf.samples):
            self.spec_curve.clear()
            return
        fs = buf.sample_rate or 1.0
        bands = list(self._bands)

        def compute():
            x = np.asarray(buf.samples[: 1 << 21])
            f, p = welch(x, fs=fs, nperseg=int(min(8192, max(256, len(x) // 16))), return_onesided=False)
            return np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(p) + 1e-30)

        self._runner.submit(compute, lambda fp: self._draw_spectrum(fp[0], fp[1], bands))

    def _draw_spectrum(self, f, p, bands):
        self.spec_curve.setData(f, p)
        for i, b in enumerate(bands):
            col = pg.mkColor(theme.SERIES[i % len(theme.SERIES)])
            fill = pg.mkColor(col)
            fill.setAlpha(55)
            reg = pg.LinearRegionItem([b.center_hz - b.bandwidth_hz / 2, b.center_hz + b.bandwidth_hz / 2],
                                      movable=False, brush=fill, pen=pg.mkPen(col))
            self.spec.addItem(reg)
            txt = pg.TextItem(f"{i + 1}", color=col, anchor=(0.5, 1.0))
            txt.setPos(b.center_hz, float(np.max(p)))
            self.spec.addItem(txt)

    def _on_row(self):
        r = self.table.currentRow()
        if 0 <= r < len(self._results):
            self._current = r
            self.signal_selected.emit(r)

    def select(self, i: int):
        if 0 <= i < self.table.rowCount() and self.table.currentRow() != i:
            self.table.selectRow(i)
